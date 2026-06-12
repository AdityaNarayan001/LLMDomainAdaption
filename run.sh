#!/usr/bin/env bash
# (3/3) RUN — self-healing, autonomous pipeline. Launch inside tmux on the GPU box:
#     tmux new -s hs './run.sh'
#
# PHASES (teacher and student NEVER coexist on the GPU — they don't fit together):
#   A) DATA-GEN  : serve the TEACHER alone -> generate SFT data -> stop teacher.
#                  SKIPPED when the SFT instruction set is already complete (no 52GB re-pull).
#   B) TRAIN     : CPT + SFT (no vLLM needed) -> merge -> models/student_merged.
#   C) RL/EVAL   : serve the MERGED student -> census gate -> flywheel (rollouts + eval).
# Gates AUTO-HALT with a logged reason. All output -> runs/pipeline_*.log.
#
# Env: MODEL (student base, Qwen2.5-Coder-7B), TEACHER (Qwen3.5-27B), ENDPOINT, CYCLES,
#      QUANT/QUANT_TEACHER (empty=bf16; modelopt_fp4 ONLY for pre-quantized checkpoints),
#      CENSUS_VERIFY.
set -uo pipefail
cd "$(dirname "$0")"

TS="$(date +%Y%m%d_%H%M%S)"; mkdir -p runs
LOG="runs/pipeline_${TS}.log"; exec > >(tee -a "$LOG") 2>&1

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B}"
TEACHER="${TEACHER:-Qwen/Qwen3.5-27B}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"
CYCLES="${CYCLES:-1000}"; CENSUS_VERIFY="${CENSUS_VERIFY:-25}"
# bf16 by default: modelopt_fp4 requires a PRE-QUANTIZED checkpoint (serve_champion.sh docs
# this) — stock Qwen3.5-27B is bf16, so an fp4 default killed Phase A at teacher serve.
QUANT="${QUANT:-}"; QUANT_TEACHER="${QUANT_TEACHER:-}"
export PREP_TEACHER="${PREP_TEACHER:-1}" MODEL TEACHER

echo "================ run ${TS}  student=$MODEL teacher=$TEACHER ================"
halt(){ echo ">>> HALT: $1"; stop_vllm; exit "${2:-1}"; }
VLLM_PID=""
stop_vllm(){ # kill the WHOLE vLLM tree — its spawned EngineCore workers don't die with the parent
             # and would otherwise orphan ~100GB of unified memory (starving CPT/SFT next).
  if [ -n "$VLLM_PID" ]; then
    pkill -9 -P "$VLLM_PID" 2>/dev/null
    kill -9 "$VLLM_PID" 2>/dev/null; wait "$VLLM_PID" 2>/dev/null
  fi
  pkill -9 -f "vllm.entrypoints.openai" 2>/dev/null   # sweep orphaned workers (safe: not in run.sh's cmdline)
  # veRL/Ray spawn rollout workers whose process *comm* is VLLM::Worker / EngineCore — pkill -f
  # (cmdline match) misses these, so they survive and orphan ~50GB of the unified pool, silently
  # starving the next stage. Reap them by comm (the run.sh ssh cmdline never matches these names).
  # raylet/gcs_server pin the Ray object store the same way after a crashed veRL run.
  for pid in $(ps -eo pid=,comm= | awk '/VLLM|EngineCore|raylet|gcs_server/ {print $1}'); do kill -9 "$pid" 2>/dev/null; done
  VLLM_PID=""; sleep 4; echo ">>> vLLM stopped"
}
serve_model(){ # $1=model $2=quant ; (re)launch vLLM alone, health-wait
  stop_vllm
  local q=(); [ -n "$2" ] && q=(--quantization "$2")
  echo ">>> serving $1 (${2:-bf16}) ..."
  # PATH must include the serve venv bin so flashinfer's JIT finds `ninja`; --max-num-seqs
  # bounds the Mamba-hybrid (Qwen3.5 teacher) state-cache so engine init doesn't fail.
  # --max-model-len 32768 fits BOTH models served here (Qwen2.5-Coder's full ctx; a safe cap
  # for the teacher's huge native ctx) so KV-cache fits at 0.6 util on the unified pool.
  PATH="$PWD/.venv-serve/bin:$PATH" \
  .venv-serve/bin/python -m vllm.entrypoints.openai.api_server --model "$1" --port 8000 \
      --max-num-seqs 256 --gpu-memory-utilization 0.6 --max-model-len 32768 "${q[@]}" \
      > "runs/vllm_${TS}.log" 2>&1 &
  VLLM_PID=$!
  for i in $(seq 1 120); do
    curl -sf "$ENDPOINT/v1/models" >/dev/null 2>&1 && { echo ">>> vLLM healthy ($1)"; return 0; }
    kill -0 "$VLLM_PID" 2>/dev/null || { echo ">>> vLLM died"; return 1; }
    sleep 10
  done; return 1
}
trap stop_vllm EXIT

# ---- 0. preflight + secrets ----
for v in .venv .venv-serve .venv-rl; do [ -x "$v/bin/python" ] || halt "missing venv $v — run ./setup.sh"; done
[ -f "$HOME/.config/llmda/gh_token" ] && export GH_TOKEN="$(cat "$HOME/.config/llmda/gh_token")" \
  && echo ">>> GH_TOKEN loaded" || echo ">>> no GH_TOKEN — PR mining limited"
# triton JIT (in vLLM init) needs Python.h at runtime; point at uv-managed CPython headers
_HDR=$(find "$HOME/.local/share/uv/python" -name Python.h -path "*3.12*" 2>/dev/null | head -1)
[ -n "$_HDR" ] && export CPATH="$(dirname "$_HDR"):${CPATH:-}" && echo ">>> CPATH set for triton JIT"

# ---- PHASE A guard: skip the teacher entirely when its outputs already exist ----
# (gen_sft is resumable, but serving the teacher means a 52GB download + 0.6-util vLLM;
#  if the instruction set is complete there is NOTHING for the teacher to do.)
SFT_TARGET="${SFT_TARGET:-1000}"
_have_instr=$(wc -l < data/datasets/sft_instructions.jsonl 2>/dev/null || echo 0)
if [ "$_have_instr" -ge "$SFT_TARGET" ]; then
  echo ">>> [A] sft_instructions.jsonl complete ($_have_instr >= $SFT_TARGET) — skipping teacher"
  export PREP_TEACHER=0
  NEED_TEACHER=0
else
  NEED_TEACHER=1
fi

# ---- 1. prep: download student (+ teacher only if needed) ----
scripts/prep.sh || halt "prep failed"

# ---- PHASE A: data-gen with the TEACHER alone ----
echo ">>> [A] ingest";        scripts/run_pipeline.sh ingest || halt "ingest failed"
if [ "$NEED_TEACHER" -eq 1 ]; then
  serve_model "$TEACHER" "$QUANT_TEACHER" || halt "teacher vLLM failed"
  echo ">>> [A] build datasets (SFT data-gen distills from the served teacher @ :8000)"
else
  echo ">>> [A] build datasets (teacher skipped — instruction set already complete)"
fi
scripts/run_pipeline.sh data || halt "dataset build failed"
stop_vllm                                              # free the teacher before training

# ---- PHASE B: train student (no vLLM resident). Eval at each boundary -> learning curve. ----
echo ">>> [B] baseline eval (un-tuned base)"; scripts/run_pipeline.sh eval --model "$MODEL" --tag base || true
echo ">>> [B] CPT";           scripts/run_pipeline.sh cpt || halt "CPT failed"
echo ">>> [B] eval after CPT"; scripts/run_pipeline.sh eval --model models/cpt/pr_mastery --tag cpt || true
echo ">>> [B] SFT";           scripts/run_pipeline.sh sft || halt "SFT failed"
echo ">>> [B] merge CPT+SFT -> models/student_merged (servable full model)"
scripts/run_pipeline.sh merge || halt "merge failed"
echo ">>> [B] eval after SFT (on the MERGED model — models/sft is adapter-only)"
scripts/run_pipeline.sh eval --model models/student_merged --tag sft || true

# ---- PHASE C: serve the TRAINED student -> census gate -> flywheel (RL/eval) ----
# Serving $MODEL (the base) here would throw away the training: rollouts/evals must hit
# the merged CPT+SFT model, and the flywheel must request the same served id.
serve_model "models/student_merged" "$QUANT" || halt "student vLLM failed"
echo ">>> [C] M1 census (HARD GATE for RL)"
scripts/run_pipeline.sh census "$CENSUS_VERIFY"; rc=$?
[ "$rc" -eq 2 ] && halt "M1 census NO-GO (CPT+SFT done; see log)" 2
[ "$rc" -ne 0 ] && halt "census errored"
echo ">>> [C] flywheel (cold-start gate + autonomous cycles)"
scripts/run_pipeline.sh flywheel --cycles "$CYCLES" --endpoint "$ENDPOINT" \
    --model models/student_merged \
    || halt "flywheel exited (cold-start NO-GO / plateau / RL pending — see log)"
echo "================ pipeline finished ================"
