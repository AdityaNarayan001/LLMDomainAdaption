#!/usr/bin/env bash
# (3/3) RUN — self-healing, autonomous pipeline. Launch inside tmux on the GPU box:
#     tmux new -s hs './run.sh'
#
# PHASES (teacher and student NEVER coexist on the GPU — they don't fit together):
#   A) DATA-GEN  : serve the 122B TEACHER alone -> generate SFT data -> stop teacher.
#   B) TRAIN     : CPT + SFT (no vLLM needed).
#   C) RL/EVAL   : serve the STUDENT -> census gate -> flywheel (rollouts + eval).
# Gates AUTO-HALT with a logged reason. All output -> runs/pipeline_*.log.
#
# Env: MODEL (student, Qwen3.5-9B), TEACHER (Qwen3.5-122B-A10B), ENDPOINT, CYCLES,
#      QUANT (empty=bf16 student; teacher served NVFP4 if QUANT_TEACHER set), CENSUS_VERIFY.
set -uo pipefail
cd "$(dirname "$0")"

TS="$(date +%Y%m%d_%H%M%S)"; mkdir -p runs
LOG="runs/pipeline_${TS}.log"; exec > >(tee -a "$LOG") 2>&1

MODEL="${MODEL:-Qwen/Qwen2.5-Coder-7B}"
TEACHER="${TEACHER:-Qwen/Qwen3.5-27B}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"
CYCLES="${CYCLES:-1000}"; CENSUS_VERIFY="${CENSUS_VERIFY:-25}"
QUANT="${QUANT:-}"; QUANT_TEACHER="${QUANT_TEACHER-modelopt_fp4}"  # '-' not ':-': empty => bf16
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
  VLLM_PID=""; sleep 4; echo ">>> vLLM stopped"
}
serve_model(){ # $1=model $2=quant ; (re)launch vLLM alone, health-wait
  stop_vllm
  local q=(); [ -n "$2" ] && q=(--quantization "$2")
  echo ">>> serving $1 (${2:-bf16}) ..."
  # PATH must include the serve venv bin so flashinfer's JIT finds `ninja`; --max-num-seqs
  # bounds the Mamba-hybrid (Qwen3.5/Nemotron) state-cache so engine init doesn't fail.
  # --max-model-len caps the model's huge native context (Qwen3.5 = 262144) so KV-cache fits
  # at 0.6 util on the shared unified pool; 64K covers teacher prompts + student rollouts (<=16K).
  PATH="$PWD/.venv-serve/bin:$PATH" \
  .venv-serve/bin/python -m vllm.entrypoints.openai.api_server --model "$1" --port 8000 \
      --max-num-seqs 256 --gpu-memory-utilization 0.6 --max-model-len 65536 "${q[@]}" \
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

# ---- 1. prep: download student + teacher (PREP_TEACHER=1) ----
scripts/prep.sh || halt "prep failed"

# ---- PHASE A: data-gen with the TEACHER alone ----
echo ">>> [A] ingest";        scripts/run_pipeline.sh ingest || halt "ingest failed"
serve_model "$TEACHER" "$QUANT_TEACHER" || halt "teacher vLLM failed"
echo ">>> [A] build datasets (SFT data-gen distills from the served teacher @ :8000)"
scripts/run_pipeline.sh data || halt "dataset build failed"
stop_vllm                                              # free the teacher before training

# ---- PHASE B: train student (no vLLM resident) ----
echo ">>> [B] CPT";           scripts/run_pipeline.sh cpt || halt "CPT failed"
echo ">>> [B] SFT";           scripts/run_pipeline.sh sft || halt "SFT failed"

# ---- PHASE C: serve student -> census gate -> flywheel (RL/eval) ----
serve_model "$MODEL" "$QUANT" || halt "student vLLM failed"
echo ">>> [C] M1 census (HARD GATE for RL)"
scripts/run_pipeline.sh census "$CENSUS_VERIFY"; rc=$?
[ "$rc" -eq 2 ] && halt "M1 census NO-GO (CPT+SFT done; see log)" 2
[ "$rc" -ne 0 ] && halt "census errored"
echo ">>> [C] flywheel (cold-start gate + autonomous cycles)"
scripts/run_pipeline.sh flywheel --cycles "$CYCLES" --endpoint "$ENDPOINT" \
    || halt "flywheel exited (cold-start NO-GO / plateau / RL pending — see log)"
echo "================ pipeline finished ================"
