#!/usr/bin/env bash
# (3/3) RUN — drive the whole pipeline autonomously (no human prompts).
# Intended to be launched INSIDE a tmux session on the GPU box, e.g.:
#     tmux new -s hs './run.sh'      # (we'll do this when you say)
#
# Gates AUTO-HALT with a logged reason (they never wait for a human). vLLM is launched
# from .venv-serve, health-checked, and torn down on exit. All output -> runs/pipeline_*.log.
#
# Stops when (see "TRAINING STOPS WHEN" in the run summary): a gate fails, the flywheel
# plateaus, the cycle budget is exhausted, or you kill the tmux session.
set -uo pipefail
cd "$(dirname "$0")"

TS="$(date +%Y%m%d_%H%M%S)"
mkdir -p runs
LOG="runs/pipeline_${TS}.log"
exec > >(tee -a "$LOG") 2>&1

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
ENDPOINT="${ENDPOINT:-http://localhost:8000}"
CYCLES="${CYCLES:-1000}"        # flywheel loops until plateau OR this budget (effectively "forever")
CENSUS_VERIFY="${CENSUS_VERIFY:-25}"

echo "================ pipeline run ${TS} ================"
echo "model=$MODEL  endpoint=$ENDPOINT  cycles=$CYCLES  log=$LOG"

halt(){ echo ">>> HALT: $1"; exit "${2:-1}"; }

# ---- preflight ----
for v in .venv .venv-serve .venv-rl; do [ -x "$v/bin/python" ] || halt "missing venv $v — run ./setup.sh"; done
command -v gh >/dev/null && gh auth status >/dev/null 2>&1 || echo ">>> WARN: gh not authenticated — ingest may fail (run 'gh auth login')."

# ---- vLLM server: launch (.venv-serve), health-wait, teardown on exit ----
echo ">>> launching vLLM ($MODEL) from .venv-serve ..."
.venv-serve/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port 8000 --quantization modelopt_fp4 > "runs/vllm_${TS}.log" 2>&1 &
VLLM_PID=$!
trap '[ -n "${VLLM_PID:-}" ] && kill "$VLLM_PID" 2>/dev/null; echo ">>> vLLM stopped"' EXIT
echo ">>> waiting for vLLM to become healthy ..."
for i in $(seq 1 90); do
  curl -sf "$ENDPOINT/v1/models" >/dev/null 2>&1 && { echo ">>> vLLM healthy"; break; }
  kill -0 "$VLLM_PID" 2>/dev/null || halt "vLLM process died (see runs/vllm_${TS}.log)"
  [ "$i" -eq 90 ] && halt "vLLM not healthy after 15min (see runs/vllm_${TS}.log)"
  sleep 10
done

# ---- pipeline stages (each auto-switches venv via run_pipeline.sh) ----
echo ">>> [1/6] ingest";  scripts/run_pipeline.sh ingest || halt "ingest failed"
echo ">>> [2/6] M1 census (HARD GATE)"
scripts/run_pipeline.sh census "$CENSUS_VERIFY"; rc=$?
[ "$rc" -eq 2 ] && halt "M1 census NO-GO: too few execution-verifiable tasks — re-scope RL (see log)" 2
[ "$rc" -ne 0 ] && halt "census errored"
echo ">>> [3/6] build datasets"; scripts/run_pipeline.sh data || halt "dataset build failed"
echo ">>> [4/6] CPT";            scripts/run_pipeline.sh cpt  || halt "CPT failed"
echo ">>> [5/6] SFT";            scripts/run_pipeline.sh sft  || halt "SFT failed"
echo ">>> [6/6] flywheel (cold-start gate + autonomous cycles)"
CYCLES="$CYCLES" scripts/run_pipeline.sh flywheel --cycles "$CYCLES" \
    || halt "flywheel exited (cold-start NO-GO, plateau, or error) — see log"

echo "================ pipeline finished (flywheel converged / budget exhausted) ================"
