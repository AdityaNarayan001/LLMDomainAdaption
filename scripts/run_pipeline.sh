#!/usr/bin/env bash
# End-to-end pipeline entrypoint with AUTO-SWITCHING venvs (see src/venvs.py):
#   .venv (pipeline+training) · .venv-serve (vLLM) · .venv-rl (SkyRL)
# Usage: scripts/run_pipeline.sh <stage> [args...]
#   ingest | census | data | cpt | sft | serve | rl | flywheel | eval | smoke
set -euo pipefail
cd "$(dirname "$0")/.."

stage="${1:-help}"; shift || true

# stage -> venv dir (mirror of STAGE_VENV in src/venvs.py)
case "$stage" in
  serve)            VENV=".venv-serve" ;;
  rl)               VENV=".venv-rl" ;;
  *)                VENV=".venv" ;;
esac
PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY not found. Create it (scripts/setup_env.sh)."; exit 1; }
echo "[run_pipeline] stage=$stage  venv=$VENV"

case "$stage" in
  ingest)   "$PY" -m src.data.ingest_repos && "$PY" -m src.data.ingest_github ;;
  census)   "$PY" -m src.data.census --verify-sample "${1:-0}" ;;     # M1 HARD GATE
  data)     "$PY" -m src.data.build_cpt && "$PY" -m src.data.build_rl \
            && "$PY" -m src.data.build_verl && "$PY" -m src.data.gen_sft \
            && "$PY" -m src.data.validate ;;          # sanity + decontam + RL-repro gate
  validate) "$PY" -m src.data.validate ;;
  eval)     "$PY" -m src.eval.run_eval "$@" ;;        # baseline/per-stage held-out perplexity
  cpt)      "$PY" -m src.train.cpt "$@" ;;
  sft)      "$PY" -m src.train.sft "$@" ;;
  merge)    "$PY" -m src.train.merge_student "$@" ;;  # CPT+SFT LoRA -> models/student_merged
  serve)    "$PY" -m vllm.entrypoints.openai.api_server "$@" ;;        # .venv-serve
  rl)       "$PY" -m src.train.rl "$@" ;;                              # .venv-rl
  flywheel) "$PY" -m src.orchestrate.flywheel "$@" ;;                  # .venv (shells out to others)
  smoke)    "$PY" -m pytest tests/ -q ;;
  help|*)   echo "stages: ingest | census | data | cpt | sft | merge | serve | rl | flywheel | smoke"; exit 1 ;;
esac
