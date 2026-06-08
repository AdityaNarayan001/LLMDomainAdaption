#!/usr/bin/env bash
# End-to-end pipeline entrypoint, gated by the milestone ladder (see PLAN.txt / README).
# Usage: scripts/run_pipeline.sh <stage>
#   ingest | census | data | cpt | sft | rl | flywheel | eval
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

stage="${1:-help}"
case "$stage" in
  ingest)   python -m src.data.ingest_repos && python -m src.data.ingest_github ;;
  census)   python -m src.data.census --verify-sample "${2:-0}" ;;   # M1 HARD GATE
  data)     python -m src.data.build_cpt && python -m src.data.build_rl && python -m src.data.build_sft ;;
  cpt)      python -m src.train.cpt "${@:2}" ;;
  sft)      python -m src.train.sft "${@:2}" ;;
  rl)       python -m src.train.rl "${@:2}" ;;
  flywheel) python -m src.orchestrate.flywheel "${@:2}" ;;
  smoke)    python -m pytest tests/ -q ;;
  help|*)   echo "stages: ingest | census | data | cpt | sft | rl | flywheel | smoke"; exit 1 ;;
esac
