#!/usr/bin/env bash
# Environment setup. Light core deps run anywhere; heavy extras only on the GPU box.
set -euo pipefail
cd "$(dirname "$0")/.."

if ! command -v uv >/dev/null 2>&1; then
  echo "installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh
fi

uv venv --python 3.10 .venv
source .venv/bin/activate

# Core (data pipeline, eval, orchestration) — installs without GPUs.
uv pip install -e .

case "${1:-core}" in
  train) uv pip install -e ".[train]" ;;
  serve) uv pip install -e ".[serve]" ;;
  rl)    uv pip install -e ".[rl]" ;;
  all)   uv pip install -e ".[train,serve,rl,dev]" ;;
  core)  uv pip install -e ".[dev]" ;;
esac

echo "done. activate with: source .venv/bin/activate"
