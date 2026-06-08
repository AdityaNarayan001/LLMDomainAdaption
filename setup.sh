#!/usr/bin/env bash
# (1/3) SETUP — provision all three auto-switching venvs + install all deps.
#   .venv (pipeline+training)  ·  .venv-serve (vLLM)  ·  .venv-rl (SkyRL via uv)
# Idempotent: safe to re-run. One-time human prerequisites it does NOT do:
#   - `gh auth login`   (needed by the ingest stage)
#   - model downloads happen lazily on first use (Qwen3.5-9B + teacher)
set -euo pipefail
cd "$(dirname "$0")"
echo "=== SETUP: building .venv / .venv-serve / .venv-rl ==="
scripts/setup_env.sh all
echo
echo "=== SETUP DONE. Next: ./smoke.sh ==="
echo "Reminder: run 'gh auth login' once before ./run.sh (ingest needs it)."
