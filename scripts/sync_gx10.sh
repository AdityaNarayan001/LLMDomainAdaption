#!/usr/bin/env bash
# Keep the local code in sync with gx10-b (CODE ONLY — never weights/checkpoints/data).
# Usage: scripts/sync_gx10.sh         # local -> remote (default)
#        scripts/sync_gx10.sh pull    # remote -> local (rare; e.g. pull back a tweaked file)
set -euo pipefail

REMOTE_HOST="gx10-b"
REMOTE_DIR="~/adityaN/LLMDomainAdaption/"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)/"

# Excludes: virtualenv, the big repo clone, checkpoints, run artifacts, caches, git.
# (/data/ /models/ /runs/ are exactly the weights/checkpoints/logs we do NOT rsync.)
# LEADING SLASH = root-anchored, so these do NOT also match src/data/, etc.
EXCLUDES=(
  --exclude='/.venv/' --exclude='/.venv-serve/' --exclude='/.venv-rl/'
  --exclude='/data/' --exclude='/models/' --exclude='/runs/'
  --exclude='/.git/' --exclude='__pycache__/' --exclude='*.pyc'
  --exclude='.pytest_cache/' --exclude='.ruff_cache/'
)

if [[ "${1:-push}" == "pull" ]]; then
  rsync -az --delete "${EXCLUDES[@]}" "${REMOTE_HOST}:${REMOTE_DIR}" "${LOCAL_DIR}"
  echo "pulled code from ${REMOTE_HOST}"
else
  rsync -az --delete "${EXCLUDES[@]}" "${LOCAL_DIR}" "${REMOTE_HOST}:${REMOTE_DIR}"
  echo "pushed code to ${REMOTE_HOST}:${REMOTE_DIR}"
fi
