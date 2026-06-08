#!/usr/bin/env bash
# Idempotent RUN-prerequisite prep — detect what's missing and fix it, so ./run.sh "just works".
#   * downloads the base model into the shared HF cache if not present (no-op if cached)
#   * (optional) downloads the SFT teacher if PREP_TEACHER=1 (large — off by default)
#   * reports gh/GH_TOKEN status (PR mining needs it; degrades gracefully otherwise)
# Safe to re-run. Models live in ~/.cache/huggingface (shared by .venv + .venv-serve).
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
TEACHER="${TEACHER:-Qwen/Qwen3.5-122B-A10B}"

ensure_model () {  # idempotent: snapshot_download skips already-cached files
  local m="$1"
  echo ">>> ensuring model cached: $m"
  .venv/bin/python - "$m" <<'PY'
import sys
from huggingface_hub import snapshot_download
m = sys.argv[1]
path = snapshot_download(m)          # downloads if missing, no-op if present
print(f"    ready: {path}")
PY
}

echo "=== PREP ==="
ensure_model "$MODEL"
if [ "${PREP_TEACHER:-0}" = "1" ]; then
  echo ">>> PREP_TEACHER=1 — downloading teacher (LARGE): $TEACHER"
  ensure_model "$TEACHER"
else
  echo ">>> skipping teacher download (set PREP_TEACHER=1 to fetch $TEACHER — large)"
fi

echo "=== gh / GH_TOKEN (for PR mining) ==="
if command -v gh >/dev/null && { [ -n "${GH_TOKEN:-}" ] || [ -n "${GITHUB_TOKEN:-}" ] || gh auth status >/dev/null 2>&1; }; then
  echo ">>> gh ready — PR mining enabled (5000 req/hr)."
else
  echo ">>> WARN: gh not authenticated and no GH_TOKEN. Repo clone + CPT still work, but"
  echo "    PR/issue mining (RL tasks, PR-Mastery) is skipped -> census may NO-GO."
  echo "    Fix: 'gh auth login' OR export GH_TOKEN=<pat>."
fi
echo "=== PREP DONE ==="
