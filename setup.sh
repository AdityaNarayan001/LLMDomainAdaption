#!/usr/bin/env bash
# (1/3) SETUP — self-healing environment provisioning. Idempotent: detects what's missing
# (uv, gh, the three venvs) and installs only that. Safe to re-run.
#   .venv (pipeline+training)  ·  .venv-serve (vLLM)  ·  .venv-rl (SkyRL via uv)
set -euo pipefail
cd "$(dirname "$0")"

echo "=== SETUP (idempotent) ==="

# uv (needed for the SkyRL env)
if ! command -v uv >/dev/null && [ ! -x "$HOME/.local/bin/uv" ]; then
  echo ">>> installing uv ..."; curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# gh (GitHub CLI — for PR/issue mining; auth is separate, see note below)
if ! command -v gh >/dev/null; then
  echo ">>> installing gh ..."
  if command -v apt-get >/dev/null; then
    (type -p curl >/dev/null || sudo apt-get install -y curl)
    curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg \
      | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" \
      | sudo tee /etc/apt/sources.list.d/github-cli.list >/dev/null
    sudo apt-get update -q && sudo apt-get install -y gh
  else
    echo ">>> WARN: no apt-get; install gh manually (https://cli.github.com) — or use GH_TOKEN."
  fi
fi

# the three venvs (setup_env.sh skips any that already exist)
scripts/setup_env.sh all

echo
echo "=== SETUP DONE. Next: ./smoke.sh, then ./run.sh ==="
echo "GitHub PR mining needs auth (rate limits): run 'gh auth login' OR export GH_TOKEN=<pat>."
