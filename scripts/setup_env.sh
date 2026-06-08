#!/usr/bin/env bash
# Provision the THREE auto-switching venvs (see src/venvs.py). The stacks have
# mutually-exclusive pins (torch/transformers), so they MUST be separate venvs that
# talk over subprocess/HTTP.
#   .venv        pipeline + training (Axolotl CPT/SFT)   -> ".[train,dev,ast]"
#   .venv-serve  vLLM inference/rollout server           -> ".[serve]"
#   .venv-rl     SkyRL RL trainer                         -> ".[rl]"
# Usage: scripts/setup_env.sh <which>   where which = main | serve | rl | all | core
set -euo pipefail
cd "$(dirname "$0")/.."

PYBIN="${PYBIN:-python3}"

REPO_ROOT="$(pwd)"
SKYRL_DIR="${SKYRL_DIR:-$REPO_ROOT/../SkyRL}"     # SkyRL cloned as a sibling of the repo
# Deps SkyRL's RL entrypoint needs that its `gpu` extra skips on aarch64 (their pinned
# vLLM/vllm-router carry an x86_64 marker, so this transitive set isn't pulled on ARM):
SKYRL_AARCH64_FIX="loguru fastapi uvicorn omegaconf skyrl_gym jaxtyping torchdata"
SKYRL_OUR_DEPS="pyyaml pydantic requests datasets tqdm rich datasketch scipy"

make_venv () {  # name, extras
  local dir="$1" extras="$2"
  echo "=== $dir  ($extras) ==="
  "$PYBIN" -m venv "$dir"
  "$dir/bin/python" -m pip install -q --upgrade pip
  "$dir/bin/python" -m pip install -e ".$extras"
}

install_skyrl () {  # NovaSky SkyRL (unified `skyrl` package) via its uv flow; .venv-rl -> it
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "=== SkyRL: clone + uv sync --extra gpu --extra ray  ($SKYRL_DIR) ==="
  [ -d "$SKYRL_DIR/.git" ] || git clone --depth 1 https://github.com/NovaSky-AI/SkyRL.git "$SKYRL_DIR"
  ( cd "$SKYRL_DIR" && uv sync --extra gpu --extra ray )
  local SKPY="$SKYRL_DIR/.venv/bin/python"
  # complete the aarch64-skipped deps + our core deps so the entrypoint + our env import
  uv pip install --python "$SKPY" -q $SKYRL_AARCH64_FIX $SKYRL_OUR_DEPS
  echo "=== point .venv-rl at SkyRL's uv venv ==="
  rm -rf "$REPO_ROOT/.venv-rl"; ln -s "$SKYRL_DIR/.venv" "$REPO_ROOT/.venv-rl"
  "$REPO_ROOT/.venv-rl/bin/python" -c "import skyrl.train.entrypoints.main_base; print('SkyRL RL env OK')"
}

case "${1:-all}" in
  core)  make_venv .venv "[dev,ast]" ;;
  main)  make_venv .venv "[train,dev,ast]" ;;
  serve) make_venv .venv-serve "[serve]" ;;
  rl)    install_skyrl ;;                          # .venv-rl is a symlink to SkyRL's uv venv
  all)
    make_venv .venv "[train,dev,ast]"
    make_venv .venv-serve "[serve]"
    install_skyrl
    ;;
  *) echo "usage: setup_env.sh main|serve|rl|all|core"; exit 1 ;;
esac
echo "done."
