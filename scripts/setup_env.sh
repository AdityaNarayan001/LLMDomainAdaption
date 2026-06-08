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

SKYRL_SRC="skyrl-train @ git+https://github.com/NovaSky-AI/SkyRL.git#subdirectory=skyrl-train"

make_venv () {  # name, extras
  local dir="$1" extras="$2"
  echo "=== $dir  ($extras) ==="
  "$PYBIN" -m venv "$dir"
  "$dir/bin/python" -m pip install -q --upgrade pip
  "$dir/bin/python" -m pip install -e ".$extras"
}

install_skyrl () {  # NovaSky SkyRL from source (skyrl-train, importable as skyrl_train)
  echo "=== .venv-rl: installing real SkyRL (skyrl-train) from source ==="
  .venv-rl/bin/python -m pip install "${SKYRL_SRC}"
}

case "${1:-all}" in
  core)  make_venv .venv "[dev,ast]" ;;
  main)  make_venv .venv "[train,dev,ast]" ;;
  serve) make_venv .venv-serve "[serve]" ;;
  rl)    make_venv .venv-rl "[rl]"; install_skyrl ;;
  all)
    make_venv .venv "[train,dev,ast]"
    make_venv .venv-serve "[serve]"
    make_venv .venv-rl "[rl]"; install_skyrl
    ;;
  *) echo "usage: setup_env.sh main|serve|rl|all|core"; exit 1 ;;
esac
echo "done."
