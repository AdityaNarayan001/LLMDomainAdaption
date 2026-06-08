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
# veRL deps pip skips on aarch64 + our core deps (so the custom reward imports our src):
VERL_FIX="cachetools uvicorn fastapi"
VERL_OUR_DEPS="pyyaml pydantic requests datasets tqdm rich datasketch scipy"

make_venv () {  # name, extras  (idempotent: skip if the venv already works)
  local dir="$1" extras="$2"
  if [ -x "$dir/bin/python" ]; then echo "=== $dir exists — skipping ==="; return; fi
  echo "=== $dir  ($extras) ==="
  "$PYBIN" -m venv "$dir"
  "$dir/bin/python" -m pip install -q --upgrade pip
  "$dir/bin/python" -m pip install -e ".$extras"
}

install_verl () {  # veRL RL trainer in .venv-verl; .venv-rl -> it
  if [ -x "$REPO_ROOT/.venv-rl/bin/python" ]; then echo "=== .venv-rl exists — skipping veRL ==="; return; fi
  echo "=== veRL: .venv-verl + pip install verl (+ aarch64-skipped deps + our deps) ==="
  "$PYBIN" -m venv .venv-verl
  .venv-verl/bin/python -m pip install -q --upgrade pip
  .venv-verl/bin/python -m pip install verl
  .venv-verl/bin/python -m pip install -q $VERL_FIX $VERL_OUR_DEPS
  rm -rf "$REPO_ROOT/.venv-rl"; ln -s "$REPO_ROOT/.venv-verl" "$REPO_ROOT/.venv-rl"
  .venv-rl/bin/python -c "import verl.trainer.main_ppo; import src.harness.reward; print('veRL RL env OK')"
}

case "${1:-all}" in
  core)  make_venv .venv "[dev,ast]" ;;
  main)  make_venv .venv "[train,dev,ast]" ;;
  serve) make_venv .venv-serve "[serve]" ;;
  rl)    install_verl ;;                            # .venv-rl -> .venv-verl (veRL)
  all)
    make_venv .venv "[train,dev,ast]"
    make_venv .venv-serve "[serve]"
    install_verl
    ;;
  *) echo "usage: setup_env.sh main|serve|rl|all|core"; exit 1 ;;
esac
echo "done."
