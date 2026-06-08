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

install_verl () {  # veRL RL trainer in .venv-verl; .venv-rl -> it. (Validated on GB10/aarch64.)
  if [ -x "$REPO_ROOT/.venv-rl/bin/python" ]; then echo "=== .venv-rl exists — skipping veRL ==="; return; fi
  command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
  echo "=== veRL: .venv-verl + verl + vllm (+ aarch64-skipped deps + our deps) ==="
  "$PYBIN" -m venv .venv-verl
  .venv-verl/bin/python -m pip install -q --upgrade pip
  .venv-verl/bin/python -m pip install verl vllm $VERL_FIX $VERL_OUR_DEPS ninja packaging wheel
  # flash-attn is mandatory (veRL's bert_padding). Build for the local GPU arch with PTX.
  # CPATH points at uv-managed CPython headers (Python.h) since system python3-dev may be absent.
  uv python install 3.12 || true
  local HDR=$(find "$HOME/.local/share/uv/python" -name Python.h -path "*3.12*" 2>/dev/null | head -1)
  echo "=== building flash-attn (arch=${TORCH_CUDA_ARCH_LIST:-12.0+PTX}) — long, one-time ==="
  CPATH="$(dirname "$HDR"):${CPATH:-}" MAX_JOBS="${MAX_JOBS:-16}" \
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0+PTX}" \
    .venv-verl/bin/python -m pip install flash-attn --no-build-isolation
  rm -rf "$REPO_ROOT/.venv-rl"; ln -s "$REPO_ROOT/.venv-verl" "$REPO_ROOT/.venv-rl"
  .venv-rl/bin/python -c "import verl.trainer.main_ppo, flash_attn; import src.harness.reward; print('veRL RL env OK', flash_attn.__version__)"
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
