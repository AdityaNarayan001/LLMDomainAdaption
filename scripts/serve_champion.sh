#!/usr/bin/env bash
# Host the exported champion model with vLLM (OpenAI-compatible endpoint).
# Usage: scripts/serve_champion.sh [MODEL_DIR]   (default: models/champion_export)
#   env: PORT (default 8000), QUANT (default modelopt_fp4; set "" for bf16)
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:-models/champion_export}"
[ -f "$MODEL/config.json" ] || { echo "no servable model at $MODEL — run ./scripts/export_champion.sh"; exit 1; }
# QUANT empty => bf16 (default). Set QUANT=modelopt_fp4 only for a pre-quantized NVFP4 model.
QUANT_ARG=(); [ -n "${QUANT:-}" ] && QUANT_ARG=(--quantization "$QUANT")
echo "serving $MODEL on :${PORT:-8000} (.venv-serve, ${QUANT:-bf16})"
exec .venv-serve/bin/python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" --port "${PORT:-8000}" "${QUANT_ARG[@]}"
