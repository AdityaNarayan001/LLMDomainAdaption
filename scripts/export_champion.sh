#!/usr/bin/env bash
# Merge the crowned champion's adapter stack into a single standalone model for serving.
# Reads runs/best_model.json (written by the flywheel's Tier-2 crowning). Runs in .venv.
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f runs/best_model.json ] || { echo "no runs/best_model.json yet (run the flywheel first)"; exit 1; }
.venv/bin/python -m src.train.export "$@"
