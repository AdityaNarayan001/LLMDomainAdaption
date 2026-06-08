"""Config loading + shared paths. Every stage is driven by a YAML in ``configs/``.

Kept dependency-light (pyyaml only) so the data/eval/orchestration layers import
cleanly on a box without the training stack installed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = two levels up from this file (src/config.py -> repo root).
ROOT = Path(__file__).resolve().parent.parent
CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
MODELS = ROOT / "models"
RUNS = ROOT / "runs"
EVAL_SETS = ROOT / "eval_sets"


def load(name: str) -> dict[str, Any]:
    """Load a stage config by short name, e.g. ``load("data")`` -> configs/data.yaml."""
    path = CONFIGS / (name if name.endswith((".yaml", ".yml")) else f"{name}.yaml")
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    with path.open() as f:
        return yaml.safe_load(f)


def ensure_dirs() -> None:
    """Create the gitignored working dirs (data/, models/, runs/) on demand."""
    for d in (DATA, MODELS, RUNS):
        d.mkdir(parents=True, exist_ok=True)
