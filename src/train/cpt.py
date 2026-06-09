"""Stage 2 — CPT launcher. Translates configs/cpt.yaml -> Axolotl config and runs it.

Knowledge injection => full FT of the 9B via 8-bit paged AdamW + gradient checkpointing
(decision #3). rsLoRA r=256 fallback if it OOMs. Phased curriculum with anti-forgetting
adapter carry-over (design from Phased-CPT). Emits one Axolotl run per phase.
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src import config


def _train_env() -> dict:
    """Env for the axolotl subprocess: venv bin on PATH (so its `accelerate` launcher resolves to
    .venv/bin/accelerate, not a stale /snap/bin), and CPATH -> uv-managed CPython headers so the
    fla causal_conv1d triton kernel (Qwen3.5 Mamba-hybrid) can JIT-compile (needs Python.h)."""
    env = {**os.environ,
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    hdr = glob.glob(str(Path.home() / ".local/share/uv/python/*3.12*/include/python3.12"))
    if hdr:
        env["CPATH"] = hdr[0] + os.pathsep + env.get("CPATH", "")
    return env


AXOLOTL = str(Path(sys.executable).with_name("axolotl"))
_VENV_ENV = _train_env()


def axolotl_config_for_phase(cfg: dict, phase: dict, prev_ckpt: str | None) -> dict:
    """Build an Axolotl YAML dict for one curriculum phase."""
    m = cfg["method"]
    hp = cfg["hyperparams"]
    ax: dict = {
        "base_model": cfg["base_model"],
        "datasets": [{"path": phase["data"], "type": "completion",
                      "field": "training_content"}],
        "sequence_len": hp["max_seq_len"],
        # packing forces fla's variable-length causal_conv1d (TileLang kernel) for the Qwen3.5
        # Mamba-hybrid, which fails to compile on aarch64/Blackwell. Disable -> standard padded
        # conv path (less throughput, but trains). Revisit if a working fla/tilelang build lands.
        "sample_packing": False,
        "gradient_checkpointing": m["gradient_checkpointing"],
        "optimizer": m["optimizer"],            # paged_adamw_8bit
        "learning_rate": hp["lr"],
        "lr_scheduler": hp["lr_schedule"],
        "warmup_ratio": hp["warmup_ratio"],
        "micro_batch_size": hp["micro_batch_size"],
        "gradient_accumulation_steps": hp["gradient_accumulation"],
        "num_epochs": phase["epochs"],
        "bf16": hp["bf16"],
        "weight_decay": hp["weight_decay"],
        "save_steps": 50,
        "output_dir": f"models/cpt/{phase['name']}",
        # embeddings frozen (decision #6): do NOT add lm_head/embed_tokens to trainable set
    }
    track = cfg.get("tracking", {})
    if track.get("wandb"):
        ax["wandb_project"] = track["wandb_project"]
        ax["report_to"] = "wandb"
    else:
        ax["report_to"] = "none"  # WandB optional — metrics.py JSONL is source of truth
    if m["full_ft"]:
        ax["adapter"] = None  # full fine-tuning
    else:  # rsLoRA fallback
        rs = m["rslora_fallback"]
        ax.update({
            "adapter": "lora", "lora_r": rs["r"], "lora_alpha": rs["alpha"],
            "use_rslora": rs["use_rslora"], "lora_target_modules": rs["target_modules"],
        })
    if prev_ckpt:
        ax["resume_from_checkpoint"] = prev_ckpt  # anti-forgetting carry-over
    return ax


def run(dry_run: bool = False) -> None:
    cfg = config.load("cpt")
    config.ensure_dirs()
    prev = None
    for phase in cfg["curriculum"]["phases"]:
        ax = axolotl_config_for_phase(cfg, phase, prev if cfg["curriculum"]["load_previous_adapter"] else None)
        cfg_path = config.RUNS / f"axolotl_cpt_{phase['name']}.yaml"
        cfg_path.parent.mkdir(parents=True, exist_ok=True)
        cfg_path.write_text(yaml.safe_dump(ax))
        print(f"[CPT] phase={phase['name']} -> {cfg_path}")
        if not dry_run:
            subprocess.run([AXOLOTL, "train", str(cfg_path)], check=True, env=_VENV_ENV)
        prev = ax["output_dir"]


if __name__ == "__main__":
    import sys
    run(dry_run="--dry-run" in sys.argv)
