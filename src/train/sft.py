"""Stage 3 — SFT launcher. LoRA on top of CPT (stacked, not merged). Axolotl.

Trains on rejection-sampled agentic trajectories + grounded instructions, with an
assistant-only loss mask. Includes the overfit guard (perplexity regression vs CPT).
"""
from __future__ import annotations

import glob
import os
import subprocess
import sys
from pathlib import Path

import yaml

from src import config
from src.eval import perplexity


def _train_env() -> dict:
    """venv bin on PATH (axolotl's accelerate launcher) + CPATH -> uv CPython headers (fla
    causal_conv1d triton JIT needs Python.h on the Qwen3.5 Mamba-hybrid)."""
    env = {**os.environ,
           "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")}
    hdr = glob.glob(str(Path.home() / ".local/share/uv/python/*3.12*/include/python3.12"))
    if hdr:
        env["CPATH"] = hdr[0] + os.pathsep + env.get("CPATH", "")
    return env


AXOLOTL = str(Path(sys.executable).with_name("axolotl"))
_VENV_ENV = _train_env()


def axolotl_config(cfg: dict, cpt_ckpt: str, out_dir: str = "models/sft") -> dict:
    lora = cfg["method"]["lora"]
    hp = cfg["hyperparams"]
    return {
        "base_model": cpt_ckpt,                       # stack on CPT
        "adapter": "qlora" if cfg["method"]["qlora"] else "lora",
        "lora_r": lora["r"], "lora_alpha": lora["alpha"], "lora_dropout": lora["dropout"],
        "lora_target_modules": lora["target_modules"],
        "datasets": [
            {"path": cfg["data"]["trajectories"], "type": "chat_template",
             "train_on_inputs": False},   # assistant-only loss mask
            {"path": cfg["data"]["instructions"], "type": "chat_template",
             "train_on_inputs": False},
        ],
        "sequence_len": hp["max_seq_len"],
        "learning_rate": hp["lr"], "lr_scheduler": hp["lr_schedule"],
        "warmup_ratio": hp["warmup_ratio"],
        "micro_batch_size": hp["micro_batch_size"],
        "gradient_accumulation_steps": hp["gradient_accumulation"],
        "num_epochs": hp["epochs"],
        "output_dir": out_dir,
        **(
            {"report_to": "wandb", "wandb_project": cfg["tracking"]["wandb_project"]}
            if cfg.get("tracking", {}).get("wandb")
            else {"report_to": "none"}
        ),
    }


def run(dry_run: bool = False, cpt_ckpt: str = "models/cpt/pr_mastery",
        out_dir: str = "models/sft") -> None:
    cfg = config.load("sft")
    ax = axolotl_config(cfg, cpt_ckpt, out_dir=out_dir)
    cfg_path = config.RUNS / "axolotl_sft.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(ax))
    print(f"[SFT] -> {cfg_path}  (out_dir={out_dir})")
    if dry_run:
        return
    subprocess.run([AXOLOTL, "train", str(cfg_path)], check=True, env=_VENV_ENV)

    # overfit guard (decision: SFT quality gates everything downstream)
    heldout = "data/datasets/cpt_heldout.jsonl"
    try:
        reg = perplexity.forgetting_regression_pct(
            perplexity.perplexity(cpt_ckpt, heldout),
            perplexity.perplexity(out_dir, heldout),
        )
        limit = cfg["overfit_guard"]["max_perplexity_regression_pct"]
        print(f"[SFT] perplexity regression vs CPT: {reg:.1f}% (limit {limit}%)")
        if reg > limit:
            print("[SFT] WARNING: overfit guard tripped — cut LR / raise replay before RL.")
    except Exception as exc:  # pragma: no cover
        print(f"[SFT] perplexity guard skipped: {exc}")


if __name__ == "__main__":
    import sys
    run(dry_run="--dry-run" in sys.argv)
