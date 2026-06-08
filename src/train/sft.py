"""Stage 3 — SFT launcher. LoRA on top of CPT (stacked, not merged). Axolotl.

Trains on rejection-sampled agentic trajectories + grounded instructions, with an
assistant-only loss mask. Includes the overfit guard (perplexity regression vs CPT).
"""
from __future__ import annotations

import subprocess

import yaml

from src import config
from src.eval import perplexity


def axolotl_config(cfg: dict, cpt_ckpt: str) -> dict:
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
        "output_dir": "models/sft",
        **(
            {"report_to": "wandb", "wandb_project": cfg["tracking"]["wandb_project"]}
            if cfg.get("tracking", {}).get("wandb")
            else {"report_to": "none"}
        ),
    }


def run(dry_run: bool = False, cpt_ckpt: str = "models/cpt/pr_mastery") -> None:
    cfg = config.load("sft")
    ax = axolotl_config(cfg, cpt_ckpt)
    cfg_path = config.RUNS / "axolotl_sft.yaml"
    cfg_path.parent.mkdir(parents=True, exist_ok=True)
    cfg_path.write_text(yaml.safe_dump(ax))
    print(f"[SFT] -> {cfg_path}")
    if dry_run:
        return
    subprocess.run(["axolotl", "train", str(cfg_path)], check=True)

    # overfit guard (decision: SFT quality gates everything downstream)
    heldout = "data/datasets/cpt_heldout.jsonl"
    try:
        reg = perplexity.forgetting_regression_pct(
            perplexity.perplexity(cpt_ckpt, heldout),
            perplexity.perplexity("models/sft", heldout),
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
