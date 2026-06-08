"""Stage 4 — RL launcher. Drives **veRL** (GRPO/DAPO) with our verifiable cargo reward.

veRL is run as a config-driven entrypoint:
    python -m verl.trainer.main_ppo <hydra overrides>
We use GRPO (adv_estimator=grpo) with DAPO-style knobs, vLLM rollout, our task parquet
(built by build_verl.py), and our **custom reward** (src/train/verl_reward.compute_score)
which applies the generated patch and runs the cargo verifier (reuses src.harness.reward).

Runs inside .venv-rl (-> veRL env). Pi harness stays the eval harness; this first RL pass
is single-turn RLVR (generate patch -> verify); multi-turn agentic rollout is the upgrade.

NOTE (validation): exact veRL 0.8 override keys + the reward-fn signature must be confirmed
on a live run; the shape (entrypoint + GRPO + custom reward + parquet) is correct.
"""
from __future__ import annotations

import subprocess
import sys

from src import config

REWARD_FN = "src/train/verl_reward.py"


def verl_overrides(cfg: dict, endpoint: str, sft_ckpt: str, out_dir: str,
                   attn: str = "sdpa", tp: int = 1, gpus: int = 1) -> list[str]:
    """veRL main_ppo overrides — DAPO + the required fields VALIDATED via the smoke test.
    `+`-prefixed keys are added (not in base main_ppo config); the rest are base keys.
    attn=sdpa avoids flash-attn on aarch64; tp=1/gpus=1 for the single-GPU GB10."""
    a = cfg["algorithm"]
    r = cfg["rollout"]
    parquet = f"{config.ROOT}/data/datasets/rl_verl_train.parquet"
    micro = f"{a['prompts_per_batch']}"
    ov = [
        # --- GRPO base + DAPO clip-higher / token-level loss (base actor keys) ---
        "algorithm.adv_estimator=grpo",
        f"actor_rollout_ref.actor.clip_ratio_low={a['clip_low']}",
        f"actor_rollout_ref.actor.clip_ratio_high={a['clip_high']}",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"actor_rollout_ref.actor.optim.lr={a['lr']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={micro}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        f"algorithm.use_kl_in_reward={'false' if a['kl_free'] else 'true'}",
        # --- data ---
        f"data.train_files={parquet}", f"data.val_files={parquet}",
        f"data.train_batch_size={a['prompts_per_batch']}",
        f"data.max_response_length={r['max_tokens']}",
        # --- model / rollout (validated required fields) ---
        f"actor_rollout_ref.model.path={sft_ckpt}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={attn}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={r['samples_per_task']}",          # group size G
        f"actor_rollout_ref.rollout.temperature={r['temperature']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp}",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.5",          # Mamba-hybrid cache headroom
        "actor_rollout_ref.rollout.max_num_seqs=256",                    # <= Mamba cache blocks (Qwen3.5)
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1",
        # --- our verifiable cargo reward ---
        f"custom_reward_function.path={config.ROOT}/{REWARD_FN}",
        "custom_reward_function.name=compute_score",
        # --- trainer ---
        f"trainer.default_local_dir={out_dir}",
        f"trainer.n_gpus_per_node={gpus}", "trainer.nnodes=1",
        "trainer.total_epochs=1", "trainer.logger=[console]",
    ]
    if a.get("dynamic_sampling"):   # DAPO dynamic sampling — `+` (not in base config)
        ov += ["+algorithm.filter_groups.enable=true",
               "+algorithm.filter_groups.metric=acc",
               "+algorithm.filter_groups.max_num_gen_batches=10"]
    return ov


def build_command(cfg: dict, endpoint: str, sft_ckpt: str, out_dir: str) -> list[str]:
    return [sys.executable, "-m", "verl.trainer.main_ppo",
            *verl_overrides(cfg, endpoint, sft_ckpt, out_dir)]


def run(dry_run: bool = False, endpoint: str = "http://localhost:8000",
        sft_ckpt: str = "models/sft", out_dir: str = "models/rl") -> None:
    cfg = config.load("rl")
    cmd = build_command(cfg, endpoint, sft_ckpt, out_dir)
    print("[RL] veRL entrypoint:\n  " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(config.ROOT), check=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--sft-ckpt", default="models/sft")
    ap.add_argument("--out", default="models/rl")
    a = ap.parse_args()
    run(dry_run=a.dry_run, endpoint=a.endpoint, sft_ckpt=a.sft_ckpt, out_dir=a.out)
