"""Stage 4 — RL launcher. Drives **veRL** (GRPO/DAPO) with our verifiable cargo reward.

veRL is run as a config-driven entrypoint:
    python -m verl.trainer.main_ppo <hydra overrides>
We use GRPO (adv_estimator=grpo) with DAPO-style knobs, vLLM rollout, our task parquet
(built by build_verl.py), and our **custom reward** (src/train/verl_reward.compute_score)
which applies the generated patch and runs the cargo verifier (reuses src.harness.reward).

Runs inside .venv-rl (-> veRL env). Pi harness stays the eval harness; this first RL pass
is single-turn RLVR (generate patch -> verify); multi-turn agentic rollout is the upgrade.

GB10 UNIFIED-MEMORY LESSONS (hard-won, baked in below — see memory rl-single-box-wall):
  * vLLM sleep mode BALLOONS init on the unified pool (CuMemAllocator reserves offload
    buffers while awake) -> enable_sleep_mode=false, free_cache_engine=false.
  * Ray's default object store (30% of RAM ~ 36GB in /dev/shm) starves the pool -> cap it
    (start ray externally with --object-store-memory=4e9, or the env below as a backstop).
  * gpu_memory_utilization barely moves vLLM's fixed ~50GB footprint; the binding peak is
    the actor->vLLM weight sync. A 7B actor + 7B rollout DOES NOT FIT one 121GB box —
    this launcher is correct for scale-out / smaller models; on the GB10 expect OOM at 7B.
"""
from __future__ import annotations

import os
import subprocess
import sys

from src import config

REWARD_FN = "src/train/verl_reward.py"


def _n_tasks() -> int:
    p = config.ROOT / "data/datasets/rl_tasks.jsonl"
    if not p.exists():
        return 0
    return sum(1 for ln in p.read_text().splitlines() if ln.strip())


def verl_overrides(cfg: dict, sft_ckpt: str, out_dir: str,
                   attn: str = "sdpa", tp: int = 1, gpus: int = 1) -> list[str]:
    """veRL main_ppo overrides — DAPO + the required fields VALIDATED via the smoke test.
    `+`-prefixed keys are added (not in base main_ppo config); the rest are base keys.
    attn=sdpa avoids flash-attn on aarch64; tp=1/gpus=1 for the single-GPU GB10."""
    a = cfg["algorithm"]
    r = cfg["rollout"]
    lora = cfg.get("policy_lora", {})
    parquet = f"{config.ROOT}/data/datasets/rl_verl_train.parquet"
    # batch must not exceed the task pool — veRL's dataloader (drop_last) would otherwise
    # yield ZERO batches and the run "trains" on nothing.
    batch = max(1, min(int(a["prompts_per_batch"]), _n_tasks() or int(a["prompts_per_batch"])))
    ov = [
        # --- GRPO base + DAPO clip-higher / token-level loss (base actor keys) ---
        "algorithm.adv_estimator=grpo",
        f"actor_rollout_ref.actor.clip_ratio_low={a['clip_low']}",
        f"actor_rollout_ref.actor.clip_ratio_high={a['clip_high']}",
        "actor_rollout_ref.actor.loss_agg_mode=token-mean",
        f"actor_rollout_ref.actor.optim.lr={a['lr']}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={batch}",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        f"algorithm.use_kl_in_reward={'false' if a['kl_free'] else 'true'}",
        f"actor_rollout_ref.actor.use_kl_loss={'false' if a['kl_free'] else 'true'}",
        # --- data (prompt window sized for issue-text prompts; overlong filtered, not crashed) ---
        f"data.train_files={parquet}", f"data.val_files={parquet}",
        f"data.train_batch_size={batch}",
        "data.max_prompt_length=2560",
        f"data.max_response_length={r['max_tokens']}",
        "data.filter_overlong_prompts=true",
        # --- model / rollout (validated on the live GB10 RL proof) ---
        f"actor_rollout_ref.model.path={sft_ckpt}",
        f"+actor_rollout_ref.model.override_config.attn_implementation={attn}",
        "actor_rollout_ref.model.use_remove_padding=false",
        f"actor_rollout_ref.model.lora_rank={lora.get('r', 32)}",
        f"actor_rollout_ref.model.lora_alpha={lora.get('alpha', 64)}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.n={r['samples_per_task']}",          # group size G
        f"actor_rollout_ref.rollout.temperature={r['temperature']}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp}",
        # unified-memory lessons (see module docstring):
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.2",
        "actor_rollout_ref.rollout.max_num_seqs=8",
        "actor_rollout_ref.rollout.max_model_len=4096",
        "actor_rollout_ref.rollout.enforce_eager=true",
        "+actor_rollout_ref.rollout.enable_sleep_mode=false",
        "actor_rollout_ref.rollout.free_cache_engine=false",
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


def build_command(cfg: dict, sft_ckpt: str, out_dir: str) -> list[str]:
    return [sys.executable, "-m", "verl.trainer.main_ppo",
            *verl_overrides(cfg, sft_ckpt, out_dir)]


def run(dry_run: bool = False,
        sft_ckpt: str = "models/student_merged", out_dir: str = "models/rl") -> None:
    """sft_ckpt must be a FULL model dir (e.g. models/student_merged) — veRL's from_pretrained
    cannot load a LoRA-adapter-only dir like models/sft; merge first (src.train.merge_student)."""
    cfg = config.load("rl")
    if not (config.ROOT / sft_ckpt / "config.json").exists():
        raise SystemExit(f"[RL] {sft_ckpt} is not a full model dir (no config.json) — "
                         "run `python -m src.train.merge_student` first")
    cmd = build_command(cfg, sft_ckpt, out_dir)
    print("[RL] veRL entrypoint:\n  " + " ".join(cmd))
    if dry_run:
        return
    env = {**os.environ,
           "RAY_memory_usage_threshold": os.environ.get("RAY_memory_usage_threshold", "0.98"),
           "PATH": os.path.expanduser("~/.cargo/bin") + os.pathsep + os.environ.get("PATH", "")}
    subprocess.run(cmd, cwd=str(config.ROOT), check=True, env=env)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sft-ckpt", default="models/student_merged")
    ap.add_argument("--out", default="models/rl")
    a = ap.parse_args()
    run(dry_run=a.dry_run, sft_ckpt=a.sft_ckpt, out_dir=a.out)
