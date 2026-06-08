"""Stage 4 — RL launcher. SkyRL DAPO over the Pi harness env.

Builds the SkyRL trainer args from configs/rl.yaml, points it at the vLLM rollout
endpoint, and feeds it our HyperswitchTaskEnv (Polar-style token-faithful trajectories).
SkyRL/Ray imported lazily (the `rl` extra). veRL is the documented fallback (Risk R-int).
"""
from __future__ import annotations

import json

from src import config
from src.harness import skyrl_env


def build_trainer_kwargs(cfg: dict) -> dict:
    a = cfg["algorithm"]
    return {
        "algorithm": a["name"],                 # dapo
        "group_size": a["group_size"],
        "prompts_per_batch": a["prompts_per_batch"],
        "clip_low": a["clip_low"], "clip_high": a["clip_high"],
        "dynamic_sampling": a["dynamic_sampling"],
        "token_level_loss": a["token_level_loss"],
        "soft_overlong_penalty": a["soft_overlong_penalty"],
        "kl_coef": a["kl_beta"] if not a["kl_free"] else 0.0,
        "learning_rate": a["lr"],
        "lora": cfg["policy_lora"],
        "rollout": cfg["rollout"],
    }


def load_tasks(cfg: dict) -> list[dict]:
    """RL task pool, filtered to the curriculum solve band by the curriculum scheduler."""
    path = config.ROOT / "data/datasets/rl_tasks.jsonl"
    tasks = [json.loads(line) for line in path.read_text().splitlines()]
    if cfg["curriculum"].get("two_mode"):
        # cold-start: tasks below the band are handled as SFT-on-gold by the flywheel,
        # not here; RL trains on the in-band set.
        pass
    return tasks


def run(dry_run: bool = False, endpoint: str = "http://localhost:8000",
        sft_ckpt: str = "models/sft") -> None:
    cfg = config.load("rl")
    kwargs = build_trainer_kwargs(cfg)
    tasks = load_tasks(cfg)
    envs = skyrl_env.make_envs(
        tasks, endpoint=endpoint, model=sft_ckpt, weights=cfg["reward"]["weights"],
        temperature=cfg["rollout"]["temperature"], max_turns=cfg["rollout"]["max_turns"],
    )
    print(f"[RL] DAPO over {len(envs)} envs; kwargs={json.dumps(kwargs)[:200]}...")
    if dry_run:
        return
    # Lazy import keeps the package usable without the rl extra.
    from skyrl_train import DAPOTrainer  # type: ignore

    trainer = DAPOTrainer(base_model=sft_ckpt, rollout_endpoint=endpoint, **kwargs)
    trainer.fit(envs)
    trainer.save("models/rl")


if __name__ == "__main__":
    import sys
    run(dry_run="--dry-run" in sys.argv)
