"""Stage 4 — RL launcher. Drives NovaSky SkyRL's config-driven entrypoint.

SkyRL is NOT a `from X import Trainer` API — it is run as a config-driven entrypoint:
    python -m skyrl.train.entrypoints.main_base <hydra/omegaconf overrides>
with the algorithm (GRPO/PPO/DAPO), model, dataset, reward, and generation backend
selected via config. This module is executed *inside* .venv-rl (the SkyRL uv venv, via
the venv router), so `skyrl` is importable here.

Generation backend = our **.venv-serve vLLM** reached over a remote OpenAI endpoint
(SkyRL supports a remote inference engine via base_url). This is also what makes RL work
on the aarch64 GB10, where SkyRL's *bundled* x86_64 vLLM/router doesn't install.

NOTE (M3): the exact override KEYS below must be finalized against SkyRL's config schema
(see SkyRL/skyrl/train/config/) and our env must be registered as a skyrl-gym env. The
shape here (entrypoint + overrides + remote engine) is correct; the precise field names
are the remaining RL-stage integration task.
"""
from __future__ import annotations

import subprocess
import sys

from src import config


def skyrl_overrides(cfg: dict, endpoint: str, sft_ckpt: str) -> list[str]:
    """Map our configs/rl.yaml -> SkyRL hydra overrides (key names are M3 TODO)."""
    a = cfg["algorithm"]
    r = cfg["rollout"]
    return [
        f"trainer.algorithm={a['name']}",            # dapo
        f"trainer.policy.model.path={sft_ckpt}",
        f"trainer.policy.lora.rank={cfg['policy_lora']['r']}",
        f"trainer.algorithm.group_size={a['group_size']}",
        f"trainer.algorithm.lr={a['lr']}",
        f"trainer.algorithm.kl_coef={0.0 if a['kl_free'] else a['kl_beta']}",
        # remote generation engine = our .venv-serve vLLM (aarch64-friendly path)
        "generator.backend=remote",
        f"generator.inference_engine.base_url={endpoint}",
        f"generator.sampling.n={r['samples_per_task']}",
        f"generator.sampling.temperature={r['temperature']}",
        f"generator.sampling.max_tokens={r['max_tokens']}",
        # our task env + dataset (registered as a skyrl-gym env — M3)
        "environment.env_class=hyperswitch",
        f"data.train_data={config.ROOT}/data/datasets/rl_tasks.jsonl",
    ]


def build_command(cfg: dict, endpoint: str, sft_ckpt: str) -> list[str]:
    return [sys.executable, "-m", "skyrl.train.entrypoints.main_base",
            *skyrl_overrides(cfg, endpoint, sft_ckpt)]


def run(dry_run: bool = False, endpoint: str = "http://localhost:8000",
        sft_ckpt: str = "models/sft") -> None:
    cfg = config.load("rl")
    cmd = build_command(cfg, endpoint, sft_ckpt)
    print("[RL] SkyRL entrypoint:\n  " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(config.ROOT), check=True)


if __name__ == "__main__":
    run(dry_run="--dry-run" in sys.argv)
