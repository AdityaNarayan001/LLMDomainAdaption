"""Persistent rollout store — every raw rollout, not just harvested winners.

Why persist ALL rollouts (incl. failures):
  * RL reproducibility — DAPO updates from the exact token-faithful trajectories;
  * debugging / reward-hacking forensics — inspect what the model actually did;
  * decouple generation from training (offline-rollout optimization, Risk R5);
  * lineage — link the rollout shard to the checkpoint that produced it (Risk R10).

Layout: runs/rollouts/cycle_<n>.jsonl(.gz), one JSON object per rollout:
  {task_id, model, reward, breakdown, messages, token_ids, logprobs, loss_mask}

token_ids/logprobs make these large, so shards are gzip-compressed by default.
"""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from src import config

ROLLOUTS = config.RUNS / "rollouts"


def shard_path(cycle: int, gzipped: bool = True) -> Path:
    return ROLLOUTS / f"cycle_{cycle}.jsonl{'.gz' if gzipped else ''}"


def save(cycle: int, rollouts: list[dict], model: str, gzipped: bool = True) -> Path:
    """Append a cycle's raw rollouts to its shard. Returns the shard path."""
    ROLLOUTS.mkdir(parents=True, exist_ok=True)
    path = shard_path(cycle, gzipped)
    opener = (lambda p: gzip.open(p, "at")) if gzipped else (lambda p: open(p, "a"))
    with opener(path) as f:
        for r in rollouts:
            f.write(json.dumps({**r, "model": model}) + "\n")
    return path


def load(cycle: int) -> list[dict]:
    """Read back a cycle's rollouts (auto-detects gzip)."""
    gz, plain = shard_path(cycle, True), shard_path(cycle, False)
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            return [json.loads(line) for line in f]
    if plain.exists():
        return [json.loads(line) for line in plain.read_text().splitlines()]
    return []


def stats(cycle: int) -> dict:
    """Quick per-cycle summary: counts, solve-rate, reward distribution."""
    rolls = load(cycle)
    n = len(rolls)
    solved = sum(1 for r in rolls if (r.get("breakdown") or {}).get("tests", 0.0) >= 1.0)
    rewards = [r.get("reward", 0.0) for r in rolls]
    return {
        "cycle": cycle,
        "rollouts": n,
        "solved": solved,
        "solve_rate": solved / n if n else 0.0,
        "mean_reward": sum(rewards) / n if n else 0.0,
    }
