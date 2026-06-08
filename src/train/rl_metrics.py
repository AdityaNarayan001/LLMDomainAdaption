"""RL/DAPO health metrics computed from a batch of grouped rollouts.

These are the signals that tell you the RL run is healthy (or quietly failing):

  * degenerate_group_rate — fraction of prompt-groups where ALL G samples pass or ALL
    fail. Zero advantage variance => zero gradient (the GRPO gradient-starvation failure
    DAPO's dynamic sampling targets). High and rising => wasted compute / stalled learning.
  * policy_entropy — mean per-token entropy proxy from sampled logprobs. A sharp drop =>
    entropy collapse (mode collapse); DAPO's clip-higher exists to prevent this.
  * mean_reward + component breakdown — is improvement coming from real tests or just
    style/lint (reward-hacking smell if dense/fmt rise while tests stay flat).
  * solve_rate — fraction of rollouts that actually passed tests.

Feed these to metrics.log("rl", "batch_health", ...) each RL step.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass


@dataclass
class BatchHealth:
    n_rollouts: int
    n_groups: int
    degenerate_group_rate: float
    policy_entropy: float
    mean_reward: float
    solve_rate: float
    reward_components: dict[str, float]


def _entropy_proxy(rollout: dict) -> float:
    """Mean token entropy proxy = mean(-logprob) over sampled assistant tokens."""
    lps = [lp for turn in rollout.get("logprobs", []) for lp in turn]
    if not lps:
        return 0.0
    return -sum(lps) / len(lps)


def compute(rollouts: list[dict]) -> BatchHealth:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rollouts:
        groups[r["task_id"]].append(r)

    degenerate = 0
    for g in groups.values():
        solved = [(_b(r).get("tests", 0.0) >= 1.0) for r in g]
        if all(solved) or not any(solved):  # zero variance -> no GRPO/DAPO signal
            degenerate += 1

    n = max(1, len(rollouts))
    comp_keys = ("tests", "dense", "compile", "fmt_clippy")
    comp = {k: sum(_b(r).get(k, 0.0) for r in rollouts) / n for k in comp_keys}

    return BatchHealth(
        n_rollouts=len(rollouts),
        n_groups=len(groups),
        degenerate_group_rate=degenerate / max(1, len(groups)),
        policy_entropy=sum(_entropy_proxy(r) for r in rollouts) / n,
        mean_reward=sum(r.get("reward", 0.0) for r in rollouts) / n,
        solve_rate=sum(1 for r in rollouts if _b(r).get("tests", 0.0) >= 1.0) / n,
        reward_components=comp,
    )


def _b(rollout: dict) -> dict:
    return rollout.get("breakdown") or {}


def health_alerts(h: BatchHealth, prev_entropy: float | None = None) -> list[str]:
    """Return human-readable warnings for the console / ledger."""
    alerts: list[str] = []
    if h.degenerate_group_rate > 0.6:
        alerts.append(
            f"HIGH degenerate-group rate {h.degenerate_group_rate:.0%} — most groups give no "
            f"gradient; tighten curriculum band or rely on DAPO dynamic sampling."
        )
    if prev_entropy is not None and prev_entropy > 0 and h.policy_entropy < 0.5 * prev_entropy:
        alerts.append(
            f"ENTROPY COLLAPSE: {prev_entropy:.2f} -> {h.policy_entropy:.2f} — check clip-higher."
        )
    if h.reward_components["tests"] < 0.05 and h.reward_components["fmt_clippy"] > 0.5:
        alerts.append("REWARD-HACK SMELL: style/lint reward high while test reward ~0.")
    return alerts


def to_log_fields(h: BatchHealth) -> dict:
    d = asdict(h)
    comps = d.pop("reward_components")
    return {**d, **{f"r_{k}": round(v, 3) for k, v in comps.items()}}
