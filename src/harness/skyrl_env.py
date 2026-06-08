"""SkyRL env adapter — wraps a hyperswitch issue->fix task as an RL environment.

Integration caveat (Risk R-int): SkyRL natively integrates OpenHands, NOT Pi. We drive
the Pi harness (src.harness.runner) and expose the result in SkyRL-Agent's expected
shape via a Polar-style API proxy (token-faithful trajectory). Validating this bridge
is an M0 deliverable; if heavy, fall back to OpenHands (SkyRL-native) or veRL.

This module is import-light; the actual skyrl-agent base class is imported lazily so
the rest of the package works without the `rl` extra installed.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.harness import runner


@dataclass
class HyperswitchTaskEnv:
    """One task = one episode. Produces a token-faithful trajectory + scalar reward."""

    task: dict
    endpoint: str
    model: str
    weights: dict
    temperature: float = 1.0
    max_turns: int = 40
    closed_context: bool = False

    def rollout(self) -> dict:
        """Generate one trajectory. Shape matches what skyrl-train expects to consume."""
        traj = runner.run_task(
            self.task,
            endpoint=self.endpoint,
            model=self.model,
            weights=self.weights,
            temperature=self.temperature,
            max_turns=self.max_turns,
            closed_context=self.closed_context,
        )
        return {
            "task_id": self.task["task_id"],
            "messages": traj.messages,
            "token_ids": traj.token_ids,     # token-faithful (for loss/advantage alignment)
            "logprobs": traj.logprobs,
            "loss_mask": traj.loss_mask,     # assistant-only
            "reward": traj.reward,
            "breakdown": traj.breakdown,
        }


def make_envs(tasks: list[dict], endpoint: str, model: str, weights: dict, **kw):
    """Factory: SkyRL drives many of these concurrently (async trajectory dispatch)."""
    return [HyperswitchTaskEnv(task=t, endpoint=endpoint, model=model, weights=weights, **kw)
            for t in tasks]
