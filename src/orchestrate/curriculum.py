"""Two-mode difficulty scheduler (resolves cold-start vs 'drop 0%-solve' conflict, Risk R1).

Per task we track a rolling solve-rate. A task is:
  * RL-eligible  if solve-rate is inside the band (e.g. 0.2-0.8) — real GRPO signal;
  * SFT-on-gold  if below the band (student can't touch it yet) — distill the gold;
  * graduated    if above the band (too easy) — drop from active pool.
As the model improves, tasks migrate up; difficulty escalates automatically. Also
enforces the connector-diversity cap (Risk R4).
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Scheduler:
    band: tuple[float, float]
    diversity_max_connector_frac: float = 0.4
    history: dict[str, list[bool]] = field(default_factory=dict)

    def record(self, task_id: str, solved: bool) -> None:
        self.history.setdefault(task_id, []).append(solved)
        self.history[task_id] = self.history[task_id][-10:]  # rolling window

    def solve_rate(self, task_id: str) -> float:
        h = self.history.get(task_id)
        if not h:
            return 0.0
        return sum(h) / len(h)

    def classify(self, task_id: str) -> str:
        r = self.solve_rate(task_id)
        lo, hi = self.band
        if r < lo:
            return "sft_on_gold"
        if r > hi:
            return "graduated"
        return "rl_eligible"

    def select_rl_batch(self, tasks: list[dict], n: int) -> list[dict]:
        """Pick an RL batch in-band, capping the connector fraction for diversity."""
        eligible = [t for t in tasks if self.classify(t["task_id"]) == "rl_eligible"]
        connectors = [t for t in eligible if "hyperswitch_connectors" in t.get("crates", [])]
        others = [t for t in eligible if t not in connectors]
        max_conn = int(n * self.diversity_max_connector_frac)
        batch = others[: n - max_conn] + connectors[:max_conn]
        return batch[:n]

    def sft_on_gold_tasks(self, tasks: list[dict]) -> list[dict]:
        return [t for t in tasks if self.classify(t["task_id"]) == "sft_on_gold"]
