"""Checkpoint + data lineage registry (Risk R10).

A "checkpoint" is a TUPLE: base-model hash + adapter stack (CPT/SFT/RL) + the
training-data manifest hash + its eval result. We never repeatedly merge adapters, so
the deployable model is this tuple. Lineage makes any cycle reproducible and any bad
promotion reversible.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from src import config


@dataclass
class Checkpoint:
    cycle: int
    base_model: str
    adapter_stack: list[str]          # e.g. ["models/cpt/pr_mastery", "models/sft", "models/rl/c3"]
    data_manifest_hash: str
    solve_rate: float
    created_at: float = field(default_factory=lambda: 0.0)  # stamped by caller (no Date.now in libs)


class Registry:
    def __init__(self, path: str = "runs/registry.jsonl"):
        self.path = config.ROOT / path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, ckpt: Checkpoint) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(ckpt)) + "\n")

    def all(self) -> list[Checkpoint]:
        if not self.path.exists():
            return []
        return [Checkpoint(**json.loads(line)) for line in self.path.read_text().splitlines()]

    def best(self) -> Checkpoint | None:
        ck = self.all()
        return max(ck, key=lambda c: c.solve_rate) if ck else None

    def promote(self, candidate: Checkpoint, incumbent: Checkpoint | None,
                gate_promote: bool) -> Checkpoint:
        """Return whichever becomes the new champion. Promote only if the gate fired."""
        if incumbent is None or gate_promote:
            self.record(candidate)
            return candidate
        return incumbent  # keep incumbent; candidate not recorded as champion
