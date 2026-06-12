"""Checkpoint + data lineage registry, with disk-bounded retention + best-selection.

A "checkpoint" is a TUPLE (base + CPT + per-cycle SFT/RL adapters). The heavy base/CPT
live once and are referenced; only the small per-cycle adapter dir (`adapter_dir`) is
rotated. Retention (Risk R10 + the user's policy):

  KEEP = champion (permanent) ∪ top-K by HS-Knowledge solve-rate ∪ latest (resume).
  Everything else is pruned from disk. Adapters are tiny, so K=5 costs ~1-2GB.

Two-tier "best" (so we keep the genuinely best, not the luckiest):
  * promotion / retention rank → HS-Knowledge solve-rate + the McNemar gate (every cycle);
  * SHIPPED best (`select_best`) → the untouched SEQUESTERED set, by lower-confidence-bound,
    with an HS-SWE non-regression guard (catches forgetting). Touched only at milestones.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field, fields

from src import config


@dataclass
class Checkpoint:
    cycle: int
    base_model: str
    adapter_stack: list[str]          # [cpt, sft_cycle, rl_cycle] — stacked, not merged
    data_manifest_hash: str
    solve_rate: float                 # HS-Knowledge (promotion/retention metric)
    adapter_dir: str | None = None    # the prunable per-cycle dir (models/cycle_<N>)
    sequestered_lcb: float | None = None   # set at milestones (Tier-2 crowning)
    hs_swe_solve: float | None = None       # for the non-regression guard
    created_at: float = field(default=0.0)  # stamped by caller (no wall-clock in libs)
    promoted: bool = False            # did the McNemar gate fire? best() ranks ONLY promoted
                                      # rows — else a lucky non-promoted candidate becomes
                                      # champion and the gate is decorative


class Registry:
    def __init__(self, path: str = "runs/registry.jsonl", keep_top_k: int = 5):
        self.path = config.ROOT / path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keep_top_k = keep_top_k

    # ---- persistence ----
    def record(self, ckpt: Checkpoint) -> None:
        with self.path.open("a") as f:
            f.write(json.dumps(asdict(ckpt)) + "\n")

    def all(self) -> list[Checkpoint]:
        if not self.path.exists():
            return []
        valid = {f.name for f in fields(Checkpoint)}
        out = []
        for line in self.path.read_text().splitlines():
            d = {k: v for k, v in json.loads(line).items() if k in valid}
            out.append(Checkpoint(**d))
        return out

    def best(self) -> Checkpoint | None:
        """Champion = best among PROMOTED checkpoints (the gate must have fired). Falls back
        to all rows only when nothing has ever been promoted (cycle-0 bootstrap)."""
        cks = self.all()
        if not cks:
            return None
        gated = [c for c in cks if c.promoted]
        return max(gated or cks, key=lambda c: c.solve_rate)

    def promote(self, candidate: Checkpoint, incumbent: Checkpoint | None,
                gate_promote: bool) -> Checkpoint:
        """New champion only if the gate fired (else keep incumbent). Candidate is still
        recorded so retention can keep it in the top-K hedge."""
        candidate.promoted = bool(incumbent is None or gate_promote)
        self.record(candidate)
        if candidate.promoted:
            return candidate
        return incumbent

    # ---- retention (disk-bounded) ----
    def keepers(self) -> set[str]:
        """adapter_dirs to KEEP: champion ∪ top-K by solve-rate ∪ latest."""
        cks = [c for c in self.all() if c.adapter_dir]
        if not cks:
            return set()
        champion = max(cks, key=lambda c: c.solve_rate)
        latest = max(cks, key=lambda c: c.cycle)
        topk = sorted(cks, key=lambda c: c.solve_rate, reverse=True)[: self.keep_top_k]
        return {c.adapter_dir for c in (champion, latest, *topk)}

    def prune(self) -> tuple[set[str], list[str]]:
        """Delete adapter dirs not in keepers(). Returns (kept, deleted_paths)."""
        keep = self.keepers()
        deleted = []
        for c in self.all():
            if c.adapter_dir and c.adapter_dir not in keep:
                p = config.ROOT / c.adapter_dir
                if p.exists():
                    shutil.rmtree(p, ignore_errors=True)
                    deleted.append(c.adapter_dir)
        return keep, deleted

    # ---- Tier-2: crown the SHIPPED best on the sequestered set ----
    def select_best(self, hs_swe_floor: float | None = None) -> Checkpoint | None:
        """Among kept candidates scored on the sequestered set, pick the most robust:
        max lower-confidence-bound, gated by HS-SWE non-regression, tie-break by solve-rate."""
        cks = [c for c in self.all()
               if c.sequestered_lcb is not None and c.adapter_dir in self.keepers()]
        if hs_swe_floor is not None:
            cks = [c for c in cks if (c.hs_swe_solve or 0.0) >= hs_swe_floor]
        if not cks:
            return self.best()  # fall back to promotion-metric champion if unscored
        return max(cks, key=lambda c: (c.sequestered_lcb, c.solve_rate))
