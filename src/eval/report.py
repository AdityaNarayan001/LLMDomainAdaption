"""Comparison table + the promotion GATE (paired McNemar, not difference-of-means).

Risk R3: comparing two solve-rate means on 50-100 tasks is underpowered. A paired
per-task win/loss test (McNemar) is far more sensitive to the small (0.5-3pp) gains a
flywheel cycle produces. We also compute the minimum detectable effect so a broken gate
(MDE > expected gain) is caught up front, and report cost/latency as a first-class axis.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

from scipy import stats

from src import config


@dataclass
class GateResult:
    candidate: str
    incumbent: str
    n: int
    candidate_only_wins: int   # candidate solved, incumbent didn't
    incumbent_only_wins: int   # incumbent solved, candidate didn't
    p_value: float
    promote: bool
    mde_pp: float              # minimum detectable effect (percentage points) at this n


def _solved_map(path) -> dict[str, bool]:
    data = json.loads(open(path).read())
    return {t["task_id"]: bool(t["solved"]) for t in data["per_task"]}


def mcnemar_gate(candidate_path, incumbent_path, alpha: float = 0.05) -> GateResult:
    cand = _solved_map(candidate_path)
    inc = _solved_map(incumbent_path)
    ids = sorted(set(cand) & set(inc))
    b = sum(1 for i in ids if cand[i] and not inc[i])   # candidate-only
    c = sum(1 for i in ids if inc[i] and not cand[i])   # incumbent-only
    n = len(ids)
    # exact McNemar (binomial on discordant pairs)
    discordant = b + c
    if discordant == 0:
        p = 1.0
    else:
        p = stats.binomtest(min(b, c), discordant, 0.5).pvalue
    promote = (p < alpha) and (b > c)
    # MDE: smallest per-task win-rate delta detectable at n, alpha, 80% power (normal approx)
    z_a, z_b = stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(0.8)
    mde = ((z_a + z_b) / math.sqrt(max(1, n))) * 100  # ~pp, rough
    return GateResult(
        candidate=str(candidate_path), incumbent=str(incumbent_path),
        n=n, candidate_only_wins=b, incumbent_only_wins=c,
        p_value=p, promote=promote, mde_pp=round(mde, 2),
    )


def comparison_table(result_paths: list[str]) -> str:
    """Render a model-vs-baseline table: solve-rate / avg-steps / tokens / $ / latency."""
    rows = []
    for p in result_paths:
        d = json.loads(open(config.RUNS / "eval" / p).read())
        tasks = d["per_task"]
        n = max(1, len(tasks))
        rows.append((
            d["model"], d["solve_rate"],
            sum(t["steps"] for t in tasks) / n,
            sum(t["tokens"] for t in tasks) / n,
            sum(t["usd"] for t in tasks),
            sum(t["latency_s"] for t in tasks) / n,
        ))
    out = ["| model | solve | avg_steps | tok/task | $ total | lat/task |",
           "|---|---|---|---|---|---|"]
    for m, s, st, tk, usd, lat in rows:
        out.append(f"| {m} | {s:.1%} | {st:.1f} | {tk:.0f} | ${usd:.2f} | {lat:.1f}s |")
    return "\n".join(out)
