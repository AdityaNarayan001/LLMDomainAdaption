"""HS-bench runner: run a model in the Pi harness over a suite, score solve-rate.

Solve = patch applies AND structured nextest shows FAIL->PASS & PASS->PASS (no log
scraping). Reports per-task win/loss (for the paired McNemar gate), plus cost/latency
as a first-class axis (the local-9B selling point). Multiple seeds for CIs.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from src import config, metrics
from src.harness import skyrl_env


@dataclass
class TaskResult:
    task_id: str
    solved: bool
    reward: float
    steps: int
    tokens: int
    latency_s: float
    usd: float


@dataclass
class SuiteResult:
    suite: str
    model: str
    seed: int
    per_task: list[TaskResult] = field(default_factory=list)

    @property
    def solve_rate(self) -> float:
        return sum(t.solved for t in self.per_task) / max(1, len(self.per_task))


def load_suite(path: str) -> list[dict]:
    tasks: list[dict] = []
    p = config.ROOT / path
    for jf in sorted(p.glob("*.json")):
        tasks.append(json.loads(jf.read_text()))
    return tasks


def run_suite(
    suite_name: str,
    suite_path: str,
    *,
    model: str,
    endpoint: str,
    weights: dict,
    seed: int = 0,
    closed_context: bool = False,
) -> SuiteResult:
    tasks = load_suite(suite_path)
    res = SuiteResult(suite=suite_name, model=model, seed=seed)
    for task in metrics.progress(tasks, "eval", f"{suite_name}/{model}"):
        env = skyrl_env.HyperswitchTaskEnv(
            task=task, endpoint=endpoint, model=model, weights=weights,
            temperature=0.0 if seed == 0 else 0.7, closed_context=closed_context,
        )
        t0 = time.time()
        roll = env.rollout()
        latency = time.time() - t0
        bd = roll.get("breakdown") or {}
        n_tokens = sum(len(t) for t in roll.get("token_ids", []))
        res.per_task.append(
            TaskResult(
                task_id=roll["task_id"],
                solved=bool(bd.get("tests", 0.0) >= 1.0),
                reward=roll["reward"],
                steps=sum(1 for m in roll["messages"] if m["role"] == "assistant"),
                tokens=n_tokens,
                latency_s=latency,
                usd=_estimate_cost(model, n_tokens),
            )
        )
    solved = sum(t.solved for t in res.per_task)
    lo, hi = metrics.wilson_ci(solved, len(res.per_task))
    metrics.log("eval", "suite_done", suite=suite_name, model=model, seed=seed,
                n=len(res.per_task), solve_rate=round(res.solve_rate, 4),
                ci95=f"[{lo:.2f},{hi:.2f}]",
                mean_latency_s=round(sum(t.latency_s for t in res.per_task) / max(1, len(res.per_task)), 1),
                usd_total=round(sum(t.usd for t in res.per_task), 2))
    return res


def _estimate_cost(model: str, tokens: int) -> float:
    # local models ~ $0 marginal; API baselines priced per Mtok (rough, configurable later)
    api_rate = {"api_frontier": 5.0}.get(model, 0.0)  # $/Mtok
    return api_rate * tokens / 1_000_000


def save(result: SuiteResult) -> None:
    out = config.RUNS / "eval" / f"{result.suite}_{result.model}_{result.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "suite": result.suite, "model": result.model, "seed": result.seed,
        "solve_rate": result.solve_rate,
        "per_task": [t.__dict__ for t in result.per_task],
    }, indent=2))
