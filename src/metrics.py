"""Unified metrics + structured logging across all stages.

Every stage logs through here so a long unattended flywheel is observable:
  * timestamped, leveled console output (rich) — readable interactively;
  * a structured JSONL sink per stage at runs/metrics/<stage>.jsonl — machine-readable,
    so the flywheel ledger and dashboards can aggregate without scraping console text;
  * helpers: Wilson confidence interval (for solve-rate), a stage timer, progress bars.

Dependency-light: rich is a core dep. Timestamps are passed in (libs avoid wall-clock
calls that would break workflow resume) — callers stamp via time.time().
"""
from __future__ import annotations

import json
import math
import time
from contextlib import contextmanager
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from src import config

_console = Console()
_METRICS = config.RUNS / "metrics"


def log(stage: str, event: str, **fields) -> None:
    """Emit one structured metric record: console line + append to runs/metrics/<stage>.jsonl."""
    _METRICS.mkdir(parents=True, exist_ok=True)
    rec = {"ts": time.time(), "stage": stage, "event": event, **fields}
    with (_METRICS / f"{stage}.jsonl").open("a") as f:
        f.write(json.dumps(rec) + "\n")
    extra = "  ".join(f"[cyan]{k}[/]={v}" for k, v in fields.items())
    _console.print(f"[dim]{time.strftime('%H:%M:%S')}[/] [bold]{stage}[/]:{event}  {extra}")


@contextmanager
def stage_timer(stage: str, what: str):
    """Time a block and log its duration (throughput visibility, Risk R5)."""
    t0 = time.time()
    log(stage, f"{what}.start")
    try:
        yield
    finally:
        log(stage, f"{what}.done", seconds=round(time.time() - t0, 1))


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial rate — proper CI on solve-rate (small n)."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def progress(iterable, stage: str, what: str):
    """Lightweight progress over a long loop (eval suite, rollout generation)."""
    from rich.progress import track

    return track(iterable, description=f"{stage}:{what}")


@dataclass
class Summary:
    """Render a clean per-run summary table to the console."""

    title: str
    rows: list[tuple[str, str]]

    def show(self) -> None:
        t = Table(title=self.title)
        t.add_column("metric")
        t.add_column("value", justify="right")
        for k, v in self.rows:
            t.add_row(k, v)
        _console.print(t)
