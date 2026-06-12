"""M1 — task-supply census. THE HARD GATE before spending CPT compute (Risk R4).

The plan's biggest unstated assumption is that enough cheaply-verifiable issue->fix
tasks exist. This module measures it: counts candidate tasks, and OPTIONALLY attempts
to reproduce a sampled subset (parent build + FAIL_TO_PASS check) to estimate the
reproducible yield. Emits a go/no-go verdict against the configured threshold.

Cheap census (no builds) runs anywhere. ``--verify-sample N`` attempts real builds in
the hardened container and needs the env from src.harness.runner.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass

from src import config
from src.data import build_rl

def _gate_min() -> int:
    """Census floor — configurable (flywheel.yaml: census_min_tasks). The plan's M1 default was
    150 real-PR tasks; with the R4 rescope (synthetic bug-injection + pattern/execution-free
    reward) a much smaller execution-verifiable core is viable, so the floor must be tunable
    rather than a hardcoded constant that permanently NO-GOs Phase C."""
    try:
        return int(config.load("flywheel").get("census_min_tasks", 150))
    except Exception:
        return 150


@dataclass
class Census:
    total_prs_with_tests: int
    single_crate: int
    multi_crate: int
    execution_verifiable: int     # cheap unit-test crates -> real RLVR (the gate metric)
    pattern_verifiable: int       # connectors/integration -> compile+similarity (execution-free)
    crate_histogram: dict[str, int]
    verified_sample: int
    verified_reproducible: int
    estimated_reproducible_total: int
    gate_min: int
    passes_gate: bool
    note: str


def _load_task_pool(cfg: dict) -> list:
    """The REAL task pool = data/datasets/rl_tasks.jsonl (mined PRs + appended synthetic
    bug-injection tasks). Rebuilding from PRs here would silently exclude the synthetic
    execution tasks — the very pool the gate is supposed to measure. Falls back to a
    fresh build only when the file doesn't exist yet."""
    pool = config.ROOT / "data/datasets/rl_tasks.jsonl"
    if pool.exists():
        out = []
        for ln in pool.read_text().splitlines():
            try:
                d = json.loads(ln)
            except json.JSONDecodeError:
                continue
            known = {f.name for f in __import__("dataclasses").fields(build_rl.RLTask)}
            out.append(build_rl.RLTask(**{k: v for k, v in d.items() if k in known}))
        return out
    return build_rl.build(cfg)


def run(cfg: dict, verify_sample: int = 0) -> Census:
    tasks = _load_task_pool(cfg)
    single = [t for t in tasks if t.single_crate]
    # the gate cares about EXECUTION-verifiable tasks (cheap unit tests, no creds) — the
    # real RLVR pool. Connector/integration tasks ("pattern") verify by compile+similarity.
    execution = [t for t in single if t.verify_mode == "execution"]
    pattern = [t for t in tasks if t.verify_mode == "pattern"]
    hist: dict[str, int] = {}
    for t in execution:
        hist[t.crates[0]] = hist.get(t.crates[0], 0) + 1

    verified_ok = 0
    verified_n = 0
    if verify_sample > 0:
        # Lazy import: needs the container env. Attempt to reproduce FAIL_TO_PASS on a sample.
        from src.harness import runner  # noqa: WPS433

        sample = execution[:verify_sample]   # only execution-mode tasks are FAIL->PASS verifiable
        verified_n = len(sample)
        for t in sample:
            try:
                verified_ok += int(runner.verify_fail_to_pass(t))
            except Exception as exc:  # pragma: no cover - env dependent
                print(f"  verify {t.task_id}: error {exc}")

    repro_rate = (verified_ok / verified_n) if verified_n else 1.0
    est_total = int(len(execution) * repro_rate)
    gate_min = _gate_min()
    passes = est_total >= gate_min

    return Census(
        total_prs_with_tests=len(tasks),
        single_crate=len(single),
        multi_crate=len(tasks) - len(single),
        execution_verifiable=len(execution),
        pattern_verifiable=len(pattern),
        crate_histogram=dict(sorted(hist.items(), key=lambda kv: -kv[1])),
        verified_sample=verified_n,
        verified_reproducible=verified_ok,
        estimated_reproducible_total=est_total,
        gate_min=gate_min,
        passes_gate=passes,
        note=(
            f"GO: {est_total} execution-verifiable RL tasks (>= {gate_min})." if passes else
            f"NO-GO: only ~{est_total} execution-verifiable tasks (< {gate_min}). "
            f"Lean RL on the {len(pattern)} pattern tasks via compile+clippy+patch-similarity "
            f"(execution-free) + synthetic connector tasks; reframe the headline claim."
        ),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-sample", type=int, default=0,
                    help="attempt real parent builds for N sampled single-crate tasks")
    args = ap.parse_args()

    cfg = config.load("data")
    census = run(cfg, verify_sample=args.verify_sample)
    out = config.RUNS / "m1_census.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(asdict(census), indent=2))

    print(json.dumps(asdict(census), indent=2))
    print(f"\n==> {census.note}")
    raise SystemExit(0 if census.passes_gate else 2)


if __name__ == "__main__":
    main()
