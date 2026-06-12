"""Synthetic, EXECUTION-verifiable RL tasks via bug-injection (mutation testing).

cargo-mutants mutates a cred-free crate's code and reports which mutants the tests CATCH.
Each caught mutant is a clean "fix the regression" task: inject the mutant (setup_patch), the
model must restore green `cargo test -p <crate>` (no creds/DB). This is the plan's R4 mitigation
for thin real-PR supply, and small mutants double as the cold-start easy ladder — a subset is
ALSO emitted as the trivial-tier ignition suite (eval_sets/trivial/, gauge-only: it measures
harness mechanics, is never used for promotion, so overlap with training is acceptable).

Run cargo-mutants first (per crate), e.g.:
    cargo mutants -p euclid --timeout 180 -j 4         # writes mutants.out/
then: python -m src.data.build_rl_synthetic --crate euclid --mutants-out data/hyperswitch/mutants.out

Appends RLTasks (verify_mode=execution) to data/datasets/rl_tasks.jsonl (idempotent: task ids
are content-hashed, duplicates are skipped on re-run), then REBUILDS the veRL parquet so the
new tasks actually reach training (previously a silent gap).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from src import config
from src.data.build_rl import RLTask

TRIVIAL_SUITE_FRAC = 0.25  # this share of caught mutants doubles as the cold-start gauge


def _git_appliable(diff_text: str, file_path: str) -> str:
    """cargo-mutants diffs use a descriptive '+++ replace ... ' header and bare paths. Rewrite to a
    standard `git apply` patch (--- a/<path> / +++ b/<path>). ONLY the first ---/+++ header pair is
    rewritten — body lines that merely START with '--- '/'+++ ' (removed SQL comments, doc-comment
    separators) must pass through untouched or the hunk corrupts."""
    out, fixed_minus, fixed_plus = [], False, False
    for ln in diff_text.splitlines():
        if not fixed_minus and ln.startswith("--- "):
            out.append(f"--- a/{file_path}"); fixed_minus = True
        elif not fixed_plus and ln.startswith("+++ "):
            out.append(f"+++ b/{file_path}"); fixed_plus = True
        else:
            out.append(ln)
    return "\n".join(out) + "\n"


def build(crate: str, mutants_out: str, repo: str) -> list[RLTask]:
    out_dir = config.ROOT / mutants_out
    outcomes = json.loads((out_dir / "outcomes.json").read_text())["outcomes"]
    # parent MUST be valid: an empty SHA makes compute_score return 0.0 for every task, silently.
    parent = subprocess.run(["git", "-C", str(config.ROOT / repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True, check=True).stdout.strip()
    if not parent:
        raise RuntimeError(f"could not resolve HEAD of {repo} — refusing to emit tasks")
    tasks: list[RLTask] = []
    for oc in outcomes:
        if oc.get("summary") != "CaughtMutant":
            continue
        mut = oc["scenario"]["Mutant"]
        if mut.get("package") != crate:
            continue
        file = mut["file"]
        func = (mut.get("function") or {}).get("function_name", "?")
        diff = _git_appliable((out_dir / oc["diff_path"]).read_text(), file)
        # content-hashed id => idempotent re-runs (same mutant -> same id -> skipped by dedup)
        hid = hashlib.sha256(f"{crate}|{file}|{diff}".encode()).hexdigest()[:8]
        tasks.append(RLTask(
            task_id=f"hs-mut-{crate}-{hid}",
            pr_number=0,
            parent_commit=parent,
            issue_text=(f"A regression was introduced in `{func}` ({file}) — the unit tests in "
                        f"crate `{crate}` now fail. Fix the implementation so `cargo test -p "
                        f"{crate}` passes again. Do not modify the tests."),
            gold_files=[file],
            test_files=[f"crates/{crate}/ (cargo test -p {crate})"],
            crates=[crate],
            single_crate=True,
            verify_cmd=f"cargo test -p {crate}",
            verify_mode="execution",
            setup_patch=diff,          # inject the bug before the model fixes it
        ))
    return tasks


def emit_trivial_suite(tasks: list[RLTask], out_dir: Path) -> int:
    """Cold-start ignition gauge (flywheel.cold_start reads eval_sets/trivial/*.json).
    Gauge-only — never used for promotion — so sourcing it from training mutants is fine."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = max(1, int(len(tasks) * TRIVIAL_SUITE_FRAC)) if tasks else 0
    for t in tasks[:n]:
        rec = asdict(t)
        rec["task_id"] = f"triv-{t.task_id}"     # distinct id-space from the RL pool
        (out_dir / f"{rec['task_id']}.json").write_text(json.dumps(rec, indent=2))
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crate", required=True)
    ap.add_argument("--mutants-out", default="data/hyperswitch/mutants.out")
    ap.add_argument("--repo", default="data/hyperswitch")
    ap.add_argument("--out", default="data/datasets/rl_tasks.jsonl")
    a = ap.parse_args()
    tasks = build(a.crate, a.mutants_out, a.repo)
    out = config.ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    have = set()
    if out.exists():                            # dedup against everything already there
        for ln in out.read_text().splitlines():
            try:
                have.add(json.loads(ln).get("task_id"))
            except json.JSONDecodeError:
                pass
    fresh = [t for t in tasks if t.task_id not in have]
    with out.open("a") as f:                    # APPEND to the existing (pattern) RL tasks
        for t in fresh:
            f.write(json.dumps(asdict(t)) + "\n")
    n_exec = sum(1 for ln in out.read_text().splitlines() if '"execution"' in ln)
    print(f"+{len(fresh)} execution tasks from {a.crate} ({len(tasks) - len(fresh)} dupes skipped)"
          f" -> {a.out} (total execution tasks now {n_exec})", flush=True)

    n_triv = emit_trivial_suite(tasks, config.ROOT / "eval_sets/trivial")
    print(f"trivial ignition suite: {n_triv} tasks -> eval_sets/trivial/", flush=True)

    # REBUILD the veRL parquet — without this the appended tasks never reach training.
    from src.data import build_verl
    n = build_verl.build()
    print(f"veRL parquet rebuilt: {n} tasks", flush=True)


if __name__ == "__main__":
    main()
