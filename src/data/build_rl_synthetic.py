"""Synthetic, EXECUTION-verifiable RL tasks via bug-injection (mutation testing).

cargo-mutants mutates a cred-free crate's code and reports which mutants the tests CATCH.
Each caught mutant is a clean "fix the regression" task: inject the mutant (setup_patch), the
model must restore green `cargo test -p <crate>` (no creds/DB). This is the plan's R4 mitigation
for thin real-PR supply, and small mutants double as the cold-start easy ladder.

Run cargo-mutants first (per crate), e.g.:
    cargo mutants -p euclid --timeout 180 -j 4         # writes mutants.out/
then: python -m src.data.build_rl_synthetic --crate euclid --mutants-out data/hyperswitch/mutants.out

Appends RLTasks (verify_mode=execution) to data/datasets/rl_tasks.jsonl.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from pathlib import Path

from src import config
from src.data.build_rl import RLTask


def _git_appliable(diff_text: str, file_path: str) -> str:
    """cargo-mutants diffs use a descriptive '+++ replace ... ' header and bare paths. Rewrite to a
    standard `git apply` patch (--- a/<path> / +++ b/<path>)."""
    out = []
    for ln in diff_text.splitlines():
        if ln.startswith("--- "):
            out.append(f"--- a/{file_path}")
        elif ln.startswith("+++ "):
            out.append(f"+++ b/{file_path}")
        else:
            out.append(ln)
    return "\n".join(out) + "\n"


def build(crate: str, mutants_out: str, repo: str) -> list[RLTask]:
    out_dir = config.ROOT / mutants_out
    outcomes = json.loads((out_dir / "outcomes.json").read_text())["outcomes"]
    parent = subprocess.run(["git", "-C", str(config.ROOT / repo), "rev-parse", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
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
        n = len(tasks) + 1
        tasks.append(RLTask(
            task_id=f"hs-mut-{crate}-{n:03d}",
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
    with out.open("a") as f:                # APPEND to the existing (pattern) RL tasks
        for t in tasks:
            f.write(json.dumps(asdict(t)) + "\n")
    n_exec = sum(1 for ln in out.read_text().splitlines() if '"execution"' in ln)
    print(f"+{len(tasks)} execution tasks from {a.crate} -> {a.out} "
          f"(total execution tasks now {n_exec})", flush=True)


if __name__ == "__main__":
    main()
