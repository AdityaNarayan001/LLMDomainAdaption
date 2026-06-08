"""Build RL tasks (issue -> fix) with verifiers, from mined PRs.

Each task = {repo @ parent_commit, issue text, gold patch, FAIL_TO_PASS/PASS_TO_PASS
tests, crate, single_crate, verify_cost}. We bias toward single-crate, cheap-to-verify
tasks (Risk R4/R6). Tasks are split disjointly from the eval set (build done in M1 census).
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import asdict, dataclass

from src import config

CRATE_RE = re.compile(r"^crates/([^/]+)/")

# Grounded in repo analysis: crates whose unit tests run with NO DB/Redis/credentials
# (cheap `cargo nextest -p <crate>` -> real FAIL->PASS reward). Connectors are NOT here:
# their tests need live connector credentials + network, so they verify by compile+clippy
# + patch-similarity-to-gold instead (execution-free).
CHEAP_VERIFY_CRATES = {
    "common_utils", "euclid", "common_enums", "cards", "masking",
    "router_env", "diesel_models", "hyperswitch_interfaces",
}


def verify_mode(crates: set[str]) -> str:
    """'execution' if all touched crates are cheap-unit-testable, else 'pattern' (compile+similarity)."""
    if crates and crates.issubset(CHEAP_VERIFY_CRATES):
        return "execution"
    return "pattern"


def crates_touched(changed_files: list[str]) -> set[str]:
    out: set[str] = set()
    for f in changed_files:
        m = CRATE_RE.match(f)
        if m:
            out.add(m.group(1))
    return out


@dataclass
class RLTask:
    task_id: str
    pr_number: int
    parent_commit: str | None
    issue_text: str
    gold_files: list[str]
    test_files: list[str]
    crates: list[str]
    single_crate: bool
    verify_cmd: str            # targeted cargo nextest for cheap verification
    verify_mode: str           # "execution" (cheap unit tests) | "pattern" (compile+similarity)


def parent_of(repo, merge_commit: str | None) -> str | None:
    if not merge_commit:
        return None
    res = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{merge_commit}^"],
        capture_output=True, text=True, check=False,
    )
    return res.stdout.strip() or None


def build(cfg: dict) -> list[RLTask]:
    repo = config.ROOT / cfg["source"]["raw_repo"]
    prs_path = config.ROOT / "data/datasets/github_prs.jsonl"
    if not prs_path.exists():
        raise FileNotFoundError("run ingest_github first to produce github_prs.jsonl")

    tasks: list[RLTask] = []
    for line in prs_path.read_text().splitlines():
        pr = json.loads(line)
        if not pr["test_files"]:
            continue  # need a verifier
        touched = crates_touched(pr["changed_files"])
        single = len(touched) == 1
        crate = next(iter(touched)) if touched else None
        verify = f"cargo nextest run -p {crate}" if single and crate else "cargo nextest run"
        tasks.append(
            RLTask(
                task_id=f"hs-pr-{pr['number']}",
                pr_number=pr["number"],
                parent_commit=parent_of(repo, pr["merge_commit"]),
                issue_text=(pr["title"] + "\n\n" + pr["body"]),
                gold_files=pr["changed_files"],
                test_files=pr["test_files"],
                crates=sorted(touched),
                single_crate=single,
                verify_cmd=verify,
                verify_mode=verify_mode(touched),
            )
        )
    return tasks


def save(tasks: list[RLTask], path: str = "data/datasets/rl_tasks.jsonl") -> int:
    out = config.ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t)) + "\n")
    return len(tasks)


if __name__ == "__main__":
    cfg = config.load("data")
    tasks = build(cfg)
    single = [t for t in tasks if t.single_crate]
    save(tasks)
    print(f"RL tasks: {len(tasks)} total; {len(single)} single-crate (cheap-verify candidates)")
