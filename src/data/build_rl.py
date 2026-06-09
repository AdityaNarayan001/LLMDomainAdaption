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
from src.data.build_cpt import strip_pr_template

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
    setup_patch: str | None = None   # regression to inject before the model fixes it (synthetic
                                     # bug-injection tasks); None for real PRs (parent_commit IS broken)


def parent_of(repo, merge_commit: str | None) -> str | None:
    if not merge_commit:
        return None
    res = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", f"{merge_commit}^"],
        capture_output=True, text=True, check=False,
    )
    return res.stdout.strip() or None


def iter_tasks(cfg: dict):
    """Yield one RLTask per verifiable PR (lazy → enables incremental writing)."""
    repo = config.ROOT / cfg["source"]["raw_repo"]
    prs_path = config.ROOT / "data/datasets/github_prs.jsonl"
    if not prs_path.exists():
        raise FileNotFoundError("run ingest_github first to produce github_prs.jsonl")
    for line in prs_path.read_text().splitlines():
        pr = json.loads(line)
        if not pr["test_files"]:
            continue  # need a verifier
        touched = crates_touched(pr["changed_files"])
        single = len(touched) == 1
        crate = next(iter(touched)) if touched else None
        verify = f"cargo nextest run -p {crate}" if single and crate else "cargo nextest run"
        yield RLTask(
            task_id=f"hs-pr-{pr['number']}",
            pr_number=pr["number"],
            parent_commit=parent_of(repo, pr["merge_commit"]),
            issue_text=(pr["title"] + "\n\n" + strip_pr_template(pr["body"])),
            gold_files=pr["changed_files"],
            test_files=pr["test_files"],
            crates=sorted(touched),
            single_crate=single,
            verify_cmd=verify,
            verify_mode=verify_mode(touched),
        )


def build(cfg: dict) -> list[RLTask]:
    """In-memory build (tests/ad-hoc). Production path: build_and_save (incremental)."""
    return list(iter_tasks(cfg))


def save(tasks: list[RLTask], path: str = "data/datasets/rl_tasks.jsonl") -> int:
    out = config.ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for t in tasks:
            f.write(json.dumps(asdict(t)) + "\n")
    return len(tasks)


def build_and_save(cfg: dict, path: str = "data/datasets/rl_tasks.jsonl", batch: int = 10) -> tuple[int, int]:
    """Stream tasks to `path`, flushing every `batch` — crash-safe + visible progress."""
    out = config.ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    total = single_n = 0
    buf: list[RLTask] = []
    with out.open("w") as f:
        for t in iter_tasks(cfg):
            buf.append(t); total += 1; single_n += int(t.single_crate)
            if len(buf) >= batch:
                for x in buf:
                    f.write(json.dumps(asdict(x)) + "\n")
                f.flush(); buf = []
                print(f"  built {total} RL tasks ({single_n} single-crate) -> {path}", flush=True)
        for x in buf:
            f.write(json.dumps(asdict(x)) + "\n")
    print(f"RL tasks: {total} total; {single_n} single-crate (cheap-verify candidates)", flush=True)
    return total, single_n


if __name__ == "__main__":
    build_and_save(config.load("data"))
