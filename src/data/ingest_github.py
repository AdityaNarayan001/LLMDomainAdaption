"""Pull closed issues + their resolving PRs (with changed files/tests) via the ``gh`` CLI.

Feeds two consumers: Stage-1 RL task mining (build_rl) and Stage-0b eval-set curation.
Uses ``gh api`` so it inherits the user's auth; falls back to anonymous REST if needed.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass

from src import config


@dataclass
class PRRecord:
    number: int
    title: str
    body: str
    merge_commit: str | None
    parent_commit: str | None
    changed_files: list[str]
    test_files: list[str]
    linked_issue: int | None


def _gh_json(args: list[str]) -> object:
    """Run ``gh <args>`` and parse JSON stdout. Raises if gh is missing/unauthed."""
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def fetch_merged_prs(repo: str, limit: int = 1000) -> list[PRRecord]:
    """List merged PRs with their changed files. Test files = paths under tests/ or *_test.rs."""
    raw = _gh_json(
        [
            "pr", "list", "--repo", repo, "--state", "merged", "--limit", str(limit),
            "--json", "number,title,body,mergeCommit,files,closingIssuesReferences",
        ]
    )
    records: list[PRRecord] = []
    for pr in raw:  # type: ignore[union-attr]
        files = [f["path"] for f in (pr.get("files") or [])]
        tests = [
            f for f in files
            if "/tests/" in f or f.endswith("_test.rs") or f.startswith("tests/")
        ]
        issues = pr.get("closingIssuesReferences") or []
        records.append(
            PRRecord(
                number=pr["number"],
                title=pr.get("title", ""),
                body=pr.get("body") or "",
                merge_commit=(pr.get("mergeCommit") or {}).get("oid"),
                parent_commit=None,  # resolved lazily via git when building tasks
                changed_files=files,
                test_files=tests,
                linked_issue=(issues[0]["number"] if issues else None),
            )
        )
    return records


def save(records: list[PRRecord], path: str = "data/datasets/github_prs.jsonl") -> int:
    out = config.ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps(asdict(r)) + "\n")
    return len(records)


if __name__ == "__main__":
    cfg = config.load("data")
    prs = fetch_merged_prs(cfg["source"]["repo"], limit=cfg["github"]["per_page"] * 10)
    n = save(prs)
    with_tests = sum(1 for p in prs if p.test_files)
    print(f"fetched {n} merged PRs; {with_tests} changed a test file")
