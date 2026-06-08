"""Pull closed issues + their resolving PRs (with changed files/tests) via the ``gh`` CLI.

Feeds two consumers: Stage-1 RL task mining (build_rl) and Stage-0b eval-set curation.
Uses ``gh api`` so it inherits the user's auth; falls back to anonymous REST if needed.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from dataclasses import asdict, dataclass

import requests

from src import config

_CLOSES = re.compile(r"\b(?:close[sd]?|fix(?:e[sd])?|resolve[sd]?)\s+#(\d+)", re.I)


def _token() -> str | None:
    return os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN")


def fetch_merged_prs_rest(repo: str, limit: int = 1000) -> list["PRRecord"]:
    """Token-based REST mining (no gh CLI). 5000 req/hr authenticated. Paginates closed
    PRs, keeps merged ones, fetches changed files, links issues via body keywords."""
    tok = _token()
    h = {"Accept": "application/vnd.github+json",
         **({"Authorization": f"Bearer {tok}"} if tok else {})}
    out: list[PRRecord] = []
    page = 1
    while len(out) < limit:
        r = requests.get(f"https://api.github.com/repos/{repo}/pulls",
                         params={"state": "closed", "per_page": 100, "page": page},
                         headers=h, timeout=60)
        r.raise_for_status()
        prs = r.json()
        if not prs:
            break
        for pr in prs:
            if not pr.get("merged_at"):
                continue
            fr = requests.get(pr["url"] + "/files", params={"per_page": 100}, headers=h, timeout=60)
            files = [f["filename"] for f in fr.json()] if fr.ok else []
            tests = [f for f in files if "/tests/" in f or f.endswith("_test.rs") or f.startswith("tests/")]
            m = _CLOSES.search(pr.get("body") or "")
            out.append(PRRecord(
                number=pr["number"], title=pr.get("title", ""), body=pr.get("body") or "",
                merge_commit=pr.get("merge_commit_sha"), parent_commit=None,
                changed_files=files, test_files=tests,
                linked_issue=(int(m.group(1)) if m else None)))
            if len(out) >= limit:
                break
        page += 1
    return out


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


def gh_ready() -> bool:
    """True if gh is installed AND authenticated (or GH_TOKEN is set). Degrade gracefully."""
    import os
    import shutil
    if shutil.which("gh") is None:
        return False
    if os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"):
        return True
    return subprocess.run(["gh", "auth", "status"], capture_output=True).returncode == 0


def _gh_json(args: list[str]) -> object:
    """Run ``gh <args>`` and parse JSON stdout. Raises if gh is missing/unauthed."""
    out = subprocess.run(["gh", *args], capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def fetch_merged_prs(repo: str, limit: int = 1000) -> list[PRRecord]:
    """List merged PRs with their changed files. Test files = paths under tests/ or *_test.rs.

    Routing: GH_TOKEN -> REST (no gh CLI needed); else gh CLI if authed; else warn + [].
    Degrades gracefully so the pipeline continues — the M1 census then NO-GO's with a clear
    reason rather than the whole run crashing."""
    if _token():
        return fetch_merged_prs_rest(repo, limit)          # token path (the box uses this)
    if not gh_ready():
        print("WARN: no GH_TOKEN and gh not authenticated — skipping PR mining. "
              "RL/PR-Mastery data will be empty (export GH_TOKEN or run 'gh auth login').")
        return []
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
