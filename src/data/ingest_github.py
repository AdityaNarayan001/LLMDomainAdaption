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


def _pr_record(pr: dict, headers: dict) -> "PRRecord":
    # PAGINATE /files: a PR touching >100 files would otherwise get a truncated list,
    # silently corrupting crates_touched/single_crate/verify_mode downstream. And an API
    # error (rate-limit 403 is likely at 1000 PRs) must be LOUD — a quietly-empty file list
    # makes build_rl drop the PR with zero trace.
    files: list[str] = []
    page = 1
    while True:
        fr = requests.get(pr["url"] + "/files", params={"per_page": 100, "page": page},
                          headers=headers, timeout=60)
        if not fr.ok:
            raise RuntimeError(f"GitHub /files failed for PR #{pr.get('number')}: "
                               f"{fr.status_code} {fr.text[:200]}")
        chunk = [f["filename"] for f in fr.json()]
        files += chunk
        if len(chunk) < 100:
            break
        page += 1
    tests = [f for f in files if "/tests/" in f or f.endswith("_test.rs") or f.startswith("tests/")]
    m = _CLOSES.search(pr.get("body") or "")
    return PRRecord(number=pr["number"], title=pr.get("title", ""), body=pr.get("body") or "",
                    merge_commit=pr.get("merge_commit_sha"), parent_commit=None,
                    changed_files=files, test_files=tests,
                    linked_issue=(int(m.group(1)) if m else None))


def mine_merged_prs(repo: str, limit: int = 1000,
                    out_path: str = "data/datasets/github_prs.jsonl", batch: int = 10) -> int:
    """Token-based REST mining that APPENDS to out_path in batches of `batch` as it goes —
    crash-safe + visible progress (no collect-all-then-write). Returns count written."""
    tok = _token()
    h = {"Accept": "application/vnd.github+json",
         **({"Authorization": f"Bearer {tok}"} if tok else {})}
    out = config.ROOT / out_path
    out.parent.mkdir(parents=True, exist_ok=True)
    written, buf, page = 0, [], 1
    with out.open("w") as f:
        while written + len(buf) < limit:
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
                buf.append(_pr_record(pr, h))
                if len(buf) >= batch:                       # flush a batch -> visible + crash-safe
                    for rec in buf:
                        f.write(json.dumps(asdict(rec)) + "\n")
                    f.flush()
                    written += len(buf); buf = []
                    print(f"  mined {written} merged PRs -> {out_path}", flush=True)
                if written + len(buf) >= limit:
                    break
            page += 1
        for rec in buf:                                     # final partial batch
            f.write(json.dumps(asdict(rec)) + "\n")
        written += len(buf)
    print(f"mined {written} merged PRs (batched x{batch}) -> {out_path}", flush=True)
    return written


def fetch_merged_prs_rest(repo: str, limit: int = 1000) -> list["PRRecord"]:
    """In-memory variant (ad-hoc/tests). Production path uses mine_merged_prs (incremental)."""
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
            if pr.get("merged_at"):
                out.append(_pr_record(pr, h))
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
    import os
    cfg = config.load("data")
    repo = cfg["source"]["repo"]
    limit = cfg["github"]["per_page"] * 10
    out = config.ROOT / "data/datasets/github_prs.jsonl"
    have = len(out.read_text().splitlines()) if out.exists() else 0
    if have >= limit and os.environ.get("FORCE_INGEST") != "1":   # idempotent: reuse on resume
        print(f"github_prs.jsonl already has {have} PRs (>= {limit}) — skipping mine "
              "(FORCE_INGEST=1 to re-mine).")
    elif _token():                                 # production path: incremental batched write
        mine_merged_prs(repo, limit=limit, batch=10)
    elif gh_ready():                               # gh CLI: bulk, then write
        save(fetch_merged_prs(repo, limit=limit))
    else:
        print("WARN: no GH_TOKEN and gh not authenticated — skipping PR mining "
              "(RL/PR-Mastery empty; census will NO-GO).")
        save([])
