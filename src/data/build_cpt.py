"""Build the CPT corpus: dependency-ordered, document-masked packing + FIM + dedup.

Phase 1 (Foundation): source + docs, this module's main path (fully implemented).
Phase 2 (Evolution): commit diffs   -> build_phase2 (git-log driven, implemented).
Phase 3 (PR Mastery): PRs/issues     -> build_phase3 (uses ingest_github output).

Output: data/datasets/cpt_phase{1,2,3}.jsonl, each line {"training_content": str, "meta": {...}}.

Token estimates use a 4-char/token heuristic to stay dependency-light; the real
tokenizer is applied in the trainer. FIM uses sentinel markers; the trainer maps
them to the model's actual FIM special tokens.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import dataclass

from datasketch import MinHash, MinHashLSH

from src import config
from src.data import ast_rust, ingest_repos

FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE = "<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>"


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _minhash(text: str, num_perm: int = 128) -> MinHash:
    m = MinHash(num_perm=num_perm)
    # 5-gram char shingles
    for i in range(0, max(1, len(text) - 5), 5):
        m.update(text[i : i + 5].encode("utf-8", "ignore"))
    return m


def dedup(files: list[tuple[str, str]], jaccard: float) -> list[tuple[str, str]]:
    """Near-dedup (path, content) pairs via MinHash/LSH. Keeps first occurrence."""
    lsh = MinHashLSH(threshold=jaccard, num_perm=128)
    kept: list[tuple[str, str]] = []
    for i, (path, content) in enumerate(files):
        mh = _minhash(content)
        if lsh.query(mh):
            continue  # near-duplicate of something already kept
        lsh.insert(str(i), mh)
        kept.append((path, content))
    return kept


def _format_fim(prefix: str, middle: str, suffix: str, mode: str) -> str:
    if mode == "psm":
        return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}"
    return f"{FIM_SUFFIX}{suffix}{FIM_PREFIX}{prefix}{FIM_MIDDLE}{middle}"


def apply_fim(content: str, mode: str, structure_aware: bool = True) -> str:
    """FIM. mode in {psm, spm}. If tree-sitter is available, mask a COMPLETE item
    (function/impl) as the middle (structure-aware); otherwise split into thirds."""
    if structure_aware and content.count("fn ") and ast_rust.available():
        span = ast_rust.span_for_fim(content)
        if span:
            raw = content.encode("utf-8")
            s, e = span
            prefix = raw[:s].decode("utf-8", "ignore")
            middle = raw[s:e].decode("utf-8", "ignore")
            suffix = raw[e:].decode("utf-8", "ignore")
            return _format_fim(prefix, middle, suffix, mode)
    n = len(content)
    a, b = n // 3, 2 * n // 3
    return _format_fim(content[:a], content[a:b], content[b:], mode)


@dataclass
class Packed:
    training_content: str
    meta: dict


def build_phase1(cfg: dict):
    """Foundation: dependency-ordered packing with file markers, document-masked, FIM ~rate.
    Yields one Packed per (deduped) file so writing can stream."""
    repo = config.ROOT / cfg["source"]["raw_repo"]
    files = ingest_repos.iter_source_files(cfg, repo)
    pairs: list[tuple[str, str]] = []
    for p in files:
        rel = str(p.relative_to(repo))
        try:
            pairs.append((rel, p.read_text(encoding="utf-8", errors="ignore")))
        except OSError:
            continue
    pairs = dedup(pairs, cfg["cpt"]["dedup_jaccard"])  # dedup needs the full set; output streams

    fim_rate = cfg["cpt"]["fim_rate"]
    marker = cfg["cpt"]["file_marker"]
    # Each packed sample = one file (document-masked => no cross-file attention bleed).
    for idx, (rel, content) in enumerate(pairs):
        body = marker.format(path=rel) + content
        # deterministic FIM assignment (no RNG -> reproducible): alternate PSM/SPM
        if rel.endswith(".rs") and (idx % 100) < int(fim_rate * 100):
            body = apply_fim(body, "psm" if idx % 2 == 0 else "spm")
        yield Packed(training_content=body, meta={"path": rel, "phase": "foundation"})


def build_phase2(cfg: dict, max_commits: int = 5000):
    """Evolution: conventional-commit message + diff as intent->change pairs (streamed)."""
    repo = config.ROOT / cfg["source"]["raw_repo"]
    log = subprocess.run(
        ["git", "-C", str(repo), "log", f"-n{max_commits}", "--format=%H%x00%s%x00%b"],
        capture_output=True, text=True, check=True,
    ).stdout
    for line in log.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\x00")
        if len(parts) < 2:
            continue
        sha, subject = parts[0], parts[1]
        diff = subprocess.run(
            ["git", "-C", str(repo), "show", "--format=", "--unified=3", sha],
            capture_output=True, text=True, check=False,
        ).stdout[:20000]  # cap diff size
        content = f"# Commit intent\n{subject}\n\n# Change\n{diff}"
        yield Packed(training_content=content, meta={"sha": sha, "phase": "evolution"})


_TEMPLATE_CHECKLIST = re.compile(r"^\s*-\s*\[[ xX]\]")          # "- [ ] Bugfix" template items
_HTML_COMMENT = re.compile(r"<!--.*?-->", re.S)                  # "<!-- Put an x ... -->"


def strip_pr_template(body: str) -> str:
    """Remove hyperswitch's PR-template boilerplate (HTML comments + checklist items + the
    empty '## Type of Change'/'## Checklist' scaffolding) so only real prose survives.
    Garbage-in-garbage-out: the checklists teach nothing and dilute the signal."""
    body = _HTML_COMMENT.sub("", body or "")
    kept = [ln for ln in body.splitlines() if not _TEMPLATE_CHECKLIST.match(ln)]
    text = "\n".join(kept)
    text = re.sub(r"\n{3,}", "\n\n", text)                       # collapse runs of blank lines
    # drop now-empty boilerplate headers
    text = re.sub(r"(?m)^#+\s*(Type of Change|Checklist|Motivation and Context)\s*$\n?", "", text)
    return text.strip()


def build_phase3(cfg: dict):
    """PR Mastery: intent (issue/PR body) -> change -> review discussion (streamed)."""
    prs_path = config.ROOT / "data/datasets/github_prs.jsonl"
    if not prs_path.exists():
        return  # run ingest_github first
    for line in prs_path.read_text().splitlines():
        pr = json.loads(line)
        body = strip_pr_template(pr["body"])
        if len(body) < 40:
            continue  # template-only PR with no real description — skip (no signal)
        content = (
            f"# PR #{pr['number']}: {pr['title']}\n\n{body}\n\n"
            f"# Files changed\n" + "\n".join(pr["changed_files"])
        )
        yield Packed(training_content=content, meta={"pr": pr["number"], "phase": "pr_mastery"})


def _write(samples, name: str, batch: int = 50) -> int:
    """Stream samples (list OR generator) to jsonl, flushing every `batch` — crash-safe + visible."""
    out = config.ROOT / "data/datasets" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    buf: list[Packed] = []
    with out.open("w") as f:
        for s in samples:
            buf.append(s); n += 1
            if len(buf) >= batch:
                for x in buf:
                    f.write(json.dumps({"training_content": x.training_content, "meta": x.meta}) + "\n")
                f.flush(); buf = []
                print(f"  wrote {n} -> {name}", flush=True)
        for x in buf:
            f.write(json.dumps({"training_content": x.training_content, "meta": x.meta}) + "\n")
    return n


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "ignore")).hexdigest()[:16]


if __name__ == "__main__":
    cfg = config.load("data")
    print(f"phase1 foundation: {_write(build_phase1(cfg), 'cpt_phase1.jsonl')} samples")
    print(f"phase2 evolution:  {_write(build_phase2(cfg), 'cpt_phase2.jsonl')} samples")
    print(f"phase3 pr_mastery: {_write(build_phase3(cfg), 'cpt_phase3.jsonl')} samples")
