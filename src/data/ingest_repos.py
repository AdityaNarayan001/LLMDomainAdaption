"""Clone hyperswitch and walk its source in crate-dependency (topological) order.

The dependency order matters for CPT repo-level packing: files that *define* types/
traits should appear in-context before files that *use* them. We derive the order
from the Cargo workspace dependency graph (path deps between member crates).

Runnable without a GPU. Uses plain ``git`` via subprocess and a tiny TOML reader.
"""
from __future__ import annotations

import fnmatch
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

try:  # py311+ has tomllib in stdlib
    import tomllib  # type: ignore
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore

from src import config


@dataclass
class Crate:
    name: str
    path: Path
    deps: set[str] = field(default_factory=set)  # intra-workspace deps only


def clone_or_update(cfg: dict) -> Path:
    """Clone the repo (shallow) or fetch if it already exists. Returns the checkout path."""
    dest = config.ROOT / cfg["source"]["raw_repo"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    if (dest / ".git").exists():
        subprocess.run(["git", "-C", str(dest), "fetch", "--all", "--tags"], check=True)
    else:
        subprocess.run(
            ["git", "clone", cfg["source"]["clone_url"], str(dest)], check=True
        )
    commit = cfg["source"].get("base_commit")
    if commit:
        subprocess.run(["git", "-C", str(dest), "checkout", commit], check=True)
    return dest


def discover_crates(repo: Path) -> dict[str, Crate]:
    """Find workspace member crates and their intra-workspace path/dependency edges."""
    crates: dict[str, Crate] = {}
    for cargo in (repo / "crates").glob("*/Cargo.toml"):
        data = tomllib.loads(cargo.read_text(encoding="utf-8", errors="ignore"))
        name = data.get("package", {}).get("name")
        if not name:
            continue
        crates[name] = Crate(name=name, path=cargo.parent)
    # second pass: edges (a crate depends on another workspace member)
    names = set(crates)
    for cargo in (repo / "crates").glob("*/Cargo.toml"):
        data = tomllib.loads(cargo.read_text(encoding="utf-8", errors="ignore"))
        name = data.get("package", {}).get("name")
        if name not in crates:
            continue
        for section in ("dependencies", "dev-dependencies", "build-dependencies"):
            for dep in (data.get(section) or {}):
                if dep in names and dep != name:
                    crates[name].deps.add(dep)
    return crates


def topo_order(crates: dict[str, Crate]) -> list[str]:
    """Kahn topological sort; ties broken by name for determinism. Cycles -> appended last."""
    indeg = {n: 0 for n in crates}
    for c in crates.values():
        for d in c.deps:
            indeg[c.name] += 1  # name depends on d => d should come first
    order: list[str] = []
    ready = sorted(n for n, d in indeg.items() if d == 0)
    seen: set[str] = set()
    while ready:
        n = ready.pop(0)
        order.append(n)
        seen.add(n)
        # decrement dependents
        for c in crates.values():
            if n in c.deps and c.name not in seen:
                indeg[c.name] -= 1
                if indeg[c.name] == 0:
                    ready.append(c.name)
        ready = sorted(set(ready))
    # any remaining (cycles) appended deterministically
    order += sorted(n for n in crates if n not in seen)
    return order


def iter_source_files(cfg: dict, repo: Path) -> list[Path]:
    """Yield source/doc files in dependency order, honoring include/exclude globs."""
    crates = discover_crates(repo)
    ordered = topo_order(crates)
    inc = cfg["cpt"]["include_globs"]
    exc = cfg["cpt"]["exclude_globs"]

    def matches(rel: str, globs: list[str]) -> bool:
        return any(fnmatch.fnmatch(rel, g) for g in globs)

    files: list[Path] = []
    # crate sources first, in topo order
    for name in ordered:
        for p in sorted(crates[name].path.rglob("*.rs")):
            rel = str(p.relative_to(repo))
            if matches(rel, inc) and not matches(rel, exc):
                files.append(p)
    # then non-crate docs/specs (order-independent)
    for p in sorted(repo.rglob("*")):
        if not p.is_file() or "crates/" in str(p.relative_to(repo)):
            continue
        rel = str(p.relative_to(repo))
        if matches(rel, inc) and not matches(rel, exc):
            files.append(p)
    return files


if __name__ == "__main__":
    cfg = config.load("data")
    repo = clone_or_update(cfg)
    crates = discover_crates(repo)
    order = topo_order(crates)
    print(f"crates: {len(crates)}  topo order (first 10): {order[:10]}")
    print(f"source files (dependency-ordered): {len(iter_source_files(cfg, repo))}")
