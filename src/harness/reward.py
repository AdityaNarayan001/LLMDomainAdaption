"""Tiered, verifiable reward (decision #8 / §D). Cheap -> expensive, short-circuiting.

    R = 1.0*tests + 0.3*dense + 0.1*compile + 0.05*fmt_clippy

All signals are execution-grounded (resist reward hacking). Tampering with protected
paths (hidden tests / Cargo manifest) => reward 0 (Risk R11). For tasks too costly to
build, fall back to execution-free patch-similarity to the gold diff (Risk R1/R6).
"""
from __future__ import annotations

import difflib
import subprocess
from dataclasses import dataclass
from pathlib import Path

from src.harness import tools


@dataclass
class RewardBreakdown:
    fmt_clippy: float
    compile: float
    dense: float
    tests: float
    total: float
    tampered: bool
    mode: str  # "execution" | "execution_free"


def _ok(cmd: list[str], workdir: Path, timeout: int) -> bool:
    try:
        return subprocess.run(
            cmd, cwd=workdir, capture_output=True, text=True, timeout=timeout
        ).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def execution_reward(
    workdir: Path,
    package: str | None,
    expected_fail_to_pass: list[str],
    tampered: bool,
    weights: dict,
    build_budget_s: int = 240,
) -> RewardBreakdown:
    if tampered:
        return RewardBreakdown(0, 0, 0, 0, 0.0, True, "execution")

    fmt = 1.0 if _ok(["cargo", "+nightly", "fmt", "--check"], workdir, 60) else 0.0
    clippy = 1.0 if _ok(["cargo", "clippy", "--quiet"], workdir, build_budget_s) else 0.0
    fmt_clippy = (fmt + clippy) / 2

    compiles = _ok(
        ["cargo", "check"] + (["-p", package] if package else []), workdir, build_budget_s
    )
    compile_r = 1.0 if compiles else 0.0

    dense = tests = 0.0
    if compiles:  # only pay for tests if it compiles
        res = tools.nextest_json(workdir, package)
        total_expected = set(expected_fail_to_pass)
        passed = set(res["passed"])
        if total_expected:
            dense = len(total_expected & passed) / len(total_expected)
            tests = 1.0 if total_expected.issubset(passed) and not res["failed"] else 0.0

    total = (
        weights["tests"] * tests
        + weights["dense"] * dense
        + weights["compile"] * compile_r
        + weights["fmt_clippy"] * fmt_clippy
    )
    return RewardBreakdown(fmt_clippy, compile_r, dense, tests, total, False, "execution")


def execution_free_reward(candidate_patch: str, gold_patch: str) -> RewardBreakdown:
    """Cold-start / unbuildable: similarity of generated diff to the gold diff (SWE-RL)."""
    ratio = difflib.SequenceMatcher(None, candidate_patch, gold_patch).ratio()
    return RewardBreakdown(0, 0, ratio, 0, ratio, False, "execution_free")
