"""veRL custom reward — apply the model's patch and run our verifiable cargo reward.

veRL calls `compute_score(data_source, solution_str, ground_truth, extra_info)` per response.
We apply `solution_str` (a unified diff) to a fresh hyperswitch checkout @ the task's parent
commit, then score with src.harness.reward (execution tests for cheap-verify crates, else
compile+clippy+patch-similarity). All execution-grounded — resists reward hacking.

HARDENING: this function must NEVER raise — veRL's reward manager calls it with no
try/except, so one slow cargo build (TimeoutExpired) or one bad task (clone/checkout/
setup_patch failure) would otherwise kill the entire training run. Any failure => 0.0.
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess

from src import config
from src.harness import reward as reward_mod
from src.harness import runner

_WEIGHTS = None
# Closing fence anchored to start-of-line (\n```) so the non-greedy body match does NOT truncate at
# an INNER ``` — Rust diffs routinely carry doc-comment code fences ("/// ```rust"), and a naive
# r"...(.*?)```" stops at the first one, yielding a corrupt (truncated) patch that never applies.
# Two passes: prefer an EXPLICIT ```diff/```patch block (so an explanatory ```rust snippet before
# the diff can't shadow it), fall back to any fenced block, then the raw text.
_DIFF_EXPLICIT = re.compile(r"```(?:diff|patch)\n(.*?)\n```", re.S)
_DIFF_ANY = re.compile(r"```[a-zA-Z]*\n(.*?)\n```", re.S)


def _weights() -> dict:
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = config.load("rl")["reward"]["weights"]
    return _WEIGHTS


def _extract_patch(solution_str: str) -> str:
    m = _DIFF_EXPLICIT.search(solution_str) or _DIFF_ANY.search(solution_str)
    patch = m.group(1) if m else solution_str
    # Preserve the body verbatim (leading "diff --git"/"--- a/" lines matter); only normalise to a
    # single trailing newline. `git apply` rejects a patch whose final line lacks one ("corrupt patch
    # at line N"), so without this every rollout patch silently scores 0.0 regardless of correctness.
    return patch.rstrip("\n") + "\n" if patch.strip() else ""


def _cargo_env(task: dict) -> dict:
    """cargo on PATH + a SHARED per-crate target dir so each scored rollout doesn't rebuild the
    crate graph from scratch in its throwaway clone (cold builds blow the verify timeout)."""
    crate = (task.get("crates") or ["_"])[0]
    cache = os.path.expanduser(f"~/.cache/hs_target/{crate}")
    os.makedirs(cache, exist_ok=True)
    return {**os.environ,
            "PATH": os.path.expanduser("~/.cargo/bin") + os.pathsep + os.environ.get("PATH", ""),
            "CARGO_TARGET_DIR": cache}


def compute_score(data_source=None, solution_str: str = "", ground_truth=None,
                  extra_info: dict | None = None) -> float:
    """Return the tiered verifiable reward for the generated patch on its task. Never raises."""
    task = extra_info or {}
    if not task.get("parent_commit"):
        return 0.0
    try:
        workdir, _ = runner.setup_workdir(task)
    except Exception as exc:  # invalid task (clone/checkout/setup_patch) — zero the whole group
        print(f"[reward] setup failed for {task.get('task_id', '?')}: {exc}", flush=True)
        return 0.0
    try:
        patch = _extract_patch(solution_str)
        applied = subprocess.run(["git", "apply", "-"], input=patch, text=True,
                                 cwd=workdir, capture_output=True).returncode == 0
        if not applied:
            return 0.0                       # doesn't even apply
        pkg = task["crates"][0] if task.get("single_crate") else None
        if task.get("verify_cmd") and task.get("setup_patch"):
            # bug-injection: setup_workdir injected the mutant; the model's patch must restore a
            # green `cargo test -p <crate>`.
            r = subprocess.run(shlex.split(task["verify_cmd"]), cwd=workdir, capture_output=True,
                               text=True, env=_cargo_env(task), timeout=900)
            return 1.0 if r.returncode == 0 else 0.0
        if task.get("verify_mode") == "pattern":
            bd = reward_mod.pattern_reward(workdir, pkg, patch, task.get("gold_patch"),
                                           tampered=False, weights=_weights())
        else:
            bd = reward_mod.execution_reward(workdir, pkg, task.get("fail_to_pass", []),
                                             tampered=False, weights=_weights())
        return float(bd.total)
    except subprocess.TimeoutExpired:
        print(f"[reward] verify timed out for {task.get('task_id', '?')} — 0.0", flush=True)
        return 0.0
    except Exception as exc:  # any other failure: score 0, never kill the training run
        print(f"[reward] error for {task.get('task_id', '?')}: {exc} — 0.0", flush=True)
        return 0.0
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
