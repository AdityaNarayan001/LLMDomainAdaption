"""veRL custom reward — apply the model's patch and run our verifiable cargo reward.

veRL calls `compute_score(data_source, solution_str, ground_truth, extra_info)` per response.
We apply `solution_str` (a unified diff) to a fresh hyperswitch checkout @ the task's parent
commit, then score with src.harness.reward (execution tests for cheap-verify crates, else
compile+clippy+patch-similarity). All execution-grounded — resists reward hacking.

NOTE (validation): confirm veRL 0.8's exact reward-fn signature/return on a live run.
"""
from __future__ import annotations

import re
import subprocess

from src import config
from src.harness import reward as reward_mod
from src.harness import runner

_WEIGHTS = None
_DIFF = re.compile(r"```(?:diff|patch)?\n(.*?)```", re.S)


def _weights() -> dict:
    global _WEIGHTS
    if _WEIGHTS is None:
        _WEIGHTS = config.load("rl")["reward"]["weights"]
    return _WEIGHTS


def _extract_patch(solution_str: str) -> str:
    m = _DIFF.search(solution_str)
    return (m.group(1) if m else solution_str).strip()


def compute_score(data_source=None, solution_str: str = "", ground_truth=None,
                  extra_info: dict | None = None) -> float:
    """Return the tiered verifiable reward for the generated patch on its task."""
    task = extra_info or {}
    if not task.get("parent_commit"):
        return 0.0
    workdir, _ = runner.setup_workdir(task)
    try:
        patch = _extract_patch(solution_str)
        applied = subprocess.run(["git", "apply", "-"], input=patch, text=True,
                                 cwd=workdir, capture_output=True).returncode == 0
        if not applied:
            return 0.0                       # doesn't even apply
        pkg = task["crates"][0] if task.get("single_crate") else None
        if task.get("verify_cmd") and task.get("setup_patch"):
            # bug-injection: setup_workdir injected the mutant; the model's patch must restore a
            # green `cargo test -p <crate>` (cargo on PATH from the user's rustup).
            import os
            env = {**os.environ,
                   "PATH": os.path.expanduser("~/.cargo/bin") + os.pathsep + os.environ.get("PATH", "")}
            r = subprocess.run(task["verify_cmd"].split(), cwd=workdir, capture_output=True,
                               text=True, env=env, timeout=900)
            return 1.0 if r.returncode == 0 else 0.0
        if task.get("verify_mode") == "pattern":
            bd = reward_mod.pattern_reward(workdir, pkg, patch, task.get("gold_patch"),
                                           tampered=False, weights=_weights())
        else:
            bd = reward_mod.execution_reward(workdir, pkg, task.get("fail_to_pass", []),
                                             tampered=False, weights=_weights())
        return float(bd.total)
    finally:
        import shutil
        shutil.rmtree(workdir, ignore_errors=True)
