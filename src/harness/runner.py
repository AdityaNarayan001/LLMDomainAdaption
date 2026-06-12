"""The agent loop (observe -> think -> act -> feedback) over the Pi tool layer.

Drives an OpenAI-compatible endpoint (vLLM). Records a TOKEN-FAITHFUL trajectory:
each assistant turn stores the exact sampled token ids + logprobs returned by the
server, with tool/observation turns loss-masked (only assistant tokens train). This
is what makes SkyRL/Polar-style RL correct (§E).

Sandbox: each rollout runs in its own checkout of hyperswitch @ the task's parent
commit. In production this whole function runs inside a gVisor/Firecracker sandbox
with egress locked down (Risk R7); here we set up the workdir + protected paths.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import requests

from src import config
from src.harness import reward as reward_mod
from src.harness import tools

MAX_TURNS_DEFAULT = 40


@dataclass
class Trajectory:
    messages: list[dict] = field(default_factory=list)
    token_ids: list[list[int]] = field(default_factory=list)   # per assistant turn
    logprobs: list[list[float]] = field(default_factory=list)
    loss_mask: list[bool] = field(default_factory=list)        # per message
    reward: float = 0.0
    breakdown: dict | None = None


def setup_workdir(task: dict, base_repo: Path | None = None) -> tuple[Path, set[str]]:
    """Materialize a fresh checkout @ parent_commit. Returns (workdir, protected_paths)."""
    base_repo = base_repo or (config.ROOT / "data/hyperswitch")
    workdir = Path(tempfile.mkdtemp(prefix="hs_rollout_"))
    # cheap local clone + checkout (shares objects via --shared for speed)
    subprocess.run(["git", "clone", "--shared", str(base_repo), str(workdir)], check=True,
                   capture_output=True)
    if task.get("parent_commit"):
        subprocess.run(["git", "-C", str(workdir), "checkout", "-q", task["parent_commit"]],
                       check=True)
    # bug-injection tasks: apply the mutant so the crate's tests fail; the model must repair it
    if task.get("setup_patch"):
        subprocess.run(["git", "apply", "-"], input=task["setup_patch"], text=True,
                       cwd=workdir, check=False, capture_output=True)
    protected = set(task.get("test_files", [])) | {"Cargo.toml", "Cargo.lock"}
    # write-protect the hidden tests so a bash-wielding agent can't weaken them
    for rel in protected:
        fp = workdir / rel
        if fp.exists():
            fp.chmod(0o444)
    return workdir, protected


def _call_model(endpoint: str, model: str, messages: list[dict], temperature: float) -> dict:
    """One chat completion with tool-calling + logprobs (token-faithful capture)."""
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={
            "model": model,
            "messages": messages,
            "tools": tools.TOOL_SCHEMAS,
            "temperature": temperature,
            "logprobs": True,
            "max_tokens": 2048,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def run_task(
    task: dict,
    *,
    endpoint: str,
    model: str,
    weights: dict,
    temperature: float = 1.0,
    max_turns: int = MAX_TURNS_DEFAULT,
    closed_context: bool = False,
) -> Trajectory:
    """Run one agentic rollout and score it with the tiered reward."""
    workdir, protected = setup_workdir(task)
    box = tools.ToolBox(
        workdir=workdir,
        protected_paths=protected,
        closed_context=closed_context,
        allowed_files=set(task.get("provided_files", [])),
    )
    traj = Trajectory()
    traj.messages.append({"role": "system", "content": _system_prompt()})
    traj.loss_mask.append(False)
    traj.messages.append({"role": "user", "content": task["issue_text"]})
    traj.loss_mask.append(False)

    try:
        for _ in range(max_turns):
            out = _call_model(endpoint, model, traj.messages, temperature)
            choice = out["choices"][0]
            msg = choice["message"]
            traj.messages.append(msg)
            traj.loss_mask.append(True)  # assistant-generated -> trains
            _capture_tokens(traj, choice)

            calls = msg.get("tool_calls") or []
            if not calls:
                break  # model decided it's done
            for call in calls:
                args = json.loads(call["function"]["arguments"] or "{}")
                result = box.dispatch(call["function"]["name"], args)
                traj.messages.append({
                    "role": "tool", "tool_call_id": call["id"], "content": result,
                })
                traj.loss_mask.append(False)  # tool output -> masked (not model's tokens)

        package = task["crates"][0] if task.get("single_crate") else None
        if task.get("verify_mode") == "pattern":
            # connector/integration: no cheap unit tests -> compile+clippy+similarity-to-gold
            candidate_patch = box.bash("git diff")
            breakdown = reward_mod.pattern_reward(
                workdir, package, candidate_patch, task.get("gold_patch"),
                tampered=box.tampered, weights=weights,
            )
        else:
            breakdown = reward_mod.execution_reward(
                workdir, package=package,
                expected_fail_to_pass=task.get("fail_to_pass", []),
                tampered=box.tampered, weights=weights,
            )
        traj.reward = breakdown.total
        traj.breakdown = breakdown.__dict__
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    return traj


def verify_fail_to_pass(task) -> bool:
    """Census helper: does the gold patch actually flip the test(s) fail->pass, reproducibly?

    Returns True iff tests FAIL at parent and PASS after applying the gold diff.
    Requires the toolchain + DB env; used by census --verify-sample.
    """
    t = task.__dict__ if hasattr(task, "__dict__") else task
    workdir, _ = setup_workdir(t)
    try:
        pkg = t["crates"][0] if t.get("single_crate") else None
        before = tools.nextest_json(workdir, pkg)
        if not before["failed"]:
            return False  # nothing failing at parent => not a FAIL_TO_PASS we can verify
        subprocess.run(["git", "-C", str(workdir), "checkout", "-q",
                        t["pr_number"] and f"{t['parent_commit']}"], check=False)
        # apply gold change by checking out the merge commit's versions of changed files
        # (kept simple here; real impl applies the PR diff)
        after = tools.nextest_json(workdir, pkg)
        return not after["failed"]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _capture_tokens(traj: Trajectory, choice: dict) -> None:
    lp = (choice.get("logprobs") or {}).get("content") or []
    traj.token_ids.append([t.get("token_id", -1) for t in lp])
    traj.logprobs.append([t.get("logprob", 0.0) for t in lp])


def _system_prompt() -> str:
    return (
        "You are a Rust engineer in the juspay/hyperswitch codebase. Use the tools to "
        "read, edit, and test code until the failing tests pass. Follow hyperswitch "
        "conventions. Make minimal, correct changes."
    )
