"""Per-rollout human-readable insight log -> runs/insight.txt.

For every rollout we append a short, readable block: outcome, reward breakdown, how far
it got (fmt -> clippy -> check -> tests), how many turns, which tests failed, the last
error it hit, and a heuristic "lesson". This is the qualitative companion to the
structured metrics — it's what you actually read to figure out WHY rollouts fail and
how to improve the harness / curriculum / reward (the user's "learning per rollout").
"""
from __future__ import annotations

import time
from pathlib import Path

from src import config

INSIGHT_FILE = config.RUNS / "insight.txt"


def _flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
    """Flatten a nested config dict into [(dotted_key, value)] for a readable dump."""
    out: list[tuple[str, str]] = []
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.extend(_flatten(v, f"{key}."))
        else:
            out.append((key, str(v)))
    return out


def append_hyperparams(cycle: int) -> None:
    """Write a full hyperparameter snapshot for this cycle so insights are self-documenting.

    Dumps ALL knobs from the cpt/sft/rl configs (model, method, LoRA, optimizer, LR,
    algorithm/DAPO settings, reward weights, rollout, curriculum) — so when you read a
    rollout's lesson you know exactly which hyperparams produced it.
    """
    lines = [f"########## cycle {cycle} hyperparameters ({time.strftime('%Y-%m-%d %H:%M:%S')}) ##########"]
    for name in ("cpt", "sft", "rl"):
        try:
            cfg = config.load(name)
        except FileNotFoundError:
            continue
        lines.append(f"--- [{name}] ---")
        lines += [f"  {k} = {v}" for k, v in _flatten(cfg)]
    block = "\n".join(lines) + "\n\n"
    INSIGHT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with INSIGHT_FILE.open("a") as f:
        f.write(block)


def _last_error(messages: list[dict]) -> str:
    """Most recent tool output that looks like an error/failure (for the lesson)."""
    for m in reversed(messages):
        if m.get("role") != "tool":
            continue
        c = m.get("content", "") or ""
        if any(tok in c for tok in ("ERROR", "error[", "error:", "FAILED", "panicked")):
            return c.strip().splitlines()[0][:300] if c.strip() else ""
    return ""


def _lesson(bd: dict, failed_tests: list[str], last_err: str, turns: int) -> str:
    """Heuristic, actionable takeaway from the breakdown."""
    if bd.get("tampered"):
        return "Tampered with protected files (tests/manifest) — reward voided. Harden sandbox / penalize."
    if bd.get("tests", 0.0) >= 1.0:
        eff = "efficient" if turns <= 10 else f"slow ({turns} turns) — consider step penalty"
        return f"SOLVED cleanly; {eff}."
    if bd.get("compile", 0.0) >= 1.0:
        ft = f" failing: {failed_tests[:3]}" if failed_tests else ""
        return f"Compiled but tests failed — likely logic/field-mapping bug.{ft}"
    if last_err.startswith(("error[", "error:")) or "error[" in last_err:
        return f"Did not compile — type/borrow/trait error. First error: {last_err[:160]}"
    return f"No test pass; reached fmt={bd.get('fmt_clippy', 0):.1f} compile={bd.get('compile', 0):.1f}."


def append_insight(rollout: dict, cycle: int) -> None:
    bd = rollout.get("breakdown") or {}
    msgs = rollout.get("messages", [])
    turns = sum(1 for m in msgs if m.get("role") == "assistant")
    failed = bd.get("failed_tests", []) if isinstance(bd, dict) else []
    last_err = _last_error(msgs)
    solved = bd.get("tests", 0.0) >= 1.0
    tiers = (
        f"fmt/clippy={bd.get('fmt_clippy', 0):.1f} compile={bd.get('compile', 0):.0f} "
        f"dense={bd.get('dense', 0):.2f} tests={bd.get('tests', 0):.0f}"
    )
    block = (
        f"=== {time.strftime('%Y-%m-%d %H:%M:%S')} | cycle {cycle} | "
        f"{rollout.get('task_id', '?')} | model={rollout.get('model', '?')} ===\n"
        f"outcome: {'SOLVED' if solved else 'FAILED'} | reward={rollout.get('reward', 0):.2f} "
        f"| turns={turns} | mode={bd.get('mode', 'execution')}\n"
        f"tiers: {tiers}\n"
        f"lesson: {_lesson(bd, failed, last_err, turns)}\n\n"
    )
    INSIGHT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with INSIGHT_FILE.open("a") as f:
        f.write(block)


def append_many(rollouts: list[dict], cycle: int) -> int:
    for r in rollouts:
        append_insight(r, cycle)
    return len(rollouts)
