"""Build SFT data: grounded instructions + rejection-sampled agent trajectories.

Two sources (decision-locked):
  1. Instructions grounded in real crate snippets (OSS-Instruct / Evol-Instruct style),
     synthesized by a teacher served via vLLM. Quality >> quantity.
  2. Agent trajectories: run the teacher (cold-start) or current model through the Pi
     harness on RL tasks, keep ONLY successful (test-verified) trajectories.

This module owns the *formatting + filtering* logic (runnable now). The generation
calls (teacher rollouts) are delegated to src.harness.runner, which needs the serve
extra; here we provide the rejection-sampling + chat-template assembly.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from src import config

SYSTEM = (
    "You are a Rust engineer working in the juspay/hyperswitch codebase. "
    "Use the provided tools to read, edit, and test code. Follow hyperswitch "
    "conventions (conventional commits, error types, connector patterns)."
)


@dataclass
class Turn:
    role: str          # system | user | assistant | tool
    content: str
    loss: bool         # True only for assistant-generated tokens (assistant-only loss mask)


def to_chat(turns: list[Turn]) -> dict:
    """Serialize a trajectory into a chat record with a per-turn loss mask."""
    return {
        "messages": [{"role": t.role, "content": t.content} for t in turns],
        "loss_mask": [t.loss for t in turns],
    }


def rejection_sample(trajectories: list[dict], min_reward: float = 1.0) -> list[dict]:
    """Keep only verified-successful trajectories (reward >= threshold). ~70-90% rejected."""
    return [t for t in trajectories if t.get("reward", 0.0) >= min_reward]


def assemble_instruction(snippet: str, problem: str, solution: str) -> dict:
    """OSS-Instruct-style (problem, solution) grounded in a real snippet -> chat record."""
    turns = [
        Turn("system", SYSTEM, loss=False),
        Turn("user", f"{problem}\n\nRelevant code:\n```rust\n{snippet}\n```", loss=False),
        Turn("assistant", solution, loss=True),
    ]
    return to_chat(turns)


def build(cfg: dict) -> tuple[list[dict], list[dict]]:
    """Return (instruction_records, trajectory_records). Reads generated raw dumps if present."""
    raw_dir = config.ROOT / "data/datasets"
    instructions: list[dict] = []
    trajectories: list[dict] = []

    raw_instr = raw_dir / "sft_instructions_raw.jsonl"
    if raw_instr.exists():
        kept = skipped = 0
        for line in raw_instr.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1; continue
            # OSS-Instruct: the answer is the REAL code snippet from the repo — never the
            # teacher's invented "solution" field (that risks hallucinated training targets).
            sol = r.get("snippet") or r.get("solution")
            if not (str(r.get("snippet", "")).strip() and str(r.get("problem", "")).strip()
                    and str(sol).strip()):
                skipped += 1; continue
            instructions.append(assemble_instruction(r["snippet"], r["problem"], sol))
            kept += 1
        print(f"instructions: kept {kept}, skipped {skipped} malformed/empty teacher outputs")

    raw_traj = raw_dir / "sft_trajectories_raw.jsonl"
    if raw_traj.exists():
        all_traj = [json.loads(line) for line in raw_traj.read_text().splitlines()]
        kept = rejection_sample(all_traj)
        trajectories = [t["chat"] for t in kept]
        print(f"trajectories: kept {len(kept)}/{len(all_traj)} "
              f"({100 * (1 - len(kept) / max(1, len(all_traj))):.0f}% rejected)")
    return instructions, trajectories


def save(instructions: list[dict], trajectories: list[dict]) -> None:
    out = config.ROOT / "data/datasets"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "sft_instructions.jsonl").open("w") as f:
        for r in instructions:
            f.write(json.dumps(r) + "\n")
    with (out / "sft_trajectories.jsonl").open("w") as f:
        for r in trajectories:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    random.seed(0)
    cfg = config.load("data")
    instr, traj = build(cfg)
    save(instr, traj)
    print(f"SFT: {len(instr)} instructions, {len(traj)} successful trajectories")
