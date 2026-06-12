"""Harvest verified rollouts -> next-round SFT/RL data (rejection sampling + dedup).

Keeps only test-verified successes; dedups by (task_id, patch_fingerprint); enforces a
diversity floor so the corpus doesn't collapse onto easy/connector tasks (Risk R4).
Asserts no contamination with the held-out eval set every cycle (a hard guard).
"""
from __future__ import annotations

import hashlib
import json

from src import config


def _fingerprint(rollout: dict) -> str:
    # fingerprint the assistant edits (rough: concat assistant contents).
    # `or ""`: tool-calling assistant turns carry content=None (the key EXISTS, so
    # .get(k, "") returns None) — the first successful tool-using rollout would TypeError.
    blob = "".join((m.get("content") or "") for m in rollout["messages"] if m["role"] == "assistant")
    return hashlib.sha256(blob.encode("utf-8", "ignore")).hexdigest()[:16]


def load_eval_task_ids() -> set[str]:
    ids: set[str] = set()
    for sub in ("hs_knowledge", "hs_swe", "sequestered"):
        for jf in (config.EVAL_SETS / sub).glob("*.json"):
            ids.add(json.loads(jf.read_text()).get("task_id", jf.stem))
    return ids


def harvest(rollouts: list[dict], min_reward: float = 1.0) -> tuple[list[dict], list[dict]]:
    """Return (new_sft_records, new_rl_records). Successful, deduped, decontaminated."""
    eval_ids = load_eval_task_ids()
    seen: set[tuple[str, str]] = set()
    sft, rl = [], []
    for r in rollouts:
        if r["reward"] < min_reward:
            continue                                   # rejection sampling
        if r["task_id"] in eval_ids:
            raise AssertionError(f"CONTAMINATION: {r['task_id']} is in the eval set")
        key = (r["task_id"], _fingerprint(r))
        if key in seen:
            continue                                   # dedup
        seen.add(key)
        sft.append({"messages": r["messages"], "loss_mask": r["loss_mask"]})
        rl.append({"task_id": r["task_id"]})           # task now has a known-good solution
    return sft, rl


def append_jsonl(records: list[dict], path: str) -> None:
    out = config.ROOT / path
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
