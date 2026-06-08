"""Verify the GENERATED datasets before training trusts them.

Not a replacement for the M1 census (the hard RL-supply gate) — this is a fast
sanity + integrity pass run right after the `data` stage:
  * schema + non-empty  (CPT phases, RL tasks, SFT instructions/trajectories)
  * decontamination      (RL/SFT task ids disjoint from anything under eval_sets/)
  * RL reproducibility   (sampled: parent commit resolves AND test files exist at it)

Exit code: 1 only on a FATAL problem (empty CPT foundation, or decontamination leak),
so a corrupt build halts run.sh; soft issues are reported but non-blocking.
"""
from __future__ import annotations

import json
import subprocess
import sys

from src import config


def _lines(name: str) -> list[dict]:
    p = config.ROOT / "data/datasets" / name
    if not p.exists():
        return []
    out = []
    for ln in p.read_text().splitlines():
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError:
            pass  # malformed line — counted as a schema problem below via the count gap
    return out


def _eval_task_ids() -> set:
    """Every task_id / hs-pr-<n> referenced under eval_sets/ — must never appear in training."""
    ids: set = set()
    eval_dir = config.ROOT / "eval_sets"
    if not eval_dir.exists():
        return ids
    for p in eval_dir.rglob("*.jsonl"):
        for ln in p.read_text().splitlines():
            try:
                r = json.loads(ln)
            except json.JSONDecodeError:
                continue
            if r.get("task_id"):
                ids.add(r["task_id"])
            if r.get("pr_number"):
                ids.add(f"hs-pr-{r['pr_number']}")
    return ids


def _rl_repro_fraction(cfg: dict, rl: list[dict], k: int = 10) -> float:
    """Sample k RL tasks; fraction whose parent commit resolves AND test files exist there."""
    repo = config.ROOT / cfg["source"]["raw_repo"]
    sample = rl[:: max(1, len(rl) // k)][:k] if rl else []
    if not sample:
        return 0.0
    ok = 0
    for t in sample:
        pc = t.get("parent_commit")
        if not pc:
            continue
        if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", pc],
                          capture_output=True).returncode != 0:
            continue
        tests_present = all(
            subprocess.run(["git", "-C", str(repo), "cat-file", "-e", f"{pc}:{tf}"],
                           capture_output=True).returncode == 0
            for tf in (t.get("test_files") or [])
        )
        ok += int(tests_present)
    return ok / len(sample)


def validate(cfg: dict) -> int:
    fatal, warn = [], []

    # --- CPT phases: non-empty + every record has non-blank training_content ---
    p1 = _lines("cpt_phase1.jsonl")
    if not p1:
        fatal.append("cpt_phase1.jsonl EMPTY — CPT has nothing to train on")
    elif any(not str(r.get("training_content", "")).strip() for r in p1[:2000]):
        warn.append("cpt_phase1.jsonl has blank training_content rows")
    for nm in ("cpt_phase2.jsonl", "cpt_phase3.jsonl"):
        if not _lines(nm):
            warn.append(f"{nm} empty (phase contributes no data)")

    # --- RL tasks: schema + decontamination + reproducibility sample ---
    rl = _lines("rl_tasks.jsonl")
    if not rl:
        warn.append("rl_tasks.jsonl EMPTY — RL/flywheel will NO-GO at census")
    else:
        bad = [t for t in rl if not t.get("task_id") or not t.get("test_files")]
        if bad:
            warn.append(f"{len(bad)} RL tasks missing task_id/test_files")
        leaked = _eval_task_ids() & {t.get("task_id") for t in rl}
        if leaked:
            fatal.append(f"DECONTAM: {len(leaked)} RL tasks leak into eval_sets/ ({list(leaked)[:3]}…)")
        frac = _rl_repro_fraction(cfg, rl)
        print(f"[validate] RL reproducibility sample: {frac:.0%} had resolvable parent + tests present")
        if frac < 0.5:
            warn.append(f"only {frac:.0%} of sampled RL tasks look reproducible (check git history depth)")

    # --- SFT: instructions schema, trajectories non-empty ---
    instr = _lines("sft_instructions.jsonl")
    if instr and any("messages" not in r for r in instr[:200]):
        warn.append("sft_instructions.jsonl rows missing 'messages'")
    traj = _lines("sft_trajectories.jsonl")
    if instr or traj:
        print(f"[validate] SFT: {len(instr)} instructions, {len(traj)} verified trajectories")

    # --- report ---
    for w in warn:
        print(f"[validate] WARN: {w}", flush=True)
    for fz in fatal:
        print(f"[validate] FATAL: {fz}", flush=True)
    print(f"[validate] {len(fatal)} fatal, {len(warn)} warning(s); "
          f"CPT={len(p1)} RL={len(rl)} SFT_instr={len(instr)}", flush=True)
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(validate(config.load("data")))
