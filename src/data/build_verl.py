"""Convert rl_tasks.jsonl -> veRL training parquet.

veRL expects a parquet with a chat `prompt` column + reward bookkeeping. We pack each
hyperswitch issue->fix task as a prompt (system + issue + the gold files to touch) and stash
the verifier context in `extra_info` so verl_reward.compute_score can apply+grade the patch.
"""
from __future__ import annotations

import json

from src import config

SYSTEM = ("You are a Rust engineer in juspay/hyperswitch. Output a unified diff (```diff ...```)"
          " that resolves the issue, following hyperswitch conventions.")


def build() -> int:
    import pandas as pd  # via datasets/pyarrow

    tasks_path = config.ROOT / "data/datasets/rl_tasks.jsonl"
    if not tasks_path.exists():
        print("WARN: no rl_tasks.jsonl — run build_rl first.")
        return 0
    rows = []
    for line in tasks_path.read_text().splitlines():
        t = json.loads(line)
        rows.append({
            "data_source": "hyperswitch",
            "prompt": [{"role": "system", "content": SYSTEM},
                       {"role": "user", "content": t["issue_text"]
                        + "\n\nFiles to change:\n" + "\n".join(t.get("gold_files", []))}],
            "reward_model": {"style": "rule", "ground_truth": ""},   # reward via custom fn
            "extra_info": {
                "task_id": t["task_id"], "parent_commit": t.get("parent_commit"),
                "crates": t.get("crates", []), "single_crate": t.get("single_crate", False),
                "verify_mode": t.get("verify_mode", "pattern"),
                "test_files": t.get("test_files", []),
                "fail_to_pass": t.get("fail_to_pass", []),
                "setup_patch": t.get("setup_patch"),     # bug-injection: inject before grading
                "verify_cmd": t.get("verify_cmd"),        # e.g. "cargo test -p euclid"
                "gold_patch": t.get("gold_patch"),        # pattern-mode similarity target —
                                                          # without it similarity reward is 0
            },
        })
    out = config.ROOT / "data/datasets/rl_verl_train.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(out)
    return len(rows)


if __name__ == "__main__":
    n = build()
    print(f"veRL parquet: {n} tasks -> data/datasets/rl_verl_train.parquet")
