"""Generate ONE illustrative record for each dataset type using the real builder code,
so the example schemas are guaranteed to match what the pipeline actually emits.

Writes to examples/*.jsonl (committed as schema documentation) and prints them.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # repo root on path

from src.data import build_cpt, build_rl, build_sft

OUT = Path(__file__).resolve().parent.parent / "examples"
OUT.mkdir(exist_ok=True)

RUST = '''pub fn refund(req: RefundRequest) -> Result<RefundResponse, Error> {
    let amount = req.amount.ok_or(Error::MissingAmount)?;
    Ok(RefundResponse { id: req.id, amount, status: Status::Pending })
}
'''


def w(name: str, obj) -> None:
    (OUT / name).write_text(json.dumps(obj, indent=2) + "\n")
    print(f"\n===== examples/{name} =====")
    print(json.dumps(obj, indent=2))


# --- CPT: one document-masked, file-marked, FIM-transformed sample ---
marker = "// File: crates/router/src/refunds.rs\n"
cpt = {
    "training_content": build_cpt.apply_fim(marker + RUST, "psm", structure_aware=False),
    "meta": {"path": "crates/router/src/refunds.rs", "phase": "foundation"},
}
w("cpt_example.jsonl", cpt)

# --- SFT: a grounded instruction record (assistant-only loss mask) ---
instr = build_sft.assemble_instruction(
    snippet=RUST,
    problem="Add idempotency-key handling to the refund flow.",
    solution="To make refunds idempotent, thread an `idempotency_key` through "
             "`RefundRequest` and short-circuit if a refund with that key exists:\n"
             "```rust\nif let Some(k) = req.idempotency_key { /* lookup */ }\n```",
)
w("sft_instruction_example.jsonl", instr)

# --- SFT: a multi-turn agent trajectory record (rejection-sampled success) ---
T = build_sft.Turn
traj = build_sft.to_chat([
    T("system", build_sft.SYSTEM, loss=False),
    T("user", "Fix #5012: refund returns Pending even after gateway confirms success.", loss=False),
    T("assistant", "I'll inspect the refund status mapping in the stripe connector.", loss=True),
    T("tool", "crates/.../stripe/transformers.rs:212: Status::Pending => ...", loss=False),
    T("assistant", "The success branch is missing. Editing the match to map "
                   "`succeeded` -> Status::Success.", loss=True),
    T("tool", "ok", loss=False),
    T("assistant", "Running the targeted test.", loss=True),
    T("tool", "test stripe::tests::refund_success ... ok", loss=False),
    T("assistant", "Tests pass. The mapping now reflects gateway confirmation.", loss=True),
])
traj["reward"] = 1.30
w("sft_trajectory_example.jsonl", traj)

# --- RL: one issue->fix task with a verifier ---
task = build_rl.RLTask(
    task_id="hs-pr-5012",
    pr_number=5012,
    parent_commit="a1b2c3d4e5f6",
    issue_text="Refund returns Pending even after the gateway confirms success.",
    gold_files=["crates/hyperswitch_connectors/src/connectors/stripe/transformers.rs"],
    test_files=["crates/hyperswitch_connectors/src/connectors/stripe/tests.rs"],
    crates=["hyperswitch_connectors"],
    single_crate=True,
    verify_cmd="cargo nextest run -p hyperswitch_connectors",
)
w("rl_task_example.jsonl", dataclasses.asdict(task))

print("\nAll example records written to examples/ (committed as schema docs).")
