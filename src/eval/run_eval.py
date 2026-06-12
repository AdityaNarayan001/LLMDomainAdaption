"""Evaluate a checkpoint against the BASELINE and append a row to the learning curve.

The baseline = the un-tuned base model. We measure held-out perplexity (knowledge injection /
forgetting) on decontaminated hyperswitch code (eval_sets/heldout_code.jsonl, written by
build_cpt and EXCLUDED from training). Each stage (base -> cpt -> sft -> rl) adds a row to
runs/learning_curve.jsonl, and we print the delta vs the `base` row so the per-stage knowledge
change is visible (and gateable).

Usage: python -m src.eval.run_eval --model <path-or-hf-id> --tag base|cpt|sft|rl
Diagnostic: never raises non-zero into the pipeline (run.sh calls it with `|| true`).
"""
from __future__ import annotations

import argparse
import json
import time

from src import config
from src.eval import perplexity

CURVE = config.RUNS / "learning_curve.jsonl"
HELDOUT = "eval_sets/heldout_code.jsonl"


def _rows() -> list[dict]:
    if not CURVE.exists():
        return []
    return [json.loads(ln) for ln in CURVE.read_text().splitlines() if ln.strip()]


def run(model: str, tag: str, heldout: str = HELDOUT, max_samples: int = 300) -> dict:
    if not (config.ROOT / heldout).exists():
        print(f"[eval] no held-out set at {heldout} (build_cpt writes it) — skipping {tag}")
        return {}
    if any(r.get("tag") == tag and r.get("model") == model for r in _rows()):
        print(f"[eval] {tag} ({model}) already in learning curve — skipping re-eval")
        return {}
    ppl = perplexity.perplexity(model, heldout, max_samples=max_samples)
    row = {"tag": tag, "model": model, "perplexity": round(ppl, 4), "ts": int(time.time())}
    CURVE.parent.mkdir(parents=True, exist_ok=True)
    base = next((r for r in _rows() if r["tag"] == "base"), None)
    with CURVE.open("a") as f:
        f.write(json.dumps(row) + "\n")
    delta = ""
    if base and tag != "base":
        pct = 100.0 * (ppl - base["perplexity"]) / max(1e-9, base["perplexity"])
        delta = f"  | vs base {base['perplexity']:.3f}: {pct:+.1f}%  ({'better' if pct < 0 else 'worse'})"
    print(f"[eval] {tag}: held-out perplexity = {ppl:.3f}{delta}", flush=True)
    return row


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--heldout", default=HELDOUT)
    ap.add_argument("--max-samples", type=int, default=300)
    a = ap.parse_args()
    run(a.model, a.tag, a.heldout, a.max_samples)
