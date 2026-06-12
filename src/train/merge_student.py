"""Merge the trained stack (CPT full model + SFT LoRA) -> ONE servable model dir.

Why this exists: models/sft is an ADAPTER-ONLY dir (adapter_model.safetensors, no
config.json) — vLLM serving, veRL's from_pretrained, and perplexity eval all need a full
model. Phase C serves the OUTPUT of this step (models/student_merged), never the base.

Idempotent: skips when the output already has weights newer than both inputs.
Usage: python -m src.train.merge_student [--cpt models/cpt/pr_mastery] [--sft models/sft]
       [--out models/student_merged]
"""
from __future__ import annotations

import argparse

from src import config


def _newest_weight_mtime(d) -> float:
    paths = list(d.glob("*.safetensors")) + list(d.glob("*.bin"))
    return max((p.stat().st_mtime for p in paths), default=0.0)


def merge(cpt: str = "models/cpt/pr_mastery", sft: str = "models/sft",
          out_dir: str = "models/student_merged") -> str:
    cpt_p, sft_p, out_p = (config.ROOT / cpt), (config.ROOT / sft), (config.ROOT / out_dir)
    if not (cpt_p / "config.json").exists():
        raise SystemExit(f"[merge] CPT model missing at {cpt} — train CPT first")
    if not (sft_p / "adapter_model.safetensors").exists():
        raise SystemExit(f"[merge] SFT adapter missing at {sft} — train SFT first")
    if (out_p / "config.json").exists() and _newest_weight_mtime(out_p) >= max(
            _newest_weight_mtime(cpt_p), _newest_weight_mtime(sft_p)):
        print(f"[merge] {out_dir} is up to date — skipping")
        return str(out_p)

    from peft import PeftModel               # lazy: needs the train extra
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[merge] {cpt} + {sft} -> {out_dir}")
    model = AutoModelForCausalLM.from_pretrained(str(cpt_p), torch_dtype="auto")
    model = PeftModel.from_pretrained(model, str(sft_p))
    model = model.merge_and_unload()
    out_p.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_p, safe_serialization=True)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out_p)
    tok = AutoTokenizer.from_pretrained(str(cpt_p))
    tok.save_pretrained(out_p)
    print(f"[merge] done -> {out_dir}")
    return str(out_p)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpt", default="models/cpt/pr_mastery")
    ap.add_argument("--sft", default="models/sft")
    ap.add_argument("--out", default="models/student_merged")
    a = ap.parse_args()
    merge(a.cpt, a.sft, a.out)
