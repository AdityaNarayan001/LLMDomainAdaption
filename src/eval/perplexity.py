"""Held-out perplexity — CPT forgetting / collapse diagnostic (not a headline metric).

Needs the `train` extra (torch + transformers); imported lazily so the rest of the
eval package runs without it.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from src import config


def _load(model_path: str):
    """Load a full model dir/hub id — OR a LoRA adapter-only dir (adapter_config.json, no
    config.json), which plain from_pretrained rejects with OSError. The SFT overfit guard
    passes models/sft (an adapter), so without this branch the guard silently never runs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    p = config.ROOT / model_path
    if (p / "adapter_config.json").exists() and not (p / "config.json").exists():
        import json as _json
        from peft import PeftModel
        base = _json.loads((p / "adapter_config.json").read_text())["base_model_name_or_path"]
        tok = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto", device_map="auto")
        model = PeftModel.from_pretrained(model, str(p))
    else:
        tok = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype="auto",
                                                     device_map="auto")
    return tok, model


def perplexity(model_path: str, heldout_jsonl: str, max_samples: int = 500) -> float:
    import torch  # lazy

    tok, model = _load(model_path)
    model.eval()

    nll, ntok = 0.0, 0
    lines = Path(config.ROOT / heldout_jsonl).read_text().splitlines()[:max_samples]
    with torch.no_grad():
        for line in lines:
            text = json.loads(line)["training_content"]
            ids = tok(text, return_tensors="pt", truncation=True, max_length=4096).input_ids.to(
                model.device
            )
            if ids.shape[1] < 2:
                continue
            out = model(ids, labels=ids)
            nll += out.loss.item() * (ids.shape[1] - 1)
            ntok += ids.shape[1] - 1
    return math.exp(nll / max(1, ntok))


def forgetting_regression_pct(cpt_ppl: float, sft_ppl: float) -> float:
    """% perplexity regression of SFT vs CPT on held-out repo code (overfit guard)."""
    return 100.0 * (sft_ppl - cpt_ppl) / max(1e-9, cpt_ppl)
