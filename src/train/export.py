"""Export the crowned champion -> a single, vLLM-servable standalone model.

We keep adapters UNMERGED during training (stacking, decision #10); for deployment we
merge the champion's stack (CPT full-FT base -> + SFT LoRA -> + RL LoRA) into ONE model
with ALL artifacts vLLM needs:
  * model weights as **safetensors** (config.json + *.safetensors)
  * full **tokenizer** (tokenizer.json / tokenizer_config.json / special_tokens_map / vocab)
  * **generation_config.json**
  * a SERVE.md with the exact `vllm serve` command + a manifest check

Reads runs/best_model.json (written by flywheel.crown_best). Needs the `train` extra.
"""
from __future__ import annotations

import json

from src import config

REQUIRED = ["config.json", "tokenizer_config.json", "generation_config.json"]


def export(out_dir: str = "models/champion_export") -> str:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    best = json.loads((config.RUNS / "best_model.json").read_text())
    stack = best["adapter_stack"]            # [cpt_full_model, sft_lora, rl_lora]
    base, adapters = stack[0], stack[1:]

    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype="auto")
    for adapter in adapters:                 # apply + merge each LoRA in order
        model = PeftModel.from_pretrained(model, str(config.ROOT / adapter))
        model = model.merge_and_unload()

    out = config.ROOT / out_dir
    out.mkdir(parents=True, exist_ok=True)
    # weights as safetensors (vLLM-preferred) + config + generation_config
    model.save_pretrained(out, safe_serialization=True)
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.save_pretrained(out)
    # tokenizer: from CPT dir if present, else fall back to the original base model
    try:
        tok = AutoTokenizer.from_pretrained(base)
    except Exception:
        tok = AutoTokenizer.from_pretrained(config.load("cpt")["base_model"])
    tok.save_pretrained(out)

    _write_serve_md(out, best)
    missing = [f for f in REQUIRED if not (out / f).exists()]
    has_weights = any(out.glob("*.safetensors"))
    print(f"[export] champion (cycle {best['cycle']}) -> {out}")
    print(f"[export] safetensors: {has_weights} | missing artifacts: {missing or 'none'}")
    if missing or not has_weights:
        raise RuntimeError(f"export incomplete — missing {missing} weights={has_weights}")
    return str(out)


def _write_serve_md(out, best: dict) -> None:
    (out / "SERVE.md").write_text(
        f"""# Champion model — vLLM serving

Crowned cycle {best['cycle']} | sequestered LCB {best.get('sequestered_lcb')} | \
HS-SWE {best.get('hs_swe_solve')} | HS-Knowledge {best.get('hs_knowledge_solve')}

Self-contained model (merged weights + tokenizer + configs). Host with vLLM:

```bash
# from .venv-serve (NVFP4 fast path on Blackwell; drop --quantization for bf16)
.venv-serve/bin/python -m vllm.entrypoints.openai.api_server \\
    --model {out} --port 8000 --quantization modelopt_fp4
```

Then it's an OpenAI-compatible endpoint at http://localhost:8000/v1 .
"""
    )


if __name__ == "__main__":
    export()
