"""Teacher-driven SFT instruction synthesis (fixes the 'instruction generation not wired' gap).

Teacher model: configurable (configs/sft.yaml teacher.model) — currently **Qwen3.5-27B**,
self-hosted via vLLM on the GX10 (Phase A, never co-resident with the student). Apache-2.0
=> distilled outputs are license-clean to train on AND publish (public repo). Same family
as the 9B student => good distribution match.

Key design (grounded in the repo analysis):
  * Most SFT *solutions* are GOLD from the repo (real connector code, PR diffs, the
    connector-template) — the teacher mostly phrases the *instruction/problem* and adds
    *reasoning* around that gold code (OSS-Instruct), rather than inventing solutions.
    This minimizes teacher dependence and hallucinated solutions.
  * Generation is OFFLINE: query teacher -> dump data/datasets/sft_instructions_raw.jsonl,
    which build_sft.build() then formats. Decouples teacher memory from training (Risk R5).

Prompt assembly is pure (unit-tested); the actual teacher call hits a vLLM endpoint.
"""
from __future__ import annotations

import json
import random
import re

import requests

from src import config
from src.data import ast_rust

OSS_INSTRUCT = """You are creating a high-quality coding exercise for the hyperswitch \
Rust codebase. Given the REAL code below, write ONE realistic engineering task a \
developer might be asked, then give the reference solution. The solution should match \
hyperswitch conventions (error types, connector patterns, conventional commits).

REAL CODE ({path}):
```rust
{snippet}
```

Respond as JSON: {{"problem": "...", "solution": "...", "reasoning": "..."}}"""

GOLD_GROUNDED = """Below is a REAL merged change from hyperswitch. Write the natural-language \
issue/instruction that would have led a developer to make exactly this change, plus a short \
reasoning of why. Do NOT restate the diff.

CHANGE ({path}):
```rust
{snippet}
```

Respond as JSON: {{"problem": "...", "reasoning": "..."}}  (the diff itself is the solution)"""


def _extract_json(text: str) -> dict:
    """Robustly parse a teacher reply to a dict. response_format=json_object already
    constrains output, but reasoning models (Qwen3.5/Nemotron) can prepend <think>… or a
    'Thinking Process:' preamble — strip it and fall back to the first balanced {...}."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def _call_teacher(endpoint: str, model: str, prompt: str, temperature: float = 0.7) -> str:
    resp = requests.post(
        f"{endpoint}/v1/chat/completions",
        json={"model": model, "messages": [{"role": "user", "content": prompt}],
              "temperature": temperature, "max_tokens": 1500,
              "response_format": {"type": "json_object"}},
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def oss_instruct_prompt(path: str, snippet: str) -> str:
    return OSS_INSTRUCT.format(path=path, snippet=snippet)


def gold_grounded_prompt(path: str, snippet: str) -> str:
    return GOLD_GROUNDED.format(path=path, snippet=snippet)


def sample_function_snippets(cfg: dict, n: int, seed: int = 0) -> list[tuple[str, str]]:
    """Sample real function snippets (AST-extracted, fallback regex) to seed OSS-Instruct."""
    from src.data import ingest_repos

    repo = config.ROOT / cfg["source"]["raw_repo"]
    files = ingest_repos.iter_source_files(cfg, repo)
    rng = random.Random(seed)
    rng.shuffle(files)
    out: list[tuple[str, str]] = []
    for p in files:
        if not str(p).endswith(".rs"):
            continue
        code = p.read_text(encoding="utf-8", errors="ignore")
        for fn in ast_rust.extract_functions(code):
            out.append((str(p.relative_to(repo)), fn))
            if len(out) >= n:
                return out
    return out


def generate(cfg: dict, teacher_cfg: dict, n: int = 2000) -> int:
    """Query the teacher over sampled snippets -> sft_instructions_raw.jsonl. Returns count."""
    endpoint, model = teacher_cfg["endpoint"], teacher_cfg["model"]
    out = config.ROOT / "data/datasets/sft_instructions_raw.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as f:
        for path, snippet in sample_function_snippets(cfg, n):
            try:
                raw = _call_teacher(endpoint, model, oss_instruct_prompt(path, snippet))
                rec = _extract_json(raw)         # skip-on-parse-error = teacher-output validation
                rec["snippet"] = snippet
                f.write(json.dumps(rec) + "\n")
                f.flush()                        # crash-safe: each teacher call persisted
                written += 1
                if written % 50 == 0:
                    print(f"  generated {written} instructions -> {out.name}", flush=True)
            except Exception:  # pragma: no cover - network/parse dependent
                continue
    return written


def gen_trajectories(endpoint: str, model: str, weights: dict, n_tasks: int = 200,
                     samples_per_task: int = 4) -> int:
    """Run a (teacher/base) model through the Pi harness on RL tasks -> raw trajectories.
    Writes data/datasets/sft_trajectories_raw.jsonl ({chat, reward}); build_sft then
    rejection-samples to verified successes. Needs the vLLM endpoint up."""
    from src.harness import runner

    tasks_path = config.ROOT / "data/datasets/rl_tasks.jsonl"
    if not tasks_path.exists():
        print("WARN: no rl_tasks.jsonl — run build_rl first; skipping trajectory gen.")
        return 0
    tasks = [json.loads(line) for line in tasks_path.read_text().splitlines()][:n_tasks]
    out = config.ROOT / "data/datasets/sft_trajectories_raw.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out.open("w") as f:
        for task in tasks:
            for _ in range(samples_per_task):
                try:
                    traj = runner.run_task(task, endpoint=endpoint, model=model,
                                           weights=weights, temperature=0.8)
                except Exception:  # pragma: no cover - runtime/env dependent
                    continue
                f.write(json.dumps({
                    "chat": {"messages": traj.messages, "loss_mask": traj.loss_mask},
                    "reward": traj.reward,
                }) + "\n")
                f.flush()                        # crash-safe: each rollout persisted
                written += 1
                if written % 25 == 0:
                    print(f"  generated {written} trajectories -> {out.name}", flush=True)
    return written


if __name__ == "__main__":
    cfg = config.load("data")
    tcfg = config.load("sft")["teacher"]
    print(f"teacher={tcfg['model']} -> generating SFT instructions...")
    print(f"wrote {generate(cfg, tcfg)} instruction records")
