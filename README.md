# LLMDomainAdaption

Fine-tune an open-weight LLM (**Qwen3.5-9B**) into an agentic coding model that is an
expert on the [`juspay/hyperswitch`](https://github.com/juspay/hyperswitch) Rust
codebase — and beats general models on domain-knowledge tasks, measured quantitatively.

Pipeline: **CPT → SFT → RL**, wrapped in a self-improving **flywheel**. MVP on an ASUS
GX10 (GB10, 128GB unified), then scale-out. Full design, decisions, memory envelope,
and risk register are in **[PLAN.txt](PLAN.txt)**.

## Architecture at a glance

| Stage | What | How |
|---|---|---|
| 0 | Eval scoreboard (HS-Knowledge + HS-SWE) | paired McNemar gate, cost/latency axis |
| 1 | Data: code+docs+commits+PRs+issues | dependency-ordered packing, FIM, MinHash dedup |
| 2 | CPT (knowledge injection) | full-FT 9B via 8-bit paged AdamW (Axolotl) |
| 3 | SFT (instruction + agentic) | LoRA on CPT, assistant-only loss mask |
| 4 | RL on the Pi harness | SkyRL **DAPO**, tiered verifiable reward |
| ∞ | Flywheel | rollout → harvest verified → retrain → gate → escalate |

Harness = **Pi** (`pi-mono`). Inference/rollouts = **vLLM** (NVFP4 on Blackwell).
RL = **SkyRL** (veRL fallback). Trainer = **Axolotl**.

## Milestone ladder (front-loaded gates — measure before burning compute)

- **M0** — eval harness + baseline table; validate SkyRL↔Pi↔Polar-proxy; confirm NVFP4 inference.
- **M1** — **task-supply census (HARD GATE):** `scripts/run_pipeline.sh census` → ≥150 verifiable single-crate tasks or re-scope RL.
- **M2** — cold-start clears the floor (trivial-tier solve ≥5%) before any RL.
- **M3** — one full CPT→SFT→RL cycle beats baseline beyond noise → commit to scale-out.

## Quick start

```bash
scripts/setup_env.sh core         # light deps (data/eval/orchestration)
scripts/run_pipeline.sh smoke     # runnable smoke tests (no GPU)

scripts/run_pipeline.sh ingest    # clone hyperswitch + pull PRs/issues
scripts/run_pipeline.sh census 25 # M1 gate (optionally verify-build a sample)
scripts/run_pipeline.sh data      # build CPT/SFT/RL datasets

# On the GX10 (needs train/serve/rl extras):
scripts/setup_env.sh all
scripts/run_pipeline.sh cpt --dry-run    # emits Axolotl configs; drop --dry-run to train
scripts/run_pipeline.sh flywheel --dry-run
```

## Reproduce on a GPU box (the three auto-switching venvs)

The three stacks have mutually-exclusive pins (torch/transformers), so they live in
separate venvs and talk over subprocess/HTTP. `src/venvs.py` + `run_pipeline.sh`
**auto-switch** to the right one per stage; you never activate them by hand.

```bash
# one command provisions all three (Linux + CUDA box):
scripts/setup_env.sh all
#   .venv        -> ".[train,dev,ast]"      pipeline + Axolotl CPT/SFT
#   .venv-serve  -> ".[serve]"              vLLM inference/rollout server
#   .venv-rl     -> symlink to SkyRL's uv venv (cloned as ../SkyRL, `uv sync --extra gpu --extra ray`)
```

RL backend = **veRL** (`setup_env.sh rl` / `install_verl`): `.venv-verl` with `verl` + a
few transitive deps; `.venv-rl` symlinks to it. RL is GRPO/DAPO via `verl.trainer.main_ppo`
with our **verifiable cargo reward** (`src/train/verl_reward.py`) — first pass is single-turn
RLVR (generate patch → verify); multi-turn agentic rollout is the upgrade. (SkyRL was
evaluated but needs the Harbor framework, which doesn't fit our Pi harness.)

**Teacher = Qwen3.5-122B-A10B**, and the run **phases the GPU** so teacher and student never
coexist (they don't fit): `run.sh` serves the **teacher alone** for SFT data-gen → tears it
down → trains the student (CPT/SFT) → serves the **student** for census + the flywheel.

**Keeping a remote training box in sync** (code only — never weights/checkpoints/data):
```bash
scripts/sync_gx10.sh          # local -> remote
scripts/sync_gx10.sh pull     # remote -> local
```
Set the host/dir at the top of `scripts/sync_gx10.sh`.

## Model saving, best-selection & serving

Checkpoints are **disk-bounded and self-curating** (SFT/RL are LoRA, so per-cycle
adapters are tiny):

- Per cycle → `models/cycle_<N>/{sft,rl}`. After each cycle the registry **keeps
  `champion + top-5 + latest`** (`keep_top_k: 5`) and prunes the rest.
- **Promotion/retention** uses the McNemar gate on held-out **HS-Knowledge**.
- The **shipped best** is crowned on the **untouched sequestered set** (max
  lower-confidence-bound + HS-SWE non-regression guard) → `runs/best_model.json`.

**Export a single, vLLM-servable model** (merges the champion's CPT→SFT→RL stack):
```bash
scripts/export_champion.sh            # -> models/champion_export/ (safetensors + tokenizer
                                      #    + config + generation_config + SERVE.md)
scripts/serve_champion.sh             # host it with vLLM (NVFP4); OpenAI endpoint :8000/v1
```
The export is fully self-contained (weights as safetensors, full tokenizer,
`config.json`, `generation_config.json`) — drop the folder on any vLLM host.

## Layout

```
configs/      one YAML per stage (data/cpt/sft/rl/eval/flywheel)
src/data/     ingestion + dataset builders + M1 census
src/harness/  Pi tools, tiered reward, sandboxed runner, SkyRL env adapter
src/eval/     HS-bench runner, perplexity, completion, McNemar gate
src/train/    Axolotl CPT/SFT launchers, SkyRL DAPO launcher
src/orchestrate/  flywheel driver, harvest, registry, curriculum
eval_sets/    committed HS-bench tasks (hs_knowledge / hs_swe / sequestered)
```

Status: scaffold + runnable data/eval/orchestration logic; training stages are
config-driven launchers (need the GPU box + extras). Next: M0/M1 on real data.
