"""Orchestrate SFT data generation: instructions (teacher) + trajectories (harness) + format.

Runs in the `data` stage AFTER build_rl, and needs the vLLM endpoint up (run.sh has it).
Teacher defaults to whatever vLLM is serving (the base model) — so NO mandatory 122B
download; set sft.yaml teacher.model to the 122B (and PREP_TEACHER=1) only to distill from
the stronger teacher.

Flow:
  1. teacher.generate         -> sft_instructions_raw.jsonl   (OSS-Instruct over real code)
  2. teacher.gen_trajectories -> sft_trajectories_raw.jsonl   (harness rollouts on RL tasks)
  3. build_sft.build/save     -> sft_instructions.jsonl + sft_trajectories.jsonl
                                 (formatted, assistant-only mask, rejection-sampled successes)
"""
from __future__ import annotations

from src import config
from src.data import build_sft, teacher


def run() -> None:
    data_cfg = config.load("data")
    sft_cfg = config.load("sft")
    tcfg = sft_cfg["teacher"]
    endpoint, model = tcfg["endpoint"], tcfg["model"]
    weights = config.load("rl")["reward"]["weights"]

    n_target = int(tcfg.get("n_instructions", 1000))   # quality >> quantity; configurable
    print(f"[gen_sft] instructions via teacher={model} @ {endpoint} (n={n_target})")
    n_instr = teacher.generate(data_cfg, tcfg, n=n_target)
    print(f"[gen_sft] {n_instr} raw instructions")

    print("[gen_sft] trajectories via harness rollouts (rejection-sampled to successes)")
    n_traj = teacher.gen_trajectories(endpoint, model, weights, n_tasks=200, samples_per_task=4)
    print(f"[gen_sft] {n_traj} raw trajectories")

    instr, traj = build_sft.build(data_cfg)
    build_sft.save(instr, traj)
    print(f"[gen_sft] formatted -> {len(instr)} instructions, {len(traj)} verified trajectories")


if __name__ == "__main__":
    run()
