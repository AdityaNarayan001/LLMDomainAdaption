"""The flywheel driver: cold-start ignition, then the compounding cycle (the heart).

cycle -1 (cold-start): synthetic-easy ladder + SFT-on-gold + execution-free reward,
   with a <5% go/no-go (Risk R1).
cycle 0+:
   rollout(best) -> harvest(verified) -> train(sft/rl) -> eval(frozen HS-Knowledge)
   -> McNemar gate -> promote if fired -> escalate -> append ledger.

This conductor wires together harvest/registry/curriculum + eval.report + train.*.
Heavy steps (train.*, vLLM rollouts) shell out; the loop logic itself runs anywhere.
"""
from __future__ import annotations

import json

from src import config, metrics, venvs
from src.eval import internal_swebench, report
from src.orchestrate import curriculum, harvest, insights, registry, rollout_store
from src.train import rl_metrics


def _ledger_append(row: dict) -> None:
    out = config.ROOT / config.load("flywheel")["ledger"]
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        f.write(json.dumps(row) + "\n")


def cold_start(cfg: dict, endpoint: str, model: str) -> bool:
    """Lift solve-rate above the floor before any online RL. Returns True if the gate passes."""
    fw = cfg["cold_start"]
    print("[cold-start] synthetic-easy ladder + SFT-on-gold + execution-free reward")
    # 1) run the trivial-tier suite to measure ignition
    res = internal_swebench.run_suite(
        "trivial", "eval_sets/trivial", model=model, endpoint=endpoint,
        weights=config.load("rl")["reward"]["weights"],
    )
    floor = fw["go_no_go_min_solve"]
    print(f"[cold-start] trivial-tier solve-rate = {res.solve_rate:.1%} (floor {floor:.0%})")
    if res.solve_rate < floor:
        print("[cold-start] NO-GO: fix harness/curriculum before spending RL compute.")
        return False
    print("[cold-start] GO: flywheel ignition reached.")
    return True


def run_cycle(cycle: int, cfg: dict, endpoint: str, sched: curriculum.Scheduler,
              reg: registry.Registry, dry_run: bool) -> None:
    best = reg.best()
    best_model = best.adapter_stack[-1] if best else "models/sft"
    print(f"\n=== cycle {cycle}: rollouts from {best_model} ===")
    insights.append_hyperparams(cycle)   # full hyperparam snapshot -> insight.txt

    # 1) rollout (best) over an in-band batch
    rl_tasks = [json.loads(l) for l in (config.ROOT / "data/datasets/rl_tasks.jsonl").read_text().splitlines()]
    batch = sched.select_rl_batch(rl_tasks, cfg["cycle"]["rollout_tasks_per_cycle"])
    rollouts = []
    if not dry_run:
        from src.harness import skyrl_env
        weights = config.load("rl")["reward"]["weights"]
        for task in batch:
            for _ in range(cfg["cycle"]["samples_per_task"]):
                roll = skyrl_env.HyperswitchTaskEnv(
                    task=task, endpoint=endpoint, model=best_model, weights=weights
                ).rollout()
                rollouts.append(roll)
                sched.record(task["task_id"], roll["reward"] >= 1.0)

    # persist ALL raw rollouts (token-faithful) before filtering — reproducibility/lineage
    if rollouts:
        shard = rollout_store.save(cycle, rollouts, model=best_model)
        # per-rollout human-readable "what happened & what to learn" -> runs/insight.txt
        insights.append_many(rollouts, cycle)
        # RL health: degenerate-group rate, entropy, reward breakdown, solve-rate
        health = rl_metrics.compute(rollouts)
        metrics.log("rl", "batch_health", cycle=cycle, shard=shard.name,
                    **rl_metrics.to_log_fields(health))
        for alert in rl_metrics.health_alerts(health):
            metrics.log("rl", "ALERT", cycle=cycle, msg=alert)

    # 2) harvest verified -> new data (decontaminated)
    new_sft, new_rl = harvest.harvest(rollouts) if rollouts else ([], [])
    harvest.append_jsonl(new_sft, "data/datasets/sft_trajectories.jsonl")
    print(f"[cycle {cycle}] harvested {len(new_sft)} verified trajectories")

    # 3) train (sft + rl) from the current best.
    #    SFT runs in THIS venv (Axolotl); RL AUTO-SWITCHES to .venv-rl (SkyRL) via venvs.run_rl.
    if not dry_run:
        from src.train import sft as sft_train
        sft_train.run(cpt_ckpt=best_model)        # Axolotl, in .venv
        venvs.run_rl()                            # SkyRL, shelled into .venv-rl

    # 4) eval candidate on frozen HS-Knowledge + 5) McNemar gate
    cand_model = "models/rl"
    promote = True
    if not dry_run:
        cand = internal_swebench.run_suite(
            "hs_knowledge", "eval_sets/hs_knowledge", model=cand_model, endpoint=endpoint,
            weights=config.load("rl")["reward"]["weights"], closed_context=True,
        )
        internal_swebench.save(cand)
        if best:
            gate = report.mcnemar_gate(
                config.RUNS / "eval" / f"hs_knowledge_{cand_model}_0.json",
                config.RUNS / "eval" / f"hs_knowledge_{best.adapter_stack[-1]}_0.json",
            )
            promote = gate.promote
            print(f"[cycle {cycle}] gate: p={gate.p_value:.3f} promote={promote} MDE={gate.mde_pp}pp")
        solve = cand.solve_rate
    else:
        solve = 0.0

    # 6) promote + 7) ledger
    cand_ckpt = registry.Checkpoint(
        cycle=cycle, base_model=config.load("cpt")["base_model"],
        adapter_stack=["models/cpt/pr_mastery", "models/sft", cand_model],
        data_manifest_hash="(stamped)", solve_rate=solve,
    )
    champ = reg.promote(cand_ckpt, best, promote)
    _ledger_append({"cycle": cycle, "solve_rate": solve, "promoted": champ is cand_ckpt,
                    "n_harvested": len(new_sft)})


def main(cycles: int = 3, endpoint: str = "http://localhost:8000", dry_run: bool = False) -> None:
    cfg = config.load("flywheel")
    rl_cfg = config.load("rl")
    sched = curriculum.Scheduler(
        band=tuple(rl_cfg["curriculum"]["solve_band"]),
        diversity_max_connector_frac=rl_cfg["curriculum"]["diversity_max_connector_frac"],
    )
    reg = registry.Registry()

    if not cold_start(cfg, endpoint, model="models/sft") and not dry_run:
        return
    for c in range(cycles):
        run_cycle(c, cfg, endpoint, sched, reg, dry_run)


if __name__ == "__main__":
    import sys
    main(dry_run="--dry-run" in sys.argv)
