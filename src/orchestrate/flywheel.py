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
    suite_dir = config.ROOT / "eval_sets/trivial"
    if not suite_dir.exists() or not any(suite_dir.glob("*.json")):
        # An EMPTY suite is a broken-harness signal, not a 0% solve-rate: 0/0 would read as
        # 0.0 and NO-GO forever with a misleading message. Say exactly what's missing.
        print("[cold-start] NO-GO: eval_sets/trivial is empty — generate it first "
              "(build_rl_synthetic emits it from caught mutants).")
        return False
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
              reg: registry.Registry, dry_run: bool, served_model: str) -> bool:
    best = reg.best()
    # Rollouts go to the vLLM ENDPOINT, which serves exactly one model id (`served_model`,
    # what run.sh launched — models/student_merged). Requesting best.adapter_stack[-1] (an
    # adapter dir vLLM never loaded) would 404 every request. Adapter hot-swap per cycle is
    # the upgrade path; until then the served model is the rollout policy.
    best_model = served_model
    print(f"\n=== cycle {cycle}: rollouts from {best_model} ===")
    insights.append_hyperparams(cycle)   # full hyperparam snapshot -> insight.txt

    # 1) rollout (best) over an in-band batch
    tasks_path = config.ROOT / "data/datasets/rl_tasks.jsonl"
    if not tasks_path.exists():
        raise SystemExit("[flywheel] data/datasets/rl_tasks.jsonl missing — run the data stage "
                         "(build_rl + build_rl_synthetic) before the flywheel")
    rl_tasks = [json.loads(l) for l in tasks_path.read_text().splitlines() if l.strip()]
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

    # 3) train (sft + rl) into PER-CYCLE dirs (so checkpoints don't overwrite -> retention).
    #    SFT (LoRA) always stacks on the CPT FULL model — an adapter dir cannot be an Axolotl
    #    base_model. RL needs a FULL model too, so the per-cycle SFT adapter is merged onto
    #    CPT first (veRL's from_pretrained cannot load adapter-only dirs).
    cycle_dir = f"models/cycle_{cycle}"
    sft_out, rl_out = f"{cycle_dir}/sft", f"{cycle_dir}/rl"
    if not dry_run:
        from src.train import merge_student, sft as sft_train
        sft_train.run(cpt_ckpt="models/cpt/pr_mastery", out_dir=sft_out)   # Axolotl, in .venv
        merged = merge_student.merge(cpt="models/cpt/pr_mastery", sft=sft_out,
                                     out_dir=f"{cycle_dir}/merged")
        venvs.run_rl("--sft-ckpt", merged, "--out", rl_out)       # veRL, shelled into .venv-rl

    # 4) eval candidate on frozen HS-Knowledge + 5) McNemar gate.
    # NOTE: evaluating the CANDIDATE requires vLLM to serve it (adapter hot-swap / re-serve);
    # with a single resident server this fails — fail SOFT (no promotion) with the reason,
    # never crash the loop.
    cand_model = rl_out
    promote = True
    solve = 0.0
    if not dry_run:
        try:
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
        except Exception as exc:
            promote = False
            print(f"[cycle {cycle}] candidate eval failed ({exc}) — not promoting. "
                  f"Serve {cand_model} (or hot-swap the adapter) to enable candidate eval.")

    # 6) promote + 7) ledger + 8) prune to champion + top-K + latest (disk-bounded)
    cand_ckpt = registry.Checkpoint(
        cycle=cycle, base_model=config.load("cpt")["base_model"],
        adapter_stack=["models/cpt/pr_mastery", sft_out, rl_out],
        data_manifest_hash="(stamped)", solve_rate=solve, adapter_dir=cycle_dir,
    )
    champ = reg.promote(cand_ckpt, best, promote)
    promoted = champ is cand_ckpt
    kept, deleted = reg.prune()
    _ledger_append({"cycle": cycle, "solve_rate": solve, "promoted": promoted,
                    "n_harvested": len(new_sft), "kept": sorted(kept), "pruned": deleted})
    if deleted:
        metrics.log("flywheel", "prune", cycle=cycle, deleted=len(deleted), kept=len(kept))
    return promoted


def crown_best(reg: registry.Registry, endpoint: str, hs_swe_floor: float) -> None:
    """Tier-2: decide the SHIPPED best on the UNTOUCHED sequestered set (decision-leakage-free).
    Evaluate only the kept candidates; pick max lower-confidence-bound with HS-SWE non-regression.
    Writes runs/best_model.json (the deployable pointer). Touched only here, not per-cycle."""
    weights = config.load("rl")["reward"]["weights"]
    scored: list[registry.Checkpoint] = []
    for c in reg.all():
        if c.adapter_dir not in reg.keepers():
            continue
        seq = internal_swebench.run_suite("sequestered", "eval_sets/sequestered",
                                          model=c.adapter_stack[-1], endpoint=endpoint,
                                          weights=weights, closed_context=True)
        swe = internal_swebench.run_suite("hs_swe", "eval_sets/hs_swe",
                                          model=c.adapter_stack[-1], endpoint=endpoint,
                                          weights=weights)
        n = len(seq.per_task)
        c.sequestered_lcb = metrics.wilson_ci(sum(t.solved for t in seq.per_task), n)[0]
        c.hs_swe_solve = swe.solve_rate
        scored.append(c)
    if not scored:
        return
    best = max(
        [c for c in scored if (c.hs_swe_solve or 0.0) >= hs_swe_floor] or scored,
        key=lambda c: (c.sequestered_lcb, c.solve_rate),
    )
    (config.RUNS / "best_model.json").write_text(json.dumps({
        "adapter_stack": best.adapter_stack, "adapter_dir": best.adapter_dir,
        "sequestered_lcb": best.sequestered_lcb, "hs_swe_solve": best.hs_swe_solve,
        "hs_knowledge_solve": best.solve_rate, "cycle": best.cycle,
    }, indent=2))
    metrics.log("flywheel", "crowned_best", cycle=best.cycle,
                sequestered_lcb=round(best.sequestered_lcb, 4), hs_swe=round(best.hs_swe_solve or 0, 4))


def main(cycles: int = 1000, endpoint: str = "http://localhost:8000", dry_run: bool = False,
         model: str = "models/student_merged") -> None:
    """`model` = the id vLLM is SERVING at `endpoint` (run.sh serves models/student_merged) —
    every rollout/eval request must use this id or vLLM 404s."""
    cfg = config.load("flywheel")
    rl_cfg = config.load("rl")
    sched = curriculum.Scheduler(
        band=tuple(rl_cfg["curriculum"]["solve_band"]),
        diversity_max_connector_frac=rl_cfg["curriculum"]["diversity_max_connector_frac"],
    )
    reg = registry.Registry(keep_top_k=cfg["registry"]["keep_top_k"])
    patience = cfg["stopping"]["plateau_patience_cycles"]

    # cold-start gate: ignite above the floor before any RL (autonomous: halt-with-reason).
    # dry_run skips it entirely — it performs REAL rollouts, which dry-run must never do.
    if not dry_run and not cold_start(cfg, endpoint, model=model):
        metrics.log("flywheel", "HALT", reason="cold-start below floor")
        raise SystemExit(3)

    flat = 0  # consecutive non-promoting cycles -> plateau
    c = -1    # defined even when cycles == 0 (the final log below uses it)
    for c in range(cycles):
        promoted = run_cycle(c, cfg, endpoint, sched, reg, dry_run, served_model=model)
        flat = 0 if promoted else flat + 1
        if flat >= patience:
            metrics.log("flywheel", "PLATEAU",
                        reason=f"{patience} cycles without promotion — stopping",
                        cycle=c)
            break
    # crown the SHIPPED best on the sequestered set (Tier-2), then we're done
    if not dry_run:
        crown_best(reg, endpoint, cfg["registry"]["hs_swe_floor"])
    metrics.log("flywheel", "done", last_cycle=c)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cycles", type=int, default=1000)
    ap.add_argument("--endpoint", default="http://localhost:8000")
    ap.add_argument("--model", default="models/student_merged",
                    help="the model id vLLM is serving at --endpoint")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(cycles=a.cycles, endpoint=a.endpoint, dry_run=a.dry_run, model=a.model)
