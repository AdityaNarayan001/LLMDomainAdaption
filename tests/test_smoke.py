"""Smoke tests — runnable without GPUs/external services.

Validates: configs load, the package imports, and the pure-logic units (topo sort,
pass@k, McNemar gate, curriculum classification, reward arithmetic) behave correctly.
"""
from __future__ import annotations

import pytest

from src import config


def test_all_configs_load():
    for name in ("data", "cpt", "sft", "rl", "eval", "flywheel"):
        cfg = config.load(name)
        assert isinstance(cfg, dict) and cfg


def test_topo_order():
    from src.data.ingest_repos import Crate, topo_order

    crates = {
        "common_utils": Crate("common_utils", config.ROOT),
        "diesel_models": Crate("diesel_models", config.ROOT, {"common_utils"}),
        "router": Crate("router", config.ROOT, {"diesel_models", "common_utils"}),
    }
    order = topo_order(crates)
    assert order.index("common_utils") < order.index("diesel_models") < order.index("router")


def test_pass_at_k():
    from src.eval.completion import pass_at_k

    assert pass_at_k(10, 0, 1) == 0.0
    assert pass_at_k(10, 10, 1) == 1.0
    assert 0.0 < pass_at_k(10, 5, 1) < 1.0


def test_fim_roundtrip():
    from src.data.build_cpt import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, apply_fim

    # "abcdefghi" -> thirds: prefix=abc, middle=def, suffix=ghi
    out = apply_fim("abcdefghi", "psm")
    assert out == f"{FIM_PREFIX}abc{FIM_SUFFIX}ghi{FIM_MIDDLE}def"
    # PSM and SPM differ only in prefix/suffix order; both end with the middle
    assert apply_fim("abcdefghi", "spm").endswith(f"{FIM_MIDDLE}def")


def test_curriculum_two_mode():
    from src.orchestrate.curriculum import Scheduler

    s = Scheduler(band=(0.2, 0.8))
    for _ in range(10):
        s.record("hard", solved=False)
    for _ in range(10):
        s.record("easy", solved=True)
    for _ in range(5):
        s.record("mid", True)
    for _ in range(5):
        s.record("mid", False)
    assert s.classify("hard") == "sft_on_gold"
    assert s.classify("easy") == "graduated"
    assert s.classify("mid") == "rl_eligible"


def test_reward_tamper_zeroes():
    from src.harness.reward import execution_reward

    w = {"tests": 1.0, "dense": 0.3, "compile": 0.1, "fmt_clippy": 0.05}
    rb = execution_reward(config.ROOT, None, [], tampered=True, weights=w)
    assert rb.total == 0.0 and rb.tampered


def test_mcnemar_gate_promotes_on_paired_wins(tmp_path):
    import json

    from src.eval.report import mcnemar_gate

    # candidate solves a strict superset -> should promote
    cand = {"per_task": [{"task_id": str(i), "solved": True} for i in range(40)]}
    inc = {"per_task": [{"task_id": str(i), "solved": i < 20} for i in range(40)]}
    cp, ip = tmp_path / "c.json", tmp_path / "i.json"
    cp.write_text(json.dumps(cand))
    ip.write_text(json.dumps(inc))
    g = mcnemar_gate(cp, ip)
    assert g.candidate_only_wins == 20 and g.incumbent_only_wins == 0
    assert g.promote


def test_wilson_ci():
    from src.metrics import wilson_ci

    lo, hi = wilson_ci(30, 100)
    assert 0.0 <= lo < 0.30 < hi <= 1.0          # CI brackets the point estimate
    assert wilson_ci(0, 0) == (0.0, 0.0)
    # smaller n => wider interval (the whole point of Risk R3)
    lo_s, hi_s = wilson_ci(3, 10)
    assert (hi_s - lo_s) > (hi - lo)


def test_rl_health_degenerate_and_components():
    from src.train.rl_metrics import compute, health_alerts

    def roll(task, tests, lp):
        return {"task_id": task, "reward": tests, "logprobs": [[lp, lp]],
                "breakdown": {"tests": tests, "dense": tests, "compile": 1.0, "fmt_clippy": 1.0}}

    # group A: all pass, group B: all fail -> both degenerate (no gradient variance)
    rolls = [roll("A", 1.0, -0.1)] * 4 + [roll("B", 0.0, -0.1)] * 4
    h = compute(rolls)
    assert h.n_groups == 2 and h.degenerate_group_rate == 1.0
    assert h.reward_components["fmt_clippy"] == 1.0

    # reward-hack smell: NOTHING passes tests but style/lint reward is high
    hacked = compute([roll("X", 0.0, -0.1)] * 4 + [roll("Y", 0.0, -0.1)] * 4)
    assert hacked.reward_components["tests"] < 0.05
    assert any("REWARD-HACK" in a for a in health_alerts(hacked))


def test_insights_lessons(monkeypatch, tmp_path):
    from src.orchestrate import insights

    monkeypatch.setattr(insights, "INSIGHT_FILE", tmp_path / "insight.txt")

    solved = {"task_id": "t1", "model": "m", "reward": 1.45, "messages": [{"role": "assistant"}],
              "breakdown": {"tests": 1.0, "compile": 1.0, "fmt_clippy": 1.0}}
    compile_fail = {"task_id": "t2", "model": "m", "reward": 0.4,
                    "messages": [{"role": "assistant"}, {"role": "tool", "content": "1 test FAILED"}],
                    "breakdown": {"tests": 0.0, "compile": 1.0, "dense": 0.5, "failed_tests": ["x::y"]}}
    tamper = {"task_id": "t3", "model": "m", "reward": 0.0, "messages": [],
              "breakdown": {"tampered": True, "tests": 0.0}}
    insights.append_many([solved, compile_fail, tamper], cycle=2)

    text = (tmp_path / "insight.txt").read_text()
    assert "SOLVED" in text and "FAILED" in text
    assert "Compiled but tests failed" in text
    assert "Tampered" in text
    assert text.count("=== ") == 3   # one block per rollout


def test_verify_mode_two_pools():
    from src.data.build_rl import verify_mode

    assert verify_mode({"euclid"}) == "execution"            # cheap unit tests, no creds
    assert verify_mode({"common_utils", "euclid"}) == "execution"
    assert verify_mode({"hyperswitch_connectors"}) == "pattern"   # needs live creds
    assert verify_mode({"euclid", "hyperswitch_connectors"}) == "pattern"  # any non-cheap -> pattern
    assert verify_mode(set()) == "pattern"


def test_teacher_prompt_assembly():
    from src.data.teacher import gold_grounded_prompt, oss_instruct_prompt

    p = oss_instruct_prompt("crates/euclid/src/ast.rs", "fn eval() {}")
    assert "crates/euclid/src/ast.rs" in p and "fn eval()" in p and "JSON" in p
    assert "diff itself is the solution" in gold_grounded_prompt("x.rs", "code")


def test_pattern_reward_tamper_zero(tmp_path):
    from src.harness.reward import pattern_reward

    w = {"tests": 1.0, "dense": 0.3, "compile": 0.1, "fmt_clippy": 0.05}
    rb = pattern_reward(tmp_path, None, "cand", "gold", tampered=True, weights=w)
    assert rb.total == 0.0 and rb.mode == "pattern"


def test_venv_router_autoswitch():
    from src import venvs

    # stages map to the right venv (auto-switch)
    assert venvs.python_for("cpt").parent.parent.name == ".venv"
    assert venvs.python_for("sft").parent.parent.name == ".venv"
    assert venvs.python_for("serve").parent.parent.name == ".venv-serve"
    assert venvs.python_for("rl").parent.parent.name == ".venv-rl"
    assert venvs.python_for("flywheel").parent.parent.name == ".venv"
    # unknown stage falls back to default
    assert venvs.python_for("whatever").parent.parent.name == ".venv"
    # interpreter path shape
    assert venvs.interpreter("serve").name == "python"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
