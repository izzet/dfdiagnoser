"""Offline fact -> finding -> plan path for the StorMer coordinated input+comm loop (synthetic windows).

Control-path checks through the PRODUCTION hops (FactEngine loop rule file -> LiveDiagnoser -> cbandit); windows
are shapes that mimic the rep0 regimes of STEP0-COMM-GATE-2026-09-25.md, not measurements.  No detection
quality, performance or trajectory evidence.  Design: liveflow_shm/STORMER-LOOP-DESIGN-2026-09-26.md.
"""
from pathlib import Path

import pytest

try:
    from . import stormer_loop_fixtures as f
    from . import test_stormer_comm_rule as c
    from . import test_coherent_fact_handoff as h
except ImportError:
    import stormer_loop_fixtures as f
    import test_stormer_comm_rule as c
    import test_coherent_fact_handoff as h

from dfdiagnoser.diagnoser import Diagnoser
from dftracer.analyzer.fact_engine import FactEngine

_RULES = c._RULE.parent
LADDER = [25, 50, 100, 200, 400]
COMM = {25: 3.0, 50: 2.0, 100: 1.3, 200: 0.9, 400: 0.6}


def test_loop_rule_file_bodies_equal_the_standalone_rules():
    def body(path):
        return path.read_text().split("\nrules:\n", 1)[1].strip()
    loop = (_RULES / "liveflow-stormer-loop.yaml").read_text().split("\nrules:\n", 1)[1]
    for name in ("liveflow-stormer-input-fetch-slope-exposure.yaml", "liveflow-stormer-comm-slope-exposure.yaml"):
        assert body(_RULES / name) in loop, name


def test_loop_rules_use_no_constant_beyond_reference_and_zero():
    import re
    for line in (_RULES / "liveflow-stormer-loop.yaml").read_text().splitlines():
        if line.strip().startswith("when:"):
            assert set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", line)) <= {"0", "0.0", "1", "1.0"}


def _run(monkeypatch, env_fn, n=40, *, joint=False, ladder=True, allow=("num_workers", "bucket_cap_mb"),
         priority=True):
    log = h._DecisionLog()
    p = f.stormer_planner(monkeypatch, allow=allow, priority=priority,
                          env={"DFOPTIMIZER_CBANDIT_JOINT_VERIFY": "1" if joint else "0"})
    monkeypatch.setattr(h.cbandit_planner, "logger", log)
    if ladder and "bucket_cap_mb" in p._levels:
        p._levels["bucket_cap_mb"] = list(LADDER)         # what train.py registers (values=[25,...,400])
    eng, diag = f.loop_engine(), Diagnoser()
    cfg, trace, allplans = {"W": 8, "cap": 25}, [], []
    for w in range(1, n + 1):
        kw = env_fn(w, cfg)
        rows, ctx = f.window(w, **kw)
        facts, fnd, plans = f.deliver(eng, diag, p, rows, ctx)
        step = [(pl.knob_id.split(".")[-1], pl.old_value, pl.new_value) for pl in plans]
        trace.append((w, dict(cfg), step))
        allplans += step
        for pl in plans:                                   # applied at the next window boundary
            if pl.knob_id.endswith("bucket_cap_mb"):
                cfg["cap"] = pl.new_value
            elif pl.knob_id.endswith("num_workers"):
                cfg["W"] = pl.new_value
    return p, trace, allplans, [k for e, k in log.entries if e == "optimizer.cbandit.decision"], cfg


def _rep0(w, cfg):
    """cap 25: comm-dominant, no stall.  Larger cap: comm falls, periodic input stalls until W grows."""
    stall = (2.2 if cfg["W"] < 12 else 0.2) if cfg["cap"] >= 50 else 0.3
    return dict(comm=COMM[cfg["cap"]], input_s=stall, compute=2.4)


def test_comm_dominant_window_acts_on_bucket_cap_first_then_input_follows(monkeypatch):
    _p, trace, plans, decisions, _ = _run(monkeypatch, _rep0, joint=True, n=12)
    assert plans[0] == ("bucket_cap_mb", 25, 50)                   # comm domain owns the first action
    assert ("num_workers", 8, 12) in plans                          # relieving comm exposed input: it follows
    first_w = next(i for i, p in enumerate(plans) if p[0] == "num_workers")
    assert first_w > 0 and plans[first_w - 1][0] == "bucket_cap_mb"
    assert decisions and all(d.get("decision_path") for d in decisions)
    assert all("config_at_decision" in d for d in decisions)       # per-node timeline (A2)
    assert all(d["barrier_redirect_enabled"] is False for d in decisions)     # A1 OFF


def test_default_per_arm_verification_reverts_the_comm_move_once_input_is_exposed(monkeypatch):
    """CHARACTERISATION of the historical behaviour (the rep0 failure shape): the cap arm's windows carry the
    stalls the cap itself exposed, so its own verification reads harm and undoes the move."""
    _p, trace, plans, _d, cfg = _run(monkeypatch, _rep0, joint=False, n=40)
    assert ("bucket_cap_mb", 25, 50) in plans
    assert ("bucket_cap_mb", 50, 25) in plans                      # reverted
    assert cfg["cap"] == 25                                         # and the comm relief is lost for the run


def test_joint_verification_keeps_the_pair_and_finishes_the_ladder(monkeypatch):
    _p, trace, plans, _d, cfg = _run(monkeypatch, _rep0, joint=True, n=40)
    assert not any(p[2] < p[1] for p in plans if p[0] == "bucket_cap_mb"), plans     # no revert
    assert cfg == {"W": 12, "cap": 400}
    assert [p[2] for p in plans if p[0] == "bucket_cap_mb"] == [50, 100, 200, 400]


def test_skew_caused_wait_is_input_action_not_comm(monkeypatch):
    """Peers' comm is inflated by one late rank's input: A = min(E, W) moves it from comm to input (skew link)."""
    def env(w, cfg):
        return dict(comm=3.0, input_s=0.3, compute=2.4, A=2.0, E=2.4, W=2.2)
    _p, trace, plans, decisions, _ = _run(monkeypatch, env, n=14, joint=True)
    assert plans and all(p[0] == "num_workers" for p in plans), plans
    assert not any(p[0] == "bucket_cap_mb" for p in plans)
    assert any(d["decision_path"].startswith(("skew_attributed", "actionable_nondominant"))
               for d in decisions if d.get("decision") == "move")


def test_uniform_comm_with_no_skew_is_a_comm_action_not_an_input_action(monkeypatch):
    _p, trace, plans, decisions, _ = _run(
        monkeypatch, lambda w, cfg: dict(comm=3.0, input_s=0.3, compute=2.4), n=6, joint=True)
    assert plans and plans[0][0] == "bucket_cap_mb"


def test_missing_skew_coverage_blocks_the_comm_fact_and_therefore_any_cap_action(monkeypatch):
    """Blocking probe off (native async DDP): no per-rank comm record, coverage 0 -> comm rule cannot fire."""
    _p, trace, plans, decisions, _ = _run(
        monkeypatch, lambda w, cfg: dict(comm=3.0, input_s=0.3, compute=2.4, cov=0.0), n=14, joint=True)
    assert not any(p[0] == "bucket_cap_mb" for p in plans)


def test_residual_dominant_window_still_declines(monkeypatch):
    """C2 guard (judge batch 2, C2a): a window whose unattributed remainder exceeds the actionable exposure
    declines even with the actionable-priority flag ON."""
    _p, trace, plans, decisions, _ = _run(
        monkeypatch, lambda w, cfg: dict(comm=1.0, input_s=0.3, compute=1.0, wall=20.0), n=10, joint=True)
    assert plans == []
    assert decisions and all(d["decision"] == "decline" for d in decisions)


def test_window_noise_makes_the_loop_abstain_and_records_why(monkeypatch):
    """Real-cell shape (judge batch 2, C7): window-wall sigma ~40 percent of the mean and a short horizon leave
    n_required beyond the horizon, so nothing moves and the decision record says why."""
    def env(w, cfg):
        comm = 3.0 if w % 2 else 0.6
        return dict(comm=comm, input_s=0.3, compute=2.4, remaining=12.0)
    _p, trace, plans, decisions, _ = _run(monkeypatch, env, n=12, joint=True)
    assert plans == []
    text = str([d.get("escalation_state") or d.get("reason") for d in decisions])
    assert "unverifiable_in_horizon" in text or "no_predicted_effect" in text or "decline" in text


def test_allowlist_excludes_prefetch_and_unrestricted_input_domain_would_reach_it(monkeypatch):
    def env(w, cfg):
        return dict(comm=0.4, input_s=2.5, compute=2.4)
    _p, _t, plans_allow, _d, _ = _run(monkeypatch, env, n=30, joint=True)
    assert not any(p[0] == "prefetch_factor" for p in plans_allow)
    assert "prefetch_factor" not in _p._short.values()
    _p2, _t2, plans_free, _d2, _ = _run(monkeypatch, env, n=30, joint=True, allow=None)
    assert "prefetch_factor" in _p2._short.values()       # the domain owns it: the allowlist is what removes it


def test_num_workers_is_bounded_by_the_registered_range_and_strict_explore_never_goes_below_original(monkeypatch):
    def env(w, cfg):
        return dict(comm=0.4, input_s=2.5, compute=2.4)
    _p, trace, plans, _d, cfg = _run(monkeypatch, env, n=60, joint=True)
    w_moves = [p for p in plans if p[0] == "num_workers"]
    assert all(8 <= p[2] <= f.KNOB_MAX_WORKERS for p in w_moves), w_moves


def _ev(pid, name, cat, func, seconds, n=1):
    return [{"window": 7, "epoch": 1, "step": 1, "pid": pid, "proc_name": name, "cat": cat,
             "func_name": func, "io_cat": 0, "time": seconds, "size": 0} for _ in range(n)]


def test_raw_events_publish_the_family_slope_columns_both_loop_rules_read():
    """The analyzer itself (not a synthetic frame) produces comm_all_reduce_* and fetch_iter_* family columns
    from StorMer-shaped raw events, and the late-arriving rank's family slope is on the right side of 1."""
    import pandas as pd
    from dftracer.analyzer.config import AnalyzerPresetConfigStormerXLayer
    from dftracer.analyzer.fastpath import FastPathPlan
    preset = AnalyzerPresetConfigStormerXLayer()
    plan = FastPathPlan.build(preset, ["window", "proc_name"], list(preset.layer_defs), f.loop_engine())
    assert plan is not None
    rows = []
    # ranks 10,11,12 wait long in all-reduce (rank 13 arrived last: least wait); rank 10 also waits longest in next(loader)
    spec = {10: (5.0, 1.0), 11: (3.0, 0.3), 12: (3.0, 0.3), 13: (0.5, 0.3)}
    for pid, (comm, fetch) in spec.items():
        n = "app#hostA#%d#%d" % (pid, pid)
        rows += _ev(pid, n, "window", "window.stop", 10.0)
        rows += _ev(pid, n, "compute", "compute", 4.0)
        rows += _ev(pid, n, "comm", "all_reduce", comm / 10.0, n=10)
        rows += _ev(pid, n, "dataloader", "fetch.iter", fetch / 10.0, n=10)
        rows += _ev(pid, n, "stall", "stall.storage", fetch)
    views = plan.flat_views(pd.DataFrame(rows), control_pids=[10, 11, 12, 13])
    pv = views[("proc_name",)]
    for col in ("comm_all_reduce_count_frac_total", "comm_all_reduce_time_proc_frac_total",
                "comm_all_reduce_ops_slope", "fetch_iter_ops_slope"):
        assert col in pv.columns, col
    slopes = pv["comm_all_reduce_ops_slope"].astype(float)
    assert slopes.loc["app#hostA#10#10"] < 1.0 < slopes.loc["app#hostA#13#13"]
    assert float(pv["fetch_iter_ops_slope"].astype(float).loc["app#hostA#10#10"]) < 1.0
    win = views[("window",)].loc[7]
    assert win["domain_skew_coverage"] == 1.0


def test_declared_bucket_cap_ladder_survives_the_wire_and_defines_the_planner_levels():
    """train.py registers knob(values=[25,50,100,200,400]); without it the grid is six LINEAR levels (each a rebuild)."""
    from dfoptimizer.planner.clbo_planner import _levels_for
    from dfoptimizer.runtime.knob import knob, knob_def_from_dict, knob_def_from_wire, knob_def_to_wire
    d = knob(default=25, range=(25, 400), values=LADDER, type=int, cost_seconds=0.0)
    kd = knob_def_from_wire(knob_def_to_wire(knob_def_from_dict("stormer.bucket_cap_mb", d,
                                                                   target_function="stormer.make_loader")))
    assert _levels_for(kd, 25) == LADDER
    lin = knob_def_from_dict("stormer.bucket_cap_mb", knob(default=25, range=(25, 400), type=int))
    assert _levels_for(lin, 25) == [25, 100, 175, 250, 325, 400]        # the historical registration


def _jv_planner(monkeypatch):
    p = f.stormer_planner(monkeypatch, env={"DFOPTIMIZER_CBANDIT_JOINT_VERIFY": "1"})
    p._levels["bucket_cap_mb"] = list(LADDER)
    return p


def _feed(p, start, stop, cap, w, wall):
    for wi in range(start, stop + 1):
        p._series.append((wi, {"num_workers": w, "bucket_cap_mb": cap}, wall + 0.001 * (wi % 3)))
        for k, v in (("bucket_cap_mb", cap), ("num_workers", w)):
            p._samples.setdefault((k, v), __import__("collections").deque(maxlen=200)).append(wall)


def _set_cfg(p, cap, w):
    p.current_values["stormer.bucket_cap_mb"] = cap
    p.current_values["stormer.num_workers"] = w


def test_joint_verification_reverts_lifo_then_reverifies_the_earlier_move(monkeypatch):
    p = _jv_planner(monkeypatch)
    _feed(p, 1, 4, 25, 8, 5.0)                                     # the configuration both moves left
    _feed(p, 5, 7, 50, 8, 6.0)
    _feed(p, 8, 13, 50, 10, 6.5)                                   # jointly worse than the base
    _set_cfg(p, 50, 10)
    p._last_move_window.update({"bucket_cap_mb": 4, "num_workers": 7})
    p._move_from.update({"bucket_cap_mb": (25, 0.5, 4, 5.0), "num_workers": (8, 0.5, 7, 6.0)})
    assert p._unconfirmed_move(13, 100.0) == ("num_workers", 8)     # most recent move first
    # apply the revert as process_finding does, then windows at (50, 8) keep being worse than the base
    p._move_from.pop("num_workers"); _set_cfg(p, 50, 8); p._last_move_window["num_workers"] = 13
    _feed(p, 14, 19, 50, 8, 6.0)
    assert p._unconfirmed_move(19, 100.0) == ("bucket_cap_mb", 25)  # the earlier move re-verified alone, reverted


def test_joint_verification_confirms_a_jointly_better_pair_and_ignores_stale_arm_samples(monkeypatch):
    p = _jv_planner(monkeypatch)
    _feed(p, 1, 2, 50, 10, 1.0)                                    # STALE: this configuration ran (and was left) earlier
    _feed(p, 3, 6, 25, 8, 5.0)
    _feed(p, 7, 9, 50, 8, 6.0)
    _feed(p, 10, 15, 50, 10, 6.5)                                  # recent truth: worse than base
    _set_cfg(p, 50, 10)
    p._last_move_window.update({"bucket_cap_mb": 6, "num_workers": 9})
    p._move_from.update({"bucket_cap_mb": (25, 0.5, 6, 5.0), "num_workers": (8, 0.5, 9, 6.0)})
    # value-keyed arms would average in the stale 1.0 s windows and could accept; configuration-keyed must not
    assert p._unconfirmed_move(15, 100.0) == ("num_workers", 8)
    q = _jv_planner(monkeypatch)
    _feed(q, 3, 6, 25, 8, 5.0); _feed(q, 7, 9, 50, 8, 6.0); _feed(q, 10, 15, 50, 10, 3.0)
    _set_cfg(q, 50, 10)
    q._last_move_window.update({"bucket_cap_mb": 6, "num_workers": 9})
    q._move_from.update({"bucket_cap_mb": (25, 0.5, 6, 5.0), "num_workers": (8, 0.5, 9, 6.0)})
    assert q._unconfirmed_move(15, 100.0) is None and "bucket_cap_mb" not in q._move_from


def test_ack_adopts_the_observed_value_and_records_a_consensus_override(monkeypatch):
    log = h._DecisionLog()
    p = _jv_planner(monkeypatch)
    monkeypatch.setattr(h.cbandit_planner, "logger", log)
    p._cur_window = 5
    p._planned_cfg[6] = {"num_workers": 8, "bucket_cap_mb": 100}
    p.current_values["stormer.bucket_cap_mb"] = 100                 # what this node planned
    p.apply_ack(plan_id="x", knob_id="stormer.bucket_cap_mb", status="applied", old_value=25, new_value=50)
    assert p.current_values["stormer.bucket_cap_mb"] == 50 and p._planned_cfg[6]["bucket_cap_mb"] == 50
    assert any(e == "optimizer.cbandit.consensus_override" and k["observed"] == 50 for e, k in log.entries)
