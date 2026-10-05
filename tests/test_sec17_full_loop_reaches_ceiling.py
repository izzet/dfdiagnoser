"""Sec 17 v2 (full input+comm loop, orchestrator decision 2026-09-27): a synthetic rep0-style sequence,
through the SAME production hops as test_stormer_loop_path.py, but feeding the new job-wide sum-of-max
(sec 16) and median-based comm-exposure-v2 (sec 17) fields directly (as the analyzer would, once wired
to a real run) -- proving the coordinated loop can reach the campaign's actual ceiling (W~24, cap 400),
not just the older test's W=12 rep0 shape. Shapes are SHAPES, not measurements -- see
liveflow_shm/STORMER-LOOP-DESIGN-2026-09-26.md sec 17.
"""
import json
import dataclasses as dc

try:
    from . import stormer_loop_fixtures as f
    from . import test_stormer_comm_rule as c
    from . import test_coherent_fact_handoff as h
except ImportError:
    import stormer_loop_fixtures as f
    import test_stormer_comm_rule as c
    import test_coherent_fact_handoff as h

from dfoptimizer.planner import clbo_planner as clbo_planner_mod

LADDER = [25, 50, 100, 200, 400]
COMM = {25: 3.0, 50: 2.0, 100: 1.3, 200: 0.9, 400: 0.6}


def _rep0_to_24(w, cfg):
    """Same shape as test_stormer_loop_path._rep0, but the input stall needs W>=24 (the campaign
    ceiling, sec 15) to fully resolve, not W>=12 -- exercises the loop over its full registered range."""
    stall = (2.2 if cfg["W"] < 24 else 0.2) if cfg["cap"] >= 50 else 0.3
    return dict(comm=COMM[cfg["cap"]], input_s=stall, compute=2.4)


def _window_with_v2_fields(w, cfg, env_fn):
    """f.window(), then inject the sec 16/17 v2 fields an analyzer wired per this design would
    produce -- set from the SAME underlying values f.window() already computed, so this test proves
    the wiring reaches the identical, already-validated convergence, not a different one.

    `deliver()` merges `ctx["_extra"]` (not `window_metrics`, which only feeds the FactEngine rule
    evaluation) into the finding's `key_metrics` -- the dict `domains_from_metrics` actually reads --
    so the v2 fields belong there.
    """
    kw = env_fn(w, cfg)
    rows, ctx = f.window(w, **kw)
    wm = ctx["window_metrics"]
    extra = ctx["_extra"]
    # sec 16: job-wide sum-of-max input, standing in for what a real cross-node reduction would give;
    # >=2 distinct steps required by the suppression rule (sec 14), so use a plausible step count.
    extra["domain_input_job_step_max_sum_time"] = wm.get("domain_input_time_proc_crit")
    extra["domain_input_step_nunique"] = 10
    # sec 17 v2: median-based comm exposure standing in for domain2's per-step exposed-comm reduction.
    extra["domain_comm_pure_exposure_time_v2"] = wm.get("domain_comm_time_proc_crit")
    extra["domain_comm_exposure_step_nunique"] = 10
    ctx["_extra"] = extra
    return rows, ctx


def test_full_loop_with_v2_fields_reaches_the_campaign_ceiling(monkeypatch):
    log = h._DecisionLog()
    p = f.stormer_planner(monkeypatch, allow=("num_workers", "bucket_cap_mb"), priority=True,
                          env={"DFOPTIMIZER_CBANDIT_JOINT_VERIFY": "1"})
    monkeypatch.setattr(h.cbandit_planner, "logger", log)
    monkeypatch.setattr(clbo_planner_mod, "logger", log)
    p._levels["bucket_cap_mb"] = list(LADDER)
    eng, diag = f.loop_engine(), __import__("dfdiagnoser.diagnoser", fromlist=["Diagnoser"]).Diagnoser()
    cfg, trace, allplans = {"W": 8, "cap": 25}, [], []
    n = 80   # more windows than the W<12 rep0's n=40: reaching W=24 needs more escalation steps
    for w in range(1, n + 1):
        rows, ctx = _window_with_v2_fields(w, cfg, _rep0_to_24)
        _facts, _fnd, plans = f.deliver(eng, diag, p, rows, ctx)
        step = [(pl.knob_id.split(".")[-1], pl.old_value, pl.new_value) for pl in plans]
        trace.append((w, dict(cfg), step))
        allplans += step
        for pl in plans:
            if pl.knob_id.endswith("bucket_cap_mb"):
                cfg["cap"] = pl.new_value
            elif pl.knob_id.endswith("num_workers"):
                cfg["W"] = pl.new_value

    # The coordinated pair the design calls for: comm relief (cap climbs the ladder) and input relief
    # (workers climb) both happen, and neither ever reverts (JOINT_VERIFY keeps the pair, sec 1/17.5).
    assert cfg["cap"] == 400, trace
    assert cfg["W"] >= 24, trace
    assert not any(p[2] < p[1] for p in allplans if p[0] == "bucket_cap_mb"), allplans
    assert not any(p[2] < p[1] for p in allplans if p[0] == "num_workers"), allplans
    # Confirm the NEW sources were actually exercised (not silently falling back to the old crit path).
    input_srcs = set()
    comm_srcs = set()
    for e, d in log.entries:
        if e == "optimizer.domains.input_source":
            input_srcs.add(d.get("source"))
        if e == "optimizer.domains.comm_source":
            comm_srcs.add(d.get("source"))
    assert "job_sum_of_max" in input_srcs, input_srcs
    assert "comm_exposed_median_v2" in comm_srcs, comm_srcs
