"""Synthetic StorMer windows for the fact -> finding -> plan path (offline control-path fixtures only).

The numbers are SHAPES, not measurements: `comm_dominant` mimics the rep0 cap-25 regime (comm the largest
domain, no input stall) and `input_stalls` the cap-400 regime (comm relieved, periodic input stalls).
See liveflow_shm/STEP0-COMM-GATE-2026-09-25.md.  Nothing here is detection quality or performance evidence.
"""
import dataclasses as dc
import json
import os
from pathlib import Path

import pandas as pd

from dfoptimizer.planner import cbandit_planner
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.types import DiagnosisFindingMsg, KnobDef, KnobResponse

try:
    from . import test_stormer_comm_rule as c
    from . import test_coherent_fact_handoff as h
except ImportError:
    import test_stormer_comm_rule as c
    import test_coherent_fact_handoff as h

from dftracer.analyzer.fact_engine import FactEngine
from dfdiagnoser.diagnoser import Diagnoser

_RULES = c._RULE.parent
LOOP_RULE = _RULES / "liveflow-stormer-loop.yaml"

KNOB_MAX_WORKERS = 24          # loop worker ceiling from the 256-lane campaign budget (orchestrator 2026-09-26)


def loop_engine():
    return FactEngine.from_rule_file(str(LOOP_RULE), strict_time_semantics=True,
                                     allow_mixed_time_aggregates=False)


def stormer_planner(monkeypatch, *, workers=8, cap=25, allow=("num_workers", "bucket_cap_mb"),
                    priority=True, env=None):
    """cbandit with StorMer's registrations (train.py 2346-2440) restricted to the loop allowlist."""
    h._freeze_p3_cbandit(monkeypatch)
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_BARRIER_REDIRECT", "0")     # A1 OFF
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_STRICT_EXPLORE", "1")       # A2 ON
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_SKEW_LINK", "1")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_ACTIONABLE_PRIORITY", "1" if priority else "0")   # C2
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_COMM_NEEDS_SKEW", "1")        # loop config: comm needs measured skew
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, str(v))
    p = CoordinateBanditPlanner()
    defs = {
        "stormer.num_workers": KnobDef(
            id="stormer.num_workers", default=workers, type=int, range=(0, KNOB_MAX_WORKERS),
            target_function="stormer.make_loader",
            responds_to={"reader_parallelism": KnobResponse(
                direction="increase", step_mode="evidence", min_persistence=2, cooldown_windows=3,
                apply_when="window_boundary")}),
        "stormer.prefetch_factor": KnobDef(
            id="stormer.prefetch_factor", default=2, type=int, range=(1, 16),
            target_function="stormer.make_loader",
            responds_to={"dataloader_prefetch": KnobResponse(
                direction="increase", step_mode="evidence", min_persistence=2, cooldown_windows=3,
                apply_when="window_boundary")}),
        "stormer.bucket_cap_mb": KnobDef(
            id="stormer.bucket_cap_mb", default=cap, type=int, range=(25, 400),
            target_function="stormer.make_loader",
            responds_to={"comm_optimization": KnobResponse(
                direction="increase", step_mode="multiply", step=2, min_persistence=1,
                cooldown_windows=3, apply_when="window_boundary")}),
    }
    cur = {"stormer.num_workers": workers, "stormer.prefetch_factor": 2, "stormer.bucket_cap_mb": cap}
    if allow is not None:
        defs = {k: v for k, v in defs.items() if k.split(".")[-1] in allow}
        cur = {k: v for k, v in cur.items() if k in defs}
    p.register_knobs(defs, current_values=cur)
    return p


def window(w, *, comm, input_s, compute, A=0.0, E=None, W=None, cov=1.0, wall=None, slopes=None,
           fetch_slopes=None, remaining=400.0, wpe=1000.0, procs=4.0):
    """One window: the typical rank's domains and the skew attribution, as the analyzer publishes them."""
    E = A if E is None else E
    W = A if W is None else W
    wall = wall or (comm + input_s + compute)
    slopes = slopes or (0.6, 1.1, 1.1, 1.2)
    fetch_slopes = fetch_slopes or (0.7, 1.05, 1.1, 1.15)
    crit_in = input_s + E                      # the rank with the largest input exposure
    rows, rmap = c.rank_rows(slopes)
    rows["fetch_iter_count_frac_total"] = 0.25
    rows["fetch_iter_time_proc_frac_total"] = 0.25
    rows["fetch_iter_ops_slope"] = list(fetch_slopes)
    ctx = c.context(rmap, window=w, A=A, E=E, W=W, cov=cov, typ_in=input_s, typ_comm=comm,
                    typ_comp=compute, wall=wall)
    wm = ctx["window_metrics"]
    wm.update(c.coherent(window=w, comm=comm, input_s=crit_in, compute=compute, wall=wall, procs=procs))
    wm["residual_time_proc_crit"] = max(wall - crit_in - comm - compute, 0.0)
    ctx["window_metrics"] = wm
    ctx["_extra"] = {"active_time": wall, "windows_remaining": remaining, "windows_per_epoch": wpe,
                     "window": w}
    return rows, ctx


class _Ev:
    def __init__(self, payload):
        self.data = payload


def deliver(engine, diag, planner, rows, ctx):
    """rule -> analysis-fact envelope -> LiveDiagnoser -> control finding -> cbandit (production hops)."""
    extra = ctx.pop("_extra")
    facts = engine.evaluate({("proc_name",): rows}, raw_stats={"run_id": "stormer-loop"}, context=ctx)
    produced = facts.get(("proc_name",), [])
    plans, findings = [], []
    if produced:
        touched = diag._handle_analysis_facts(
            _Ev(json.dumps({"view_type": "proc_name",
                            "facts": [dc.asdict(f) for f in produced]}).encode("utf-8")), metadata={})
        findings = diag._build_control_findings(extra["window"], touched)
        for f in findings:
            msg = h._to_optimizer_finding(f)
            # the analyzer's control-window aggregates (horizon / wall) reach cbandit beside the fact
            msg.key_metrics = dict(msg.key_metrics or {}, **{k: v for k, v in extra.items() if k != "window"})
            plans.extend(planner.process_finding(msg))
    diag.state.advance_window()
    return produced, findings, plans
