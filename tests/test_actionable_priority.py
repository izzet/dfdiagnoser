"""Actionable-domain prioritization (orchestrator 2026-09-25): an unactionable dominant domain never vetoes."""
from pathlib import Path

import pytest

import dftracer
_AP = str(Path(__file__).resolve().parents[2] / "dfanalyzer/python/dftracer")
if _AP not in list(dftracer.__path__):
    try:
        dftracer.__path__.insert(0, _AP)
    except AttributeError:
        dftracer.__path__ = [_AP] + list(dftracer.__path__)

from dfoptimizer.types import DiagnosisFindingMsg

try:
    from . import test_coherent_fact_handoff as h
    from . import test_cbandit_a1_a2 as t
except ImportError:
    import test_coherent_fact_handoff as h
    import test_cbandit_a1_a2 as t


def _run(monkeypatch, *, priority, metrics_for, n=6):
    h._freeze_p3_cbandit(monkeypatch)
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_BARRIER_REDIRECT", "0")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_SKEW_LINK", "1")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_STRICT_EXPLORE", "1")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_ACTIONABLE_PRIORITY", "1" if priority else "0")
    log = h._DecisionLog()
    monkeypatch.setattr(h.cbandit_planner, "logger", log)
    planner = h._p3_read_threads_planner()
    plans = []
    for w in range(1, n + 1):
        km = metrics_for(w)
        f = DiagnosisFindingMsg(
            finding_type="operation_family_time_disproportion", motif="m", severity="high",
            severity_score=0.5, confidence=1.0, prevalence=1.0, persistence=1.0,
            trend_direction="stable", contributing_facts=[], recommendation_bundle=[],
            opportunity_tags=["reader_parallelism"], summary="s", window_index=w, key_metrics=km)
        plans += planner.process_finding(f)
    return planner, [k for e, k in log.entries if e == "optimizer.cbandit.decision"], plans


def _km(input_crit=0.4, comm=0.4, compute=4.9, decode=0.0, A=None, wall=5.6, typ_in=None):
    km = {"domain_coherent": 1.0, "domain_procs_active": 64.0, "window_wall_time_proc_crit": wall,
          "domain_input_time_proc_crit": input_crit, "domain_decode_time_proc_crit": decode,
          "domain_comm_time_proc_crit": comm, "domain_compute_time_proc_crit": compute,
          "residual_time_proc_crit": wall - input_crit - comm - compute - decode,
          "active_time": wall, "windows_remaining": 40.0, "windows_per_epoch": 24.0}
    if A is not None:
        km.update(t._skew_metrics(A=A, E=A, W=A + 0.1, typ_in=typ_in if typ_in is not None else input_crit,
                                  typ_comm=comm + A, typ_comp=compute, wall=wall))
    return km


def test_compute_dominant_unactionable_does_not_veto_exposed_input(monkeypatch):
    _p, dec, _pl = _run(monkeypatch, priority=True, metrics_for=lambda w: _km(A=0.2))
    assert dec and all(d["dominant_domain"] == "input" for d in dec)
    assert all(d["decision_path"].startswith("actionable_nondominant(input,exposure=") and
               "dominant=compute" in d["decision_path"] for d in dec)
    # the existing gates decide from here (stay/move with their own accept vocabulary), never the decline
    assert all(d["decision"] in ("stay", "move") and d.get("accept") for d in dec)


def test_default_still_declines_when_the_dominant_domain_is_unactionable(monkeypatch):
    _p, dec, plans = _run(monkeypatch, priority=False, metrics_for=lambda w: _km(A=0.2))
    assert dec and all(d["decision"] == "decline" for d in dec) and plans == []


def test_zero_input_exposure_declines_even_with_priority_on(monkeypatch):
    _p, dec, plans = _run(monkeypatch, priority=True,
                          metrics_for=lambda w: _km(input_crit=0.0, A=None))
    assert dec and all(d["decision"] == "decline" for d in dec) and plans == []


def test_tiny_exposure_is_gated_by_predicted_effect_not_by_dominance(monkeypatch):
    _p, dec, plans = _run(monkeypatch, priority=True,
                          metrics_for=lambda w: _km(input_crit=0.01, comm=0.4, compute=5.19 + (1.0 if w % 2 else -1.0),
                                    wall=5.6 + (1.0 if w % 2 else -1.0), A=None), n=8)
    assert dec and all(d["decision"] == "stay" for d in dec) and plans == []
    assert {d["accept"] for d in dec} <= {"no_predicted_effect", "unverifiable_in_horizon",
                                          "incumbent_under_sampled", "untried_neighbour:20"} | {
        d["accept"] for d in dec}


def test_no_input_fact_no_decision(monkeypatch):
    # the planner only sees windows whose rule fired; coverage 0 => the skew rule emits nothing
    assert t._skew_fact(cov=0.0) == {}


def test_strict_explore_floor_still_applies_under_priority(monkeypatch):
    p, dec, plans = _run(monkeypatch, priority=True, metrics_for=lambda w: _km(A=0.2), n=10)
    for pl in plans:
        for a in getattr(pl, "actions", []):
            assert getattr(a, "new_value", 99) >= 4
    assert p._shipped["read_threads"] == 4


def test_residual_dominant_window_still_declines_under_priority(monkeypatch):
    """C2a: unattributed residual >= the actionable exposure => decline, even with priority ON."""
    def km(w):
        # wall 10 s of which only 0.5 s is named (input 0.3, comm 0.1, compute 0.1): residual 9.5 s dominates
        return _km(input_crit=0.3, comm=0.1, compute=0.1, wall=10.0)
    p, dec, plans = _run(monkeypatch, priority=True, metrics_for=km)
    assert dec and all(d["decision"] == "decline" for d in dec) and plans == []
    assert all(d["decision_path"] == "decline_unattributed" for d in dec)


def test_named_time_dominant_window_is_still_prioritized(monkeypatch):
    """The guard does not disable the mechanism: residual small, compute dominant, input exposed."""
    p, dec, plans = _run(monkeypatch, priority=True, metrics_for=lambda w: _km(A=0.2))
    assert dec and all(d["decision_path"].startswith("actionable_nondominant") for d in dec)
