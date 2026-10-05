"""A1/A2 (judge batch 1, 2026-09-25): barrier-redirect switch, decision_path, strict exploration."""
from pathlib import Path

import pytest

# Use THIS worktree's analyzer sources (same redirect as node_agent.bootstrap).
import dftracer
_AP = str(Path(__file__).resolve().parents[2] / "dfanalyzer/python/dftracer")
if _AP not in list(dftracer.__path__):
    try:
        dftracer.__path__.insert(0, _AP)
    except AttributeError:
        dftracer.__path__ = [_AP] + list(dftracer.__path__)
from dfoptimizer.types import KnobDef, KnobResponse
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner

try:
    from . import test_coherent_fact_handoff as h
except ImportError:
    import test_coherent_fact_handoff as h


def _decisions(log):
    return [k for e, k in log.entries if e == "optimizer.cbandit.decision"]


def _replay_r1(monkeypatch, *, redirect):
    """The retained r1 comm-dominant windows (input exposure > 0) through rule->diagnoser->cbandit."""
    h._freeze_p3_cbandit(monkeypatch)
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_BARRIER_REDIRECT", "1" if redirect else "0")
    log = h._DecisionLog()
    monkeypatch.setattr(h.cbandit_planner, "logger", log)
    engine, diag, planner = h._p3_engine(), h.Diagnoser(), h._p3_read_threads_planner()
    plans = []
    for rec in h._R1_FACT_WINDOWS:
        _f, _fi, new = h._deliver_p3_fact(engine, diag, planner, rec)
        plans += new
    return planner, _decisions(log), plans


def test_barrier_redirect_default_on_records_path(monkeypatch):
    planner, decisions, _ = _replay_r1(monkeypatch, redirect=True)
    assert planner._barrier_redirect is True
    assert decisions and all("decision_path" in d and "config_at_decision" in d and d["node"]
                             for d in decisions)
    assert all(d["decision_path"] == "barrier_redirect" for d in decisions)
    assert all(d["storage_suspect"] is True for d in decisions)


def test_barrier_redirect_off_comm_dominant_windows_cannot_reach_an_input_action(monkeypatch):
    planner, decisions, plans = _replay_r1(monkeypatch, redirect=False)
    assert planner._barrier_redirect is False
    assert plans == []
    assert decisions and all(d["decision"] == "decline" for d in decisions)
    assert all(d["decision_path"] == "decline_unattributed" and d["storage_suspect"] is False
               for d in decisions)


def _rt_planner(monkeypatch, *, strict, shipped, current=None, hi=40, lo=1):
    h._freeze_p3_cbandit(monkeypatch)
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_STRICT_EXPLORE", "1" if strict else "0")
    p = CoordinateBanditPlanner()
    kd = KnobDef(id="dlio.read_threads", default=shipped, type=int, range=(lo, hi),
                 cost_seconds=0.206, target_function="DLIOBenchmark.make_loader",
                 responds_to={"reader_parallelism": KnobResponse(
                     direction="increase", step_mode="evidence", min_persistence=1,
                     cooldown_windows=1, apply_when="window_boundary")})
    p.register_knobs({"dlio.read_threads": kd}, current_values={"dlio.read_threads": shipped})
    if current is not None:
        p.current_values["dlio.read_threads"] = current
    return p


def test_strict_explore_never_proposes_below_the_shipped_value(monkeypatch):
    p = _rt_planner(monkeypatch, strict=True, shipped=4)
    assert p._shipped["read_threads"] == 4
    choice, detail = p._pick("read_threads", 4, "input", exposed_s=0.5, remaining=30.0)
    assert all(d["level"] >= 4 for d in detail)      # lower neighbour removed from consideration
    assert choice >= 4


def test_default_keeps_historical_candidates(monkeypatch):
    p = _rt_planner(monkeypatch, strict=False, shipped=4)
    _c, detail = p._pick("read_threads", 4, "input", exposed_s=0.5, remaining=30.0)
    assert any(d["level"] < 4 for d in detail)


def test_strict_explore_exact_tie_between_untried_levels_is_no_prediction(monkeypatch):
    # shipped=1 so BOTH neighbours (1 is current here?) -> use current=4 with shipped=1: 1 and up-level are admissible and tie
    p = _rt_planner(monkeypatch, strict=True, shipped=1, current=4)
    choice, detail = p._pick("read_threads", 4, "input", exposed_s=0.5, remaining=30.0)
    scores = sorted(d["score"] for d in detail if d["level"] != 4)
    assert len(scores) == 2 and scores[0] == scores[1]      # the r5 shape: exact tie, n=0 for both
    assert choice == 4 and p._last_pick_note.startswith("tie_no_prediction")
    # the historical planner picks the lowest level of the tie
    q = _rt_planner(monkeypatch, strict=False, shipped=1, current=4)
    assert q._pick("read_threads", 4, "input", exposed_s=0.5, remaining=30.0)[0] < 4


def test_unconfirmed_move_revert_is_retained_as_regression_budget():
    assert hasattr(CoordinateBanditPlanner, "_unconfirmed_move")


# ---- measured skew link (replacement for the heuristic redirect) --------------------------------
from pathlib import Path
from dftracer.analyzer.fact_engine import FactEngine

_SKEW_RULE = (Path(__file__).resolve().parents[2]
              / "dfanalyzer/python/dftracer/analyzer/configs/fact_rules/liveflow-tf-input-skew-slope-exposure.yaml")
_REC = dict(h._R1_FACT_WINDOWS[3])                    # comm-dominant retained window shape


def _skew_metrics(A=20.0, E=20.0, W=22.0, cov=1.0, typ_in=0.3, typ_comm=25.0, typ_comp=11.8, wall=43.3):
    return {"domain_skew_coverage": cov, "domain_input_skew_attributed_time": A,
            "domain_input_skew_excess_time": E, "domain_comm_barrier_wait_time": W,
            "domain_skew_late_proc_input_time": 20.3, "domain_skew_late_proc_comm_time": 1.0,
            "domain_comm_residual_after_skew_time": typ_comm - A,
            "domain_input_time_proc_typ": typ_in, "domain_decode_time_proc_typ": 0.0,
            "domain_comm_time_proc_typ": typ_comm, "domain_compute_time_proc_typ": typ_comp,
            "window_wall_time_proc_typ": wall}


def test_skew_from_metrics_positive_and_suppressions():
    sk = CoordinateBanditPlanner._skew_from_metrics(_skew_metrics(), 43.3)
    assert sk["A"] == 20.0 and sk["domains"]["input"] == pytest.approx(20.3)
    assert sk["domains"]["comm"] == pytest.approx(5.0)
    # the link moves time between input and comm only: the typical rank's total is unchanged
    m = _skew_metrics()
    assert sum(sk["domains"].values()) == pytest.approx(
        m["domain_input_time_proc_typ"] + m["domain_comm_time_proc_typ"] + m["domain_compute_time_proc_typ"])
    for bad in (_skew_metrics(cov=0.0), _skew_metrics(A=0.0), _skew_metrics(A=-1.0),
                _skew_metrics(A=21.0, E=20.0, W=22.0),          # A must be min(E, W)
                _skew_metrics(A=26.0, E=30.0, W=30.0, typ_comm=25.0)):   # A cannot exceed comm
        assert CoordinateBanditPlanner._skew_from_metrics(bad, 43.3) is None
    partial = _skew_metrics(); del partial["domain_comm_time_proc_typ"]
    assert CoordinateBanditPlanner._skew_from_metrics(partial, 43.3) is None


def _skew_engine():
    return FactEngine.from_rule_file(str(_SKEW_RULE), strict_time_semantics=True,
                                     allow_mixed_time_aggregates=False)


def _skew_fact(**kw):
    rec = dict(_REC)
    ctx = h._p3_context(rec)
    ctx["completed_window_provenance"] = {"stall_storage_rank_coverage": 1.0}
    ctx["window_metrics"].update(_skew_metrics(**kw))
    rows = h._p3_rows(proc_name=rec["proc_name"], count_share=rec["count"], time_share=rec["time"],
                      slope=rec["slope"])
    return _skew_engine().evaluate({("proc_name",): rows}, raw_stats={"run_id": "skew"}, context=ctx)


def test_skew_rule_requires_coverage_one_and_positive_attribution():
    (fact,) = _skew_fact()[("proc_name",)]
    assert fact.evidence["metrics"]["domain_input_skew_attributed_time"] == 20.0
    assert _skew_fact(cov=0.0) == {} and _skew_fact(A=0.0, E=0.0) == {}


def test_skew_rule_uses_only_zero_and_one_constants():
    import re
    when = next(l for l in _SKEW_RULE.read_text().splitlines() if l.strip().startswith("when:"))
    assert set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", when)) <= {"0", "0.0", "1", "1.0"}


def _drive_skew(monkeypatch, *, skew_link, redirect=False, n=5, **kw):
    h._freeze_p3_cbandit(monkeypatch)
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_BARRIER_REDIRECT", "1" if redirect else "0")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_SKEW_LINK", "1" if skew_link else "0")
    log = h._DecisionLog()
    monkeypatch.setattr(h.cbandit_planner, "logger", log)
    diag, planner = h.Diagnoser(), h._p3_read_threads_planner()
    for w in range(1, n + 1):
        rec = dict(_REC, window=w)
        ctx = h._p3_context(rec)
        ctx["completed_window_provenance"] = {"stall_storage_rank_coverage": 1.0}
        ctx["window_metrics"].update(_skew_metrics(**kw))
        rows = h._p3_rows(proc_name=rec["proc_name"], count_share=rec["count"],
                          time_share=rec["time"], slope=rec["slope"])
        facts = _skew_engine().evaluate({("proc_name",): rows}, raw_stats={"run_id": "s"}, context=ctx)
        if not facts:
            continue
        produced = facts[("proc_name",)]
        touched = diag._handle_analysis_facts(
            h._Event(h.json.dumps({"view_type": "proc_name",
                                   "facts": [h.dc.asdict(f) for f in produced]}).encode()), metadata={})
        for f in diag._build_control_findings(w, touched):
            planner.process_finding(h._to_optimizer_finding(f))
    return planner, _decisions(log)


def test_skew_link_makes_input_dominant_and_records_the_path(monkeypatch):
    planner, decisions = _drive_skew(monkeypatch, skew_link=True)
    assert decisions
    assert all(d["dominant_domain"] == "input" for d in decisions)
    assert all(d["decision_path"].startswith("skew_attributed(A=20,E=20,W=22,L=20.3)") for d in decisions)
    assert all(d["storage_suspect"] is False for d in decisions)          # the heuristic did not fire
    # the input exposure the predicted effect sees is input_typ + A, i.e. bounded by the measured A
    assert all(v == pytest.approx(20.3) for v in planner._epoch_dom["input"])


def test_skew_link_off_comm_dominant_declines_without_redirect(monkeypatch):
    planner, decisions = _drive_skew(monkeypatch, skew_link=False)
    assert decisions and all(d["decision"] == "decline" for d in decisions)
    assert all(d["decision_path"] == "decline_unattributed" for d in decisions)


def test_small_attribution_leaves_comm_dominant_and_declines(monkeypatch):
    planner, decisions = _drive_skew(monkeypatch, skew_link=True, A=1.0, E=1.0, W=3.0)
    assert decisions and all(d["decision"] == "decline" for d in decisions)   # 25 - 1 comm still wins


def test_shipped_value_is_the_applications_start_not_the_bootstrap_default(monkeypatch):
    p = _rt_planner(monkeypatch, strict=True, shipped=1)        # optimizer env-bootstrap registry first
    assert p._shipped["read_threads"] == 1
    kd = p.knobs["dlio.read_threads"]
    p.register_knobs({"dlio.read_threads": kd}, current_values={"dlio.read_threads": 4})  # app's own
    assert p._shipped["read_threads"] == 4
    choice, detail = p._pick("read_threads", 4, "input", exposed_s=0.5, remaining=30.0)
    assert all(d["level"] >= 4 for d in detail)                 # level 1 is below the real Original
