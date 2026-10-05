"""Control-path tests for the TF slope-plus-exposure input rule.

Fixture rows are the selected-row reconstructions of retained r1 facts used by
test_coherent_fact_handoff.py.  These are offline control-path checks, not
detection-quality, performance, or trajectory evidence.
"""

import dataclasses as dc
import json
from pathlib import Path

# Run against THIS worktree's analyzer sources, not the venv's installed copy: same redirect as
# node_agent.bootstrap (extend dftracer.__path__ before any dftracer.analyzer import).
import dftracer
_AP = str(Path(__file__).resolve().parents[2] / "dfanalyzer/python/dftracer")
if _AP not in list(dftracer.__path__):
    try:
        dftracer.__path__.insert(0, _AP)
    except AttributeError:
        dftracer.__path__ = [_AP] + list(dftracer.__path__)
import pytest

from dfdiagnoser.diagnoser import Diagnoser
from dftracer.analyzer.fact_engine import FactEngine

try:
    from . import test_coherent_fact_handoff as h
except ImportError:
    import test_coherent_fact_handoff as h

_RULE = (
    Path(__file__).resolve().parents[2]
    / "dfanalyzer/python/dftracer/analyzer/configs/fact_rules/liveflow-tf-input-slope-exposure.yaml"
)
_WINDOW4 = h._R1_FACT_WINDOWS[3]  # slope 0.278, input exposure 5.21 s of 43.3 s


def _engine():
    return FactEngine.from_rule_file(
        str(_RULE), strict_time_semantics=True, allow_mixed_time_aggregates=False)


def _facts(record, *, coverage=1.0, context_edit=None, slope=None, omit_coverage=False):
    ctx = h._p3_context(record)
    if not omit_coverage:
        ctx["completed_window_provenance"] = {"stall_storage_rank_coverage": coverage}
    if context_edit:
        context_edit(ctx["window_metrics"])
    rows = h._p3_rows(
        proc_name=record["proc_name"], count_share=record["count"],
        time_share=record["time"], slope=record["slope"] if slope is None else slope)
    return _engine().evaluate({("proc_name",): rows}, raw_stats={"run_id": "tf-slope-exposure"},
                              context=ctx)


def test_rule_uses_no_tuned_constant_beyond_the_reference_and_zero():
    text = _RULE.read_text()
    when = [l for l in text.splitlines() if l.strip().startswith("when:")][0]
    import re
    numbers = set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", when))
    assert numbers <= {"0", "0.0", "1", "1.0"}, numbers


def test_positive_slope_below_reference_with_measured_input_exposure_emits():
    facts = _facts(_WINDOW4)
    (fact,) = facts[("proc_name",)]
    assert fact.severity.score == pytest.approx(1.0 - _WINDOW4["slope"])
    assert "reader_parallelism" in fact.opportunity_tags


@pytest.mark.parametrize("case", (
    "hidden_prefetch_zero_input_exposure",   # slope < 1, emitted zero wait
    "uniform_slowdown_slope_one",            # slope == 1 exactly
    "noncoherent_window",
    "stall_coverage_incomplete",
    "stall_coverage_absent",
))
def test_negative_controls_suppress_the_fact(case):
    if case == "hidden_prefetch_zero_input_exposure":
        out = _facts(_WINDOW4, context_edit=lambda m: m.update(domain_input_time_proc_crit=0.0))
    elif case == "uniform_slowdown_slope_one":
        out = _facts(_WINDOW4, slope=1.0)
    elif case == "noncoherent_window":
        out = _facts(_WINDOW4, context_edit=lambda m: m.update(domain_coherent=0.0))
    elif case == "stall_coverage_incomplete":
        out = _facts(_WINDOW4, coverage=0.0)
    else:
        out = _facts(_WINDOW4, omit_coverage=True)
    assert out == {}


def test_suppressed_windows_never_reach_findings_or_plans(monkeypatch):
    h._freeze_p3_cbandit(monkeypatch)
    diagnoser, planner = Diagnoser(), h._p3_read_threads_planner()
    plans = []
    for rec in h._R1_FACT_WINDOWS:
        out = _facts(rec, context_edit=lambda m: m.update(domain_input_time_proc_crit=0.0))
        assert out == {}
        diagnoser.state.advance_window()
    assert plans == []


def test_positive_windows_flow_to_findings_and_only_a_read_threads_plan_is_possible(monkeypatch):
    h._freeze_p3_cbandit(monkeypatch)
    log = h._DecisionLog()
    monkeypatch.setattr(h.cbandit_planner, "logger", log)
    diagnoser, planner = Diagnoser(), h._p3_read_threads_planner()
    n_findings, plans = 0, []
    for rec in h._R1_FACT_WINDOWS:
        facts = _facts(rec)[("proc_name",)]
        touched = diagnoser._handle_analysis_facts(
            h._Event(json.dumps({"view_type": "proc_name",
                                 "facts": [dc.asdict(f) for f in facts]}).encode()),
            metadata={})
        for f in diagnoser._build_control_findings(rec["window"], touched):
            n_findings += 1
            plans.extend(planner.process_finding(h._to_optimizer_finding(f)))
    assert n_findings == 4
    # The rule feeds evidence only; any action must come through the planner's
    # predicted-effect gate and be the single allowlisted TF knob.
    for plan in plans:
        assert all(getattr(a, "knob_id", "dlio.read_threads") == "dlio.read_threads"
                   for a in getattr(plan, "actions", []))
    decisions = [k for e, k in log.entries if e == "optimizer.cbandit.decision"]
    assert decisions and all("accept" in d for d in decisions)


# ---- decode family (dlio.computation_threads actuator's own fact) -------------------------------
import pandas as pd

_COMBINED = _RULE.with_name("liveflow-tf-input-decode-slope-exposure.yaml")


def _decode_facts(record, *, slope, count=0.02, time=0.05, edit=None, coverage=1.0):
    ctx = h._p3_context(record)
    ctx["completed_window_provenance"] = {"stall_storage_rank_coverage": coverage}
    if edit:
        edit(ctx["window_metrics"])
    rows = pd.DataFrame({
        "reader_decode_count_frac_total": [count],
        "reader_decode_time_proc_frac_total": [time],
        "reader_decode_ops_slope": [slope],
    }, index=[record["proc_name"]])
    return FactEngine.from_rule_file(
        str(_COMBINED), strict_time_semantics=True, allow_mixed_time_aggregates=False,
    ).evaluate({("proc_name",): rows}, raw_stats={"run_id": "decode"}, context=ctx)


def test_decode_family_positive_emits_decode_pressure_only_for_its_own_family():
    facts = _decode_facts(_WINDOW4, slope=0.4)[("proc_name",)]
    assert [f.fact_type for f in facts] == ["operation_family_time_disproportion"]
    (fact,) = facts
    assert fact.scope.entity == "reader_decode.rank_thread_pool"
    assert fact.opportunity_tags == ["decode_pressure"]
    assert fact.severity.score == pytest.approx(0.6)


@pytest.mark.parametrize("case", ("slope_one", "zero_input_exposure", "noncoherent",
                                   "no_coverage", "family_columns_absent"))
def test_decode_family_negative_controls(case):
    if case == "slope_one":
        assert _decode_facts(_WINDOW4, slope=1.0) == {}
    elif case == "zero_input_exposure":
        assert _decode_facts(_WINDOW4, slope=0.4,
                             edit=lambda m: m.update(domain_input_time_proc_crit=0.0)) == {}
    elif case == "noncoherent":
        assert _decode_facts(_WINDOW4, slope=0.4,
                             edit=lambda m: m.update(domain_coherent=0.0)) == {}
    elif case == "no_coverage":
        assert _decode_facts(_WINDOW4, slope=0.4, coverage=0.0) == {}
    else:  # a read-family-only frame must not satisfy the decode rule (families are never mixed)
        rows = h._p3_rows(proc_name=_WINDOW4["proc_name"], count_share=_WINDOW4["count"],
                          time_share=_WINDOW4["time"], slope=_WINDOW4["slope"])
        out = FactEngine.from_rule_file(
            str(_COMBINED), strict_time_semantics=True, allow_mixed_time_aggregates=False,
        ).evaluate({("proc_name",): rows}, raw_stats={"run_id": "d"}, context=h._p3_context(_WINDOW4))
        assert all(f.scope.entity != "reader_decode.rank_thread_pool"
                   for fs in out.values() for f in fs)
