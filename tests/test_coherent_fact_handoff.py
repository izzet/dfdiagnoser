"""Installed-component handoff test for coherent domain evidence.

This is a control-path fixture, not workload or performance evidence.  It uses
the FactEngine envelope shape consumed by Diagnoser, then hands the resulting
control finding to the existing CoordinateBanditPlanner without Mofka.
"""

import dataclasses as dc
import json

import pandas as pd
import pytest

from dfdiagnoser.diagnoser import Diagnoser
from dfoptimizer.planner import cbandit_planner
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.types import DiagnosisFindingMsg, KnobDef
from dftracer.analyzer.fact_engine import FactEngine
from dftracer.analyzer.fact_rules import build_fact_rule
from dftracer.analyzer.metrics import set_view_metrics


class _Event:
    def __init__(self, payload):
        self.data = payload


class _DecisionLog:
    def __init__(self):
        self.entries = []

    def info(self, event, **kwargs):
        self.entries.append((event, kwargs))

    def __getattr__(self, _name):
        return lambda *_args, **_kwargs: None


def _rule():
    # The operation-family slope is carried as evidence only.  Domain selection
    # remains the planner's coherent exposure argmax, not a threshold rule.
    return build_fact_rule({
        "id": "coherent.domain.evidence",
        "priority": 1,
        "source_view": "epoch",
        "fact_type": "coherent_domain_evidence",
        "emit_mode": "window",
        "required_metrics": [
            "active_time", "domain_coherent", "domain_procs_active",
            "window_wall_time_proc_crit", "domain_input_time_proc_crit",
            "domain_decode_time_proc_crit", "domain_comm_time_proc_crit",
            "domain_compute_time_proc_crit", "residual_time_proc_crit",
            "read_ops_slope",
        ],
        "when": "domain_coherent == 1.0",
        "severity_score": "0.8",
        "opportunity_tags": ["reader_parallelism"],
    })


def _frame(slope):
    return pd.DataFrame([{
        "active_time": 10.0,
        "domain_coherent": 1.0,
        "domain_procs_active": 2.0,
        "window_wall_time_proc_crit": 10.0,
        "domain_input_time_proc_crit": 8.0,
        "domain_decode_time_proc_crit": 1.0,
        "domain_comm_time_proc_crit": 0.5,
        "domain_compute_time_proc_crit": 0.5,
        # Negative means overlap; it is not an invalidity marker.
        "residual_time_proc_crit": -1.0,
        "read_ops_slope": slope,
    }])


def _planner():
    planner = CoordinateBanditPlanner()
    planner.register_knobs({
        "dlio.read_threads": KnobDef(
            id="dlio.read_threads", default=1, type=int, range=(1, 4),
            target_function="make_loader",
        ),
        "dlio.decode_threads": KnobDef(
            id="dlio.decode_threads", default=1, type=int, range=(1, 4),
            target_function="make_loader",
        ),
    })
    return planner


def _to_optimizer_finding(finding):
    return DiagnosisFindingMsg(
        finding_type=finding.finding_type,
        motif=finding.motif,
        severity=finding.severity,
        severity_score=finding.severity_score,
        confidence=finding.confidence,
        prevalence=finding.trend.prevalence,
        persistence=finding.trend.persistence,
        trend_direction=finding.trend.trend_direction,
        contributing_facts=finding.contributing_facts,
        recommendation_bundle=finding.recommendation_bundle,
        opportunity_tags=finding.opportunity_tags,
        summary=finding.summary,
        window_index=finding.trend.last_seen_window,
        key_metrics=finding.key_metrics,
    )


def _drive(slope, decision_log):
    engine = FactEngine([_rule()])
    diagnoser = Diagnoser()
    planner = _planner()
    plans = []

    for window in range(3):
        facts = engine.evaluate({("epoch",): _frame(slope)}, raw_stats={"run_id": "handoff"})
        fact = facts[("epoch",)][0]
        envelope = {"view_type": "epoch", "facts": [dc.asdict(fact)]}
        touched = diagnoser._handle_analysis_facts(
            _Event(json.dumps(envelope).encode("utf-8")), metadata={}
        )
        findings = diagnoser._build_control_findings(diagnoser.state.current_window, touched)
        assert len(findings) == 1
        assert findings[0].key_metrics["read_ops_slope"] == slope
        plans.extend(planner.process_finding(_to_optimizer_finding(findings[0])))
        diagnoser.state.advance_window()
    decisions = [kwargs for event, kwargs in decision_log.entries
                 if event == "optimizer.cbandit.decision"]
    return plans, planner, decisions


def test_coherent_fact_handoff_keeps_slope_evidence_out_of_domain_selection(monkeypatch):
    decision_log = _DecisionLog()
    monkeypatch.setattr(cbandit_planner, "logger", decision_log)
    low_slope_plans, low_slope_planner, low_decisions = _drive(0.25, decision_log)
    decision_log.entries.clear()
    high_slope_plans, high_slope_planner, high_decisions = _drive(4.0, decision_log)

    # The complete handoff carries the coherent input vector to the optimizer.
    # It deliberately does not invent a candidate effect from the slope: with no
    # arm evidence, both variants decline rather than actuate.
    assert list(low_slope_planner._epoch_dom["input"]) == [8.0, 8.0]
    assert list(high_slope_planner._epoch_dom["input"]) == [8.0, 8.0]
    assert low_slope_plans == []
    assert high_slope_plans == []
    for decisions in (low_decisions, high_decisions):
        assert decisions
        assert all(entry["decision"] == "stay" for entry in decisions)
        assert all(entry["dominant_domain"] == "input" for entry in decisions)
        assert all(entry["accept"] == "no_predicted_effect" for entry in decisions)


def test_wisio_slope_detects_within_family_and_hands_severity_to_livediagnoser(monkeypatch):
    # The analyzer first normalizes only the read operation family.  This is a
    # controlled gate: it proves the 45-degree reference reaches the online
    # fact/diagnosis/optimizer handoff, not a workload or performance claim.
    slopes = set_view_metrics(
        pd.DataFrame({
            "read_count_sum": [1.0, 9.0],
            "read_size_sum": [1.0, 9.0],
            "read_time_sum": [9.0, 1.0],
        }),
        metric_boundaries={},
        is_view_process_based=True,
    )
    assert slopes.loc[0, "read_ops_slope"] == pytest.approx(1.0 / 9.0)
    assert slopes.loc[1, "read_ops_slope"] == pytest.approx(9.0)

    rule = build_fact_rule({
        "id": "wisio.read.slope.reference45",
        "priority": 1,
        "source_view": "epoch",
        "fact_type": "operation_family_time_disproportion",
        "emit_mode": "window",
        "required_metrics": [
            "active_time", "domain_coherent", "domain_procs_active",
            "window_wall_time_proc_crit", "domain_input_time_proc_crit",
            "domain_decode_time_proc_crit", "domain_comm_time_proc_crit",
            "domain_compute_time_proc_crit", "residual_time_proc_crit",
            "read_ops_slope",
        ],
        # This is the geometric 45-degree reference (1.0), not a tuned
        # workload threshold.  LiveDiagnoser owns persistence and LiveOptimizer
        # still requires a predicted effect before it may actuate.
        "derived_metrics": {
            "read_ops_slope_ref45_deficit": "clip01(1.0 - read_ops_slope)",
        },
        "when": "domain_coherent == 1.0 and read_ops_slope_ref45_deficit > 0.0",
        "severity_score": "read_ops_slope_ref45_deficit",
        "opportunity_tags": ["reader_parallelism"],
    })

    def frame(slope):
        base = _frame(0.0)
        base["read_ops_slope"] = slope
        return base

    engine = FactEngine([rule])
    # The peer above the reference never enters diagnosis.
    assert engine.evaluate({("epoch",): frame(slopes.loc[1, "read_ops_slope"])}, raw_stats={"run_id": "slope"}).get(("epoch",), []) == []
    # Equality is the geometric reference, not an issue.  A zero-time family
    # has no valid normalized time share and must also be suppressed rather
    # than coerced into a severity.
    equal = set_view_metrics(
        pd.DataFrame({
            "read_count_sum": [1.0, 1.0],
            "read_size_sum": [1.0, 1.0],
            "read_time_sum": [1.0, 1.0],
        }), metric_boundaries={}, is_view_process_based=True,
    )
    assert equal.loc[0, "read_ops_slope"] == pytest.approx(1.0)
    assert engine.evaluate({("epoch",): frame(equal.loc[0, "read_ops_slope"])}, raw_stats={"run_id": "slope"}).get(("epoch",), []) == []
    zero_time = set_view_metrics(
        pd.DataFrame({
            "read_count_sum": [1.0, 1.0],
            "read_size_sum": [1.0, 1.0],
            "read_time_sum": [0.0, 0.0],
        }), metric_boundaries={}, is_view_process_based=True,
    )
    assert pd.isna(zero_time.loc[0, "read_ops_slope"])
    assert engine.evaluate({("epoch",): frame(zero_time.loc[0, "read_ops_slope"])}, raw_stats={"run_id": "slope"}).get(("epoch",), []) == []

    decision_log = _DecisionLog()
    monkeypatch.setattr(cbandit_planner, "logger", decision_log)
    diagnoser = Diagnoser()
    planner = _planner()
    for _window in range(3):
        facts = engine.evaluate({("epoch",): frame(slopes.loc[0, "read_ops_slope"])}, raw_stats={"run_id": "slope"})
        assert len(facts[("epoch",)]) == 1
        fact = facts[("epoch",)][0]
        assert fact.severity.score == pytest.approx(8.0 / 9.0)
        assert fact.evidence["metrics"]["read_ops_slope_ref45_deficit"] == pytest.approx(8.0 / 9.0)
        touched = diagnoser._handle_analysis_facts(
            _Event(json.dumps({"view_type": "epoch", "facts": [dc.asdict(fact)]}).encode("utf-8")), metadata={}
        )
        finding = diagnoser._build_control_findings(diagnoser.state.current_window, touched)[0]
        assert finding.severity_score == pytest.approx(8.0 / 9.0)
        assert finding.key_metrics["read_ops_slope"] == pytest.approx(1.0 / 9.0)
        # Slope detects within the family; the coherent vector still selects
        # input and the optimizer remains safe without predicted arm effects.
        assert planner.process_finding(_to_optimizer_finding(finding)) == []
        diagnoser.state.advance_window()

    decisions = [kwargs for event, kwargs in decision_log.entries if event == "optimizer.cbandit.decision"]
    assert decisions and all(entry["dominant_domain"] == "input" for entry in decisions)
    assert all(entry["accept"] == "no_predicted_effect" for entry in decisions)
