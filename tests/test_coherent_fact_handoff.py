"""Installed-component handoff test for coherent domain evidence.

This is a control-path fixture, not workload or performance evidence.  It uses
the FactEngine envelope shape consumed by Diagnoser, then hands the resulting
control finding to the existing CoordinateBanditPlanner without Mofka.
"""

import dataclasses as dc
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from dfdiagnoser.diagnoser import Diagnoser
from dfoptimizer.planner import cbandit_planner
from dfoptimizer.planner.cbandit_planner import CoordinateBanditPlanner
from dfoptimizer.types import DiagnosisFindingMsg, KnobDef, KnobResponse
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


_P3_RULE = (
    Path(__file__).resolve().parents[2]
    / "dfanalyzer/python/dftracer/analyzer/configs/fact_rules/p3-tf-rank-thread-slope.yaml"
)

# Inline excerpts from the retained real-TF P3 detector facts.  The test does
# not read the artifact at runtime, but makes its provenance reviewable:
# /p/vast1/izzet/lfmr/p3-tf-detect-20260922-r1/nodes/corona188/agent/
# analysis_facts_window_000001.json
# sha256=2bdd8561645dcf1052b42fbc2e366199aa0b9f77ed3162fe6783266fa55ede69
# analysis_facts_window_000002.json
# sha256=3e7cb16b21a23b589e402a88ebbb822256c309420c29daff074e17f28084df5f
# analysis_facts_window_000003.json
# sha256=5dd4ad5c80a09505b0d30543a94c5bfbc2fc991a1aeb804877debf51f5f7164f
# analysis_facts_window_000004.json
# sha256=89c937aa3431f1fceee0d6b4ea1ee809f9305a9554c7332e29f5277ea881d142
# This is a selected-row reconstruction of retained facts, not the full
# analyzer population nor a reconstruction of its selected-row minimum.
_R1_RANK_PIDS = [119120, 119121, 119122, 119123, 119124, 119125, 119126, 119127]
_R1_FACT_WINDOWS = (
    {"window": 1, "proc_name": "app#corona188#119125#122215", "pid": 119125,
     "tid": 122215, "count": 0.0003534023003277003, "time": 0.004732097148886726,
     "slope": 0.07468196218474547, "procs": 6.0, "wall": 62.611048,
     "input": 0.877183, "decode": 0.0, "comm": 49.46207700000003,
     "compute": 11.796605999999969, "residual": 0.47518199999999666,
     "step_min": 1.0, "step_max": 1228.0, "step_nunique": 1228.0},
    {"window": 2, "proc_name": "app#corona188#119121#122170", "pid": 119121,
     "tid": 122170, "count": 0.02043310219392975, "time": 0.0465234503508002,
     "slope": 0.4392000601816563, "procs": 8.0, "wall": 54.477448,
     "input": 1.145712, "decode": 0.0, "comm": 41.188723999999986,
     "compute": 11.770692999999952, "residual": 0.37231900000006135,
     "step_min": 1.0, "step_max": 2456.0, "step_nunique": 1229.0},
    {"window": 3, "proc_name": "app#corona188#119127#122138", "pid": 119127,
     "tid": 122138, "count": 0.03043972706595906, "time": 0.05159610491049709,
     "slope": 0.5899617251876348, "procs": 8.0, "wall": 48.409131,
     "input": 0.286195, "decode": 0.0, "comm": 35.92738700000002,
     "compute": 11.797533999999963, "residual": 0.3980150000000222,
     "step_min": 1.0, "step_max": 3684.0, "step_nunique": 1229.0},
    {"window": 4, "proc_name": "app#corona188#119120#122141", "pid": 119120,
     "tid": 122141, "count": 0.02048339682209598, "time": 0.07370455649985866,
     "slope": 0.27791221865822174, "procs": 8.0, "wall": 43.311811,
     "input": 5.210112, "decode": 0.0, "comm": 25.936255,
     "compute": 11.778631999999956, "residual": 0.3868120000000417,
     "step_min": 1.0, "step_max": 4912.0, "step_nunique": 1229.0},
)


def _p3_engine():
    """Load the parked TF rule itself, rather than restating it in this test."""
    return FactEngine.from_rule_file(
        str(_P3_RULE), strict_time_semantics=True,
        allow_mixed_time_aggregates=False,
    )


def _p3_rows(*, proc_name, count_share, time_share, slope):
    return pd.DataFrame({
        "reader_posix_read_count_frac_total": [count_share],
        "reader_posix_read_time_proc_frac_total": [time_share],
        "reader_posix_read_ops_slope": [slope],
    }, index=[proc_name])


def _p3_context(record, **overrides):
    # This is the analyzer-owned completed-window context, not a CSV-derived
    # substitute.  The numeric PID map independently confirms field three of
    # the display identity; field four remains only the thread/span identity.
    context = {
        "native_window": record["window"],
        "native_host": "corona188",
        "rank_pids": _R1_RANK_PIDS,
        "row_pid_by_index": {record["proc_name"]: record["pid"]},
        "window_metrics": {
            "domain_coherent": 1.0,
            "domain_procs_active": record["procs"],
            "window_wall_time_proc_crit": record["wall"],
            "domain_input_time_proc_crit": record["input"],
            "domain_decode_time_proc_crit": record["decode"],
            "domain_comm_time_proc_crit": record["comm"],
            "domain_compute_time_proc_crit": record["compute"],
            "residual_time_proc_crit": record["residual"],
            "epoch_nunique": 1.0,
            "step_nunique": record["step_nunique"],
            "step_min": record["step_min"],
            "step_max": record["step_max"],
        },
    }
    context.update(overrides)
    return context


def _p3_read_threads_planner():
    """The production read_threads registration relevant to this fact tag."""
    minimum = int(os.environ.get("DLIO_KNOB_MIN_THREADS", "0"))
    maximum = int(os.environ.get("DLIO_KNOB_MAX_THREADS", str(os.cpu_count() or 8)))
    minimum = min(minimum, maximum)
    planner = CoordinateBanditPlanner()
    planner.register_knobs({
        "dlio.prefetch_size": KnobDef(
            id="dlio.prefetch_size", default=2, type=int, range=(1, 16),
            cost_seconds=0.206, target_function="DLIOBenchmark.make_loader",
            responds_to={"dataloader_prefetch": KnobResponse(
                direction="increase", step=2, min_persistence=1,
                cooldown_windows=2, apply_when="window_boundary",
            )},
        ),
        "dlio.read_threads": KnobDef(
            id="dlio.read_threads", default=max(0, minimum), type=int,
            range=(minimum, maximum),
            cost_seconds=0.206, target_function="DLIOBenchmark.make_loader",
            responds_to={"reader_parallelism": KnobResponse(
                direction="increase", step_mode="evidence", min_persistence=1,
                cooldown_windows=1, apply_when="window_boundary",
            )},
        ),
    }, current_values={"dlio.read_threads": 4, "dlio.prefetch_size": 2})
    return planner


def _deliver_p3_fact(engine, diagnoser, planner, record, *, context=None):
    facts = engine.evaluate(
        {("proc_name",): _p3_rows(
            proc_name=record["proc_name"], count_share=record["count"],
            time_share=record["time"], slope=record["slope"],
        )}, raw_stats={"run_id": "p3-composed"},
        context=_p3_context(record) if context is None else context,
    )
    if not facts:
        return [], [], []
    produced = facts[("proc_name",)]
    touched = diagnoser._handle_analysis_facts(
        _Event(json.dumps({"view_type": "proc_name", "facts": [dc.asdict(f) for f in produced]}).encode("utf-8")),
        metadata={},
    )
    findings = diagnoser._build_control_findings(record["window"], touched)
    plans = []
    for finding in findings:
        plans.extend(planner.process_finding(_to_optimizer_finding(finding)))
    return produced, findings, plans


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


def test_synthetic_input_dominant_handoff_keeps_slope_evidence_out_of_domain_selection(monkeypatch):
    """Synthetic control-path coverage; retained r1 replay is tested separately."""
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


def _freeze_p3_cbandit(monkeypatch):
    # Freeze the ordinary controller, capacity, and DLIO registration inputs.
    # The upper bound is this runner's resolved CPU count, never an artificial
    # four-thread cap; the production default takes the same os.cpu_count path.
    cpu_max = os.cpu_count() or 8
    monkeypatch.setenv("DLIO_KNOB_MIN_THREADS", "0")
    monkeypatch.setenv("DLIO_KNOB_MAX_THREADS", str(cpu_max))
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_ACCEPT", "1")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_PREDICTED_EFFECT", "1")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_ALWAYS_ACT", "0")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_MIN_SAMPLES", "3")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_MIN_DWELL", "3")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_DOMAIN_BLIND", "0")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_ESCALATE", "1")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_C", "0.35")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_COST_W", "1.0")
    monkeypatch.setenv("DFOPTIMIZER_CBANDIT_HORIZON_WEIGHT", "1")
    monkeypatch.setenv("DFOPTIMIZER_CAPACITY_KNOB", "")
    monkeypatch.setenv("DFOPTIMIZER_CAPACITY_PROCS_PER_NODE", "")
    monkeypatch.setenv("DFOPTIMIZER_CAPACITY_RANKS_PER_NODE", "")
    monkeypatch.setenv("DFDIAGNOSER_METRICS_FROM", "current")


@pytest.mark.parametrize("case", ("zero_normalization", "nonfinite_normalization",
                                   "wrong_host", "mapped_pid_mismatch",
                                   "non_rank_pid", "balanced_reference"))
def test_p3_tf_rule_suppresses_invalid_or_ineligible_selected_rows(monkeypatch, case):
    """Each malformed/no-eligible case is suppressed by the fact rule itself."""
    _freeze_p3_cbandit(monkeypatch)
    engine = _p3_engine()
    record = dict(_R1_FACT_WINDOWS[0])
    context = _p3_context(record)
    if case == "zero_normalization":
        record.update(count=0.0, slope=0.0)
    elif case == "nonfinite_normalization":
        record.update(time=float("nan"), slope=float("nan"))
    elif case == "wrong_host":
        record["proc_name"] = record["proc_name"].replace("corona188", "corona999")
        context["row_pid_by_index"] = {record["proc_name"]: record["pid"]}
    elif case == "mapped_pid_mismatch":
        context["row_pid_by_index"] = {record["proc_name"]: 999999}
    elif case == "non_rank_pid":
        record["proc_name"] = "app#corona188#999999#122215"
        record["pid"] = 999999
        context["row_pid_by_index"] = {record["proc_name"]: 999999}
    else:
        record.update(count=0.5, time=0.5, slope=1.0)

    facts = engine.evaluate(
        {("proc_name",): _p3_rows(
            proc_name=record["proc_name"], count_share=record["count"],
            time_share=record["time"], slope=record["slope"],
        )}, raw_stats={"run_id": "p3-suppression"}, context=context,
    )
    assert facts == {}


def test_p3_tf_r1_excerpt_preserves_comm_dominant_context_and_records_decisions(monkeypatch):
    """Replay retained r1 fact excerpts; it is not an input-dominant synthetic case."""
    _freeze_p3_cbandit(monkeypatch)
    decision_log = _DecisionLog()
    monkeypatch.setattr(cbandit_planner, "logger", decision_log)
    engine, diagnoser, planner = _p3_engine(), Diagnoser(), _p3_read_threads_planner()
    assert planner.knobs["dlio.read_threads"].range[0] <= 4 < planner.knobs["dlio.read_threads"].range[1]
    plans = []

    for record in _R1_FACT_WINDOWS:
        facts, findings, new_plans = _deliver_p3_fact(engine, diagnoser, planner, record)
        assert len(facts) == len(findings) == 1
        metrics = findings[0].key_metrics
        assert facts[0].window.window == record["window"]
        assert metrics["selected_rank_pid"] == record["pid"]
        assert metrics["selected_thread_id"] == record["tid"]
        assert metrics["domain_comm_time_proc_crit"] == pytest.approx(record["comm"])
        assert metrics["domain_comm_time_proc_crit"] > metrics["domain_input_time_proc_crit"]
        assert findings[0].trend.last_seen_window == record["window"]
        plans.extend(new_plans)

    decisions = [kwargs for event, kwargs in decision_log.entries
                 if event == "optimizer.cbandit.decision"]
    assert len(decisions) == 3
    # The retained vectors are comm-dominant.  cbandit deliberately records
    # input because its barrier-attribution branch redirects a comm-dominant
    # window with no comm-owned knob to registered storage candidates.
    assert [entry["dominant_domain"] for entry in decisions] == ["input"] * 3
    assert [entry["accept"] for entry in decisions] == [
        "no_predicted_effect", "no_predicted_effect", "unverifiable_in_horizon",
    ]
    expected_domain_times = [
        {name: round(record[name], 4)
         for name in ("input", "decode", "comm", "compute")}
        for record in _R1_FACT_WINDOWS[:-1]
    ]
    assert [entry["domain_times"] for entry in decisions] == expected_domain_times
    assert len(plans) == 0
    assert planner.current_values["dlio.read_threads"] == 4
    assert planner.current_values["dlio.prefetch_size"] == 2


_P3_R3_ROOT = Path("/p/vast1/izzet/lfmr/p3-tf-tail-diagnostic-20260923-r3")


def test_p3_tf_r3_post_horizon_payloads_decline_unpriced_action(monkeypatch):
    """Retrospective one-node replay: valid slope facts are not action proof.

    The source run's original detector verdict remains red.  The separate
    accepted retrospective gate classified window 11 as reader-absent; this
    replay uses the ten post-horizon payloads actually published to
    LiveDiagnoser, without fabricating a finding or injecting a plan.  Missing
    cluster artifacts skip this cluster-specific evidence check rather than
    substituting another run.
    """
    if not _P3_R3_ROOT.is_dir():
        pytest.skip("retained Corona r3 artifacts unavailable")
    originals = {
        "p3-tf-detect-verdict.json": "eda57897be1fe55e1f64c5b978476225a167a54c51b64c1576bb3e89f14a5475",
        "p3-tf-detect-retrospective-v2.json": "f73a2eac6efba4333873aadc102f456956f245fc8a8e5329e6a008a783d98e8f",
    }
    for name, digest in originals.items():
        assert hashlib.sha256((_P3_R3_ROOT / name).read_bytes()).hexdigest() == digest
    log = _P3_R3_ROOT / "nodes/corona207/agent/dfdiagnoser_agent/dfdiagnoser.log"
    assert hashlib.sha256(log.read_bytes()).hexdigest() == (
        "1d0dd6e6c3458b5ed2ab7b098fcb51b90f7cb0f9359a40af36ea97b09bfd2152")
    publications = [json.loads(line) for line in log.read_text().splitlines()
                    if '"event": "analysis_facts.payload_published"' in line]
    assert len(publications) == 10
    _freeze_p3_cbandit(monkeypatch)
    decision_log = _DecisionLog()
    monkeypatch.setattr(cbandit_planner, "logger", decision_log)
    diagnoser, planner = Diagnoser(), _p3_read_threads_planner()
    assert planner.knobs["dlio.read_threads"].range[0] <= 4 < planner.knobs["dlio.read_threads"].range[1]
    for window, publication in enumerate(publications, 1):
        assert publication["window_index"] == window
        facts = json.loads(publication["payload"])["facts"]
        assert len(facts) == 1
        fact = facts[0]
        assert fact["provenance"]["rule_id"] == (
            "cosmoflow.reader_posix_read.rank_thread_pool.reference45.v1")
        assert fact["window"]["window"] == window
        metrics = fact["evidence"]["metrics"]
        assert 0.0 < metrics["reader_posix_read_ops_slope"] < 1.0
        assert metrics["domain_comm_time_proc_crit"] > metrics["domain_input_time_proc_crit"]
        assert metrics["window_index"] == window
        assert metrics["windows_per_epoch"] == 11
        assert metrics["windows_remaining"] == 11 - window
        touched = diagnoser._handle_analysis_facts(
            _Event(publication["payload"].encode()), metadata={})
        findings = diagnoser._build_control_findings(window, touched)
        assert len(findings) == 1
        assert findings[0].key_metrics["window_index"] == window
        assert findings[0].key_metrics["windows_per_epoch"] == 11
        assert findings[0].key_metrics["windows_remaining"] == 11 - window
        assert [plan for finding in findings
                for plan in planner.process_finding(_to_optimizer_finding(finding))] == []
    decisions = [payload for event, payload in decision_log.entries
                 if event == "optimizer.cbandit.decision"]
    assert len(decisions) == 9
    assert all(item["decision"] == "stay" for item in decisions)
    assert [item["accept"] for item in decisions] == [
        "no_predicted_effect", "no_predicted_effect",
        *(["unverifiable_in_horizon"] * 7),
    ]
    assert planner.current_values["dlio.read_threads"] == 4


def _replay_p3_input_exposure(monkeypatch, *, input_exposure):
    """Matched chronological controls; only the named input exposure differs."""
    _freeze_p3_cbandit(monkeypatch)
    decision_log = _DecisionLog()
    monkeypatch.setattr(cbandit_planner, "logger", decision_log)
    engine, diagnoser, planner = _p3_engine(), Diagnoser(), _p3_read_threads_planner()
    plans = []
    for source in _R1_FACT_WINDOWS:
        record = dict(source, input=(input_exposure if input_exposure is not None else source["input"]))
        facts, findings, new_plans = _deliver_p3_fact(engine, diagnoser, planner, record)
        assert len(facts) == len(findings) == 1
        plans.extend(new_plans)
    decisions = [kwargs for event, kwargs in decision_log.entries
                 if event == "optimizer.cbandit.decision"]
    return planner, decisions, plans


def test_p3_tf_missing_input_exposure_diverges_after_sufficient_history(monkeypatch):
    """Counterfactual zero-input control; r1 rows remain otherwise unchanged."""
    positive, positive_decisions, positive_plans = _replay_p3_input_exposure(
        monkeypatch, input_exposure=None)
    zero, zero_decisions, zero_plans = _replay_p3_input_exposure(monkeypatch, input_exposure=0.0)

    # Four submitted findings yield three finalized chronological observations,
    # meeting MIN_SAMPLES=3 before the last decision.  The controllers retain
    # the same rows/knobs; only this named predicted-effect input is changed.
    assert len(positive_decisions) == len(zero_decisions) == 3
    assert len(positive._epoch_dom["input"]) == len(zero._epoch_dom["input"]) == 3
    assert all(value > 0.0 for value in positive._epoch_dom["input"])
    assert list(zero._epoch_dom["input"]) == [0.0, 0.0, 0.0]
    assert [entry["accept"] for entry in positive_decisions] == [
        "no_predicted_effect", "no_predicted_effect", "unverifiable_in_horizon",
    ]
    assert [entry["accept"] for entry in zero_decisions] == ["no_predicted_effect"] * 3
    assert positive_decisions[-1]["accept"] != zero_decisions[-1]["accept"]
    assert positive_plans == zero_plans == []
    assert "dlio.read_threads" in zero.knobs
    assert zero.knobs["dlio.read_threads"].range[0] <= 4 < zero.knobs["dlio.read_threads"].range[1]
    assert zero.current_values["dlio.read_threads"] == 4


def test_synthetic_wisio_slope_detects_within_family_and_hands_severity_to_livediagnoser(monkeypatch):
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
