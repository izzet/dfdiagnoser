"""Control-path tests for the StorMer worker-read slope + measured-wait input rule.

Synthetic fixtures (no run data): offline control-path checks, not detection quality, performance,
worker-POSIX coverage on a real run, or trajectory evidence.  See
liveflow_shm/DETECTOR-CARD-stormer-input.md.
"""

import re
from copy import deepcopy
from pathlib import Path

import dftracer
_AP = str(Path(__file__).resolve().parents[2] / "dfanalyzer/python/dftracer")
if _AP not in list(dftracer.__path__):
    try:
        dftracer.__path__.insert(0, _AP)
    except AttributeError:
        dftracer.__path__ = [_AP] + list(dftracer.__path__)

import pandas as pd
import pytest

from dftracer.analyzer.fact_engine import FactEngine

_RULE = (Path(__file__).resolve().parents[2]
         / "dfanalyzer/python/dftracer/analyzer/configs/fact_rules/liveflow-stormer-input-slope-exposure.yaml")


def _engine():
    return FactEngine.from_rule_file(
        str(_RULE), strict_time_semantics=True, allow_mixed_time_aggregates=False)


def _context(*, input_s=0.9, decode_s=0.3):
    return {
        "native_window": 4,
        "worker_pids": [20, 21],
        "row_pid_by_index": {"worker-20": 20, "worker-21": 21},
        "window_metrics": {
            "domain_coherent": 1.0, "domain_procs_active": 4.0,
            "window_wall_time_proc_crit": 10.0,
            "domain_input_time_proc_crit": input_s,
            "domain_decode_time_proc_crit": decode_s,
            "domain_comm_time_proc_crit": 5.0,
            "domain_compute_time_proc_crit": 3.0,
            "residual_time_proc_crit": 10.0 - (input_s + decode_s + 8.0),
            "epoch_nunique": 1.0, "step_nunique": 20.0, "step_min": 1.0, "step_max": 20.0,
        },
        "completed_window_worker_roster_provenance": {"worker_roster_coverage": 1.0},
    }


def _facts(context, *, slope=0.4, count=0.10, time=0.25):
    rows = pd.DataFrame({
        "reader_posix_read_count_frac_total": [count, 0.30],
        "reader_posix_read_time_proc_frac_total": [time, 0.25],
        "reader_posix_read_ops_slope": [slope, 1.2],
    }, index=["worker-20", "worker-21"])
    return _engine().evaluate({("proc_name",): rows}, raw_stats={"run_id": "stormer"}, context=context)


def test_rule_uses_no_constant_beyond_reference_and_zero():
    when = [l for l in _RULE.read_text().splitlines() if l.strip().startswith("when:")][0]
    assert set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", when)) <= {"0", "0.0", "1", "1.0"}
    assert "stall_storage_rank_coverage" not in when      # DLIO coverage semantics NOT transferred
    assert "rank_thread_only" not in _RULE.read_text().replace("# ", "").split("rules:")[1]


def test_positive_names_measured_total_estimated_split_and_worker_slope_separately():
    out = _facts(_context(input_s=0.9, decode_s=0.3))
    (fact,) = out[("proc_name",)]
    assert fact.severity.score == pytest.approx(0.6)               # 1 - slope of the slowest worker
    assert fact.opportunity_tags == ["reader_parallelism"]         # never a decode knob
    m = _evidence(fact)
    assert m["measured_total_wait_time_proc_crit"] == pytest.approx(1.2)
    assert m["estimated_storage_wait_share"] == pytest.approx(0.75)
    assert m["reader_posix_read_ops_slope"] == pytest.approx(0.4)


def test_decode_only_estimated_wait_still_counts_as_measured_total_but_reports_zero_storage_share():
    (fact,) = _facts(_context(input_s=0.0, decode_s=0.8))[("proc_name",)]
    m = _evidence(fact)
    assert m["measured_total_wait_time_proc_crit"] == pytest.approx(0.8)
    assert m["estimated_storage_wait_share"] == pytest.approx(0.0)


def test_stall_storage_coverage_field_is_not_consulted_zero_duration_omission():
    # StorMer omits zero-duration stall records, so a partial-coverage value must neither veto a
    # critical-rank-measured wait nor be able to enable an otherwise-suppressed window.
    partial = _context()
    partial["completed_window_provenance"] = {"stall_storage_rank_coverage": 0.0}
    assert len(_facts(partial)[("proc_name",)]) == 1
    full = _context(input_s=0.0, decode_s=0.0)
    full["completed_window_provenance"] = {"stall_storage_rank_coverage": 1.0}
    assert _facts(full) == {}


@pytest.mark.parametrize("case", (
    "hidden_prefetch_zero_measured_wait",   # slope < 1, both wait terms 0
    "uniform_slowdown_slope_one",           # slope == 1 exactly
    "noncoherent_window",
    "worker_roster_incomplete",
    "worker_roster_absent",
    "no_verified_workers",                  # worker rows cannot be selected
))
def test_negative_controls_suppress_the_fact(case):
    ctx, kw = _context(), {}
    if case == "hidden_prefetch_zero_measured_wait":
        ctx = _context(input_s=0.0, decode_s=0.0)
    elif case == "uniform_slowdown_slope_one":
        kw["slope"] = 1.0
    elif case == "noncoherent_window":
        ctx["window_metrics"]["domain_coherent"] = 0.0
    elif case == "worker_roster_incomplete":
        ctx["completed_window_worker_roster_provenance"]["worker_roster_coverage"] = 0.0
    elif case == "worker_roster_absent":
        ctx.pop("completed_window_worker_roster_provenance")
    else:
        ctx["worker_pids"] = []
    assert _facts(ctx, **kw) == {}, case


def test_missing_or_zero_normalization_suppresses_rather_than_becoming_evidence():
    assert _facts(_context(), count=0.0) == {}
    assert _facts(_context(), time=0.0) == {}
    assert _facts(_context(), slope=float("nan")) == {}


def _evidence(fact):
    return dict(fact.evidence["metrics"])


def test_fastpath_plan_builds_for_the_stormer_xlayer_preset():
    from dftracer.analyzer.config import AnalyzerPresetConfigStormerXLayer
    from dftracer.analyzer.fastpath import FastPathPlan
    preset = AnalyzerPresetConfigStormerXLayer()
    plan = FastPathPlan.build(preset, ["window", "proc_name"], list(preset.layer_defs), _engine())
    assert plan is not None, "rule inputs are not all producible from the stormer_xlayer preset"
