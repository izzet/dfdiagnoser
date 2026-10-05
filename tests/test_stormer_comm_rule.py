"""Control-path tests for the StorMer all-reduce slope + skew-attributed comm exposure rule.

Synthetic fixtures (no run data): offline control-path checks, not detection quality, performance,
or trajectory evidence.  See liveflow_shm/STORMER-LOOP-DESIGN-2026-09-26.md.
"""

import re
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
         / "dfanalyzer/python/dftracer/analyzer/configs/fact_rules/liveflow-stormer-comm-slope-exposure.yaml")

HOST = "tuolumne1001"
RANK_PIDS = [1001, 1002, 1003, 1004]


def _engine(rule=_RULE):
    return FactEngine.from_rule_file(
        str(rule), strict_time_semantics=True, allow_mixed_time_aggregates=False)


def rank_rows(slopes=(0.6, 1.1, 1.1, 1.2), *, count=0.25, time=0.25):
    names = ["app#%s#%d#%d" % (HOST, pid, pid) for pid in RANK_PIDS]
    rows = pd.DataFrame({
        "comm_all_reduce_count_frac_total": [count] * 4,
        "comm_all_reduce_time_proc_frac_total": [time] * 4,
        "comm_all_reduce_ops_slope": list(slopes),
    }, index=names)
    return rows, {n: pid for n, pid in zip(names, RANK_PIDS)}


def skew_metrics(*, A=0.0, E=None, W=None, cov=1.0, typ_in=0.2, typ_comm=2.0, typ_dec=0.0,
                 typ_comp=3.0, wall=5.6, residual=None, late_in=None):
    E = A if E is None else E
    W = A if W is None else W
    return {
        "domain_skew_coverage": cov, "domain_input_skew_attributed_time": A,
        "domain_input_skew_excess_time": E, "domain_comm_barrier_wait_time": W,
        "domain_skew_late_proc_input_time": typ_in + E if late_in is None else late_in,
        "domain_skew_late_proc_comm_time": 0.3,
        "domain_comm_residual_after_skew_time": (typ_comm - A) if residual is None else residual,
        "domain_input_time_proc_typ": typ_in, "domain_decode_time_proc_typ": typ_dec,
        "domain_comm_time_proc_typ": typ_comm, "domain_compute_time_proc_typ": typ_comp,
        "window_wall_time_proc_typ": wall,
    }


def coherent(*, window=4, comm=2.0, input_s=0.2, compute=3.0, wall=5.6, procs=4.0, steps=20.0):
    return {
        "domain_coherent": 1.0, "domain_procs_active": procs,
        "window_wall_time_proc_crit": wall,
        "domain_input_time_proc_crit": input_s, "domain_decode_time_proc_crit": 0.0,
        "domain_comm_time_proc_crit": comm, "domain_compute_time_proc_crit": compute,
        "residual_time_proc_crit": wall - input_s - comm - compute,
        "epoch_nunique": 1.0, "step_nunique": steps, "step_min": 1.0, "step_max": steps,
    }


def context(rows_map, *, window=4, host=HOST, rank_pids=None, **skew):
    wm = coherent(window=window)
    wm.update(skew_metrics(**skew))
    return {"native_window": window, "native_host": host,
            "rank_pids": RANK_PIDS if rank_pids is None else rank_pids,
            "row_pid_by_index": rows_map, "window_metrics": wm}


def facts(*, slopes=(0.6, 1.1, 1.1, 1.2), count=0.25, time=0.25, edit=None, ctx_edit=None, **skew):
    rows, rmap = rank_rows(slopes, count=count, time=time)
    ctx = context(rmap, **skew)
    if edit:
        edit(ctx["window_metrics"])
    if ctx_edit:
        ctx_edit(ctx)
    return _engine().evaluate({("proc_name",): rows}, raw_stats={"run_id": "stormer-comm"}, context=ctx)


def _evidence(fact):
    return dict(fact.evidence["metrics"])


def test_rule_uses_no_constant_beyond_reference_and_zero():
    when = [l for l in _RULE.read_text().splitlines() if l.strip().startswith("when:")][0]
    assert set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", when)) <= {"0", "0.0", "1", "1.0"}
    assert "stall_storage_rank_coverage" not in when     # StorMer omits zero-duration stall records


def test_positive_names_family_slope_residual_exposure_and_attribution_separately():
    out = facts(A=0.5, W=0.6, E=0.9, typ_comm=2.0)
    (fact,) = out[("proc_name",)]
    assert fact.severity.score == pytest.approx(0.4)            # 1 - min all-reduce slope
    assert fact.opportunity_tags == ["comm_optimization"]      # never an input knob
    m = _evidence(fact)
    assert m["comm_all_reduce_ops_slope"] == pytest.approx(0.6)
    assert m["domain_comm_residual_after_skew_time"] == pytest.approx(1.5)   # 2.0 - A: not raw comm
    assert m["domain_input_skew_attributed_time"] == pytest.approx(0.5)
    assert m["comm_exposure_after_skew_share"] == pytest.approx(0.75)


def test_no_skew_positive_exposure_is_all_comm():
    (fact,) = facts(A=0.0, typ_comm=2.0)[("proc_name",)]
    m = _evidence(fact)
    assert m["domain_comm_residual_after_skew_time"] == pytest.approx(2.0)
    assert m["comm_exposure_after_skew_share"] == pytest.approx(1.0)


@pytest.mark.parametrize("case", (
    "uniform_comm_slope_one",              # every rank-thread has its share: slope exactly 1
    "skew_fully_explains_wait",            # residual 0: the wait is the late rank's input, not comm
    "missing_coverage_zero",               # blocking probe off / a rank has no comm record
    "missing_coverage_absent",
    "missing_residual_column",
    "nan_residual",
    "noncoherent_window",
    "wrong_host",
    "non_rank_pid",
    "mapped_pid_mismatch",
    "zero_count_normalization",
    "zero_time_normalization",
    "nan_slope",
))
def test_negative_controls_suppress_the_fact(case):
    kw, edit, ctx_edit = {"A": 0.5, "W": 0.6, "E": 0.9}, None, None
    if case == "uniform_comm_slope_one":
        kw["slopes"] = (1.0, 1.0, 1.0, 1.0)
    elif case == "skew_fully_explains_wait":
        kw.update(residual=0.0)
    elif case == "missing_coverage_zero":
        kw["cov"] = 0.0
    elif case == "missing_coverage_absent":
        edit = lambda m: m.pop("domain_skew_coverage")
    elif case == "missing_residual_column":
        edit = lambda m: m.pop("domain_comm_residual_after_skew_time")
    elif case == "nan_residual":
        kw["residual"] = float("nan")
    elif case == "noncoherent_window":
        edit = lambda m: m.update(domain_coherent=0.0)
    elif case == "wrong_host":
        ctx_edit = lambda c: c.update(native_host="otherhost")
    elif case == "non_rank_pid":
        ctx_edit = lambda c: c.update(rank_pids=[7, 8, 9, 10])
    elif case == "mapped_pid_mismatch":
        ctx_edit = lambda c: c["row_pid_by_index"].update({k: 1 for k in c["row_pid_by_index"]})
    elif case == "zero_count_normalization":
        kw["count"] = 0.0
    elif case == "zero_time_normalization":
        kw["time"] = 0.0
    else:
        kw["slopes"] = (float("nan"),) * 4
    assert facts(edit=edit, ctx_edit=ctx_edit, **kw) == {}, case


def test_worker_rows_and_input_family_do_not_stand_in_for_the_all_reduce_family():
    rows, rmap = rank_rows()
    rows = rows.drop(columns=["comm_all_reduce_ops_slope"])
    rows["reader_posix_read_ops_slope"] = [0.3] * 4          # a different family is never borrowed
    ctx = context(rmap, A=0.5)
    out = _engine().evaluate({("proc_name",): rows}, raw_stats={"run_id": "x"}, context=ctx)
    assert out == {}


def test_fastpath_plan_builds_for_the_stormer_xlayer_preset():
    from dftracer.analyzer.config import AnalyzerPresetConfigStormerXLayer
    from dftracer.analyzer.fastpath import FastPathPlan
    preset = AnalyzerPresetConfigStormerXLayer()
    plan = FastPathPlan.build(preset, ["window", "proc_name"], list(preset.layer_defs), _engine())
    assert plan is not None, "comm rule inputs are not all producible from the stormer_xlayer preset"
