"""Control-path tests for the StorMer rank-side input rule (fetch_iter family slope + measured critical-rank wait).

Synthetic; offline candidate, NOT judge-reviewed.  See liveflow_shm/STORMER-LOOP-DESIGN-2026-09-26.md sec 3.
"""
import re

import pandas as pd
import pytest

try:
    from . import test_stormer_comm_rule as c
except ImportError:
    import test_stormer_comm_rule as c

_RULE = c._RULE.parent / "liveflow-stormer-input-fetch-slope-exposure.yaml"


def _facts(*, slopes=(0.5, 1.1, 1.1, 1.3), count=0.25, time=0.25, input_s=0.9, decode_s=0.3, edit=None,
           ctx_edit=None, **skew):
    rows, rmap = c.rank_rows()
    rows = pd.DataFrame({"fetch_iter_count_frac_total": [count] * 4,
                         "fetch_iter_time_proc_frac_total": [time] * 4,
                         "fetch_iter_ops_slope": list(slopes)}, index=rows.index)
    ctx = c.context(rmap, **skew)
    wm = ctx["window_metrics"]
    wm.update(domain_input_time_proc_crit=input_s, domain_decode_time_proc_crit=decode_s)
    if edit:
        edit(wm)
    if ctx_edit:
        ctx_edit(ctx)
    return c._engine(_RULE).evaluate({("proc_name",): rows}, raw_stats={"run_id": "x"}, context=ctx)


def test_no_constant_beyond_reference_and_zero_and_no_stall_coverage_term():
    when = [l for l in _RULE.read_text().splitlines() if l.strip().startswith("when:")][0]
    assert set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?", when)) <= {"0", "0.0", "1", "1.0"}
    assert "stall_storage_rank_coverage" not in when and "skew" not in when


def test_positive_names_family_slope_measured_total_and_estimated_split_separately():
    (fact,) = _facts()[("proc_name",)]
    assert fact.severity.score == pytest.approx(0.5)
    assert fact.opportunity_tags == ["reader_parallelism"]
    m = dict(fact.evidence["metrics"])
    assert m["measured_total_wait_time_proc_crit"] == pytest.approx(1.2)
    assert m["estimated_storage_wait_share"] == pytest.approx(0.75)
    assert m["fetch_iter_ops_slope"] == pytest.approx(0.5)


def test_input_fact_fires_without_skew_coverage_because_skew_is_the_comm_rules_conjunct():
    assert len(_facts(cov=0.0)[("proc_name",)]) == 1


@pytest.mark.parametrize("case", ("hidden_prefetch_zero_wait", "uniform_slope_one", "noncoherent", "wrong_host",
                                  "non_rank_pid", "zero_count", "zero_time", "nan_slope"))
def test_negative_controls_suppress_the_fact(case):
    kw = {}
    if case == "hidden_prefetch_zero_wait":
        kw.update(input_s=0.0, decode_s=0.0)
    elif case == "uniform_slope_one":
        kw["slopes"] = (1.0,) * 4
    elif case == "noncoherent":
        kw["edit"] = lambda m: m.update(domain_coherent=0.0)
    elif case == "wrong_host":
        kw["ctx_edit"] = lambda x: x.update(native_host="other")
    elif case == "non_rank_pid":
        kw["ctx_edit"] = lambda x: x.update(rank_pids=[7])
    elif case == "zero_count":
        kw["count"] = 0.0
    elif case == "zero_time":
        kw["time"] = 0.0
    else:
        kw["slopes"] = (float("nan"),) * 4
    assert _facts(**kw) == {}, case
