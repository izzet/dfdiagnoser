import dataclasses as dc
import pandas as pd
from typing import Any, Dict, List, Literal, Optional, Tuple


FileOutputFormat = Literal["csv", "json", "parquet"]


@dc.dataclass
class TrendEvidence:
    prevalence: float
    persistence: int
    onset_window: int
    peak_severity_window: int
    last_seen_window: int
    support_windows: int
    trend_direction: str  # "worsening", "improving", "stable", "insufficient_data"


@dc.dataclass
class DiagnosisFinding:
    finding_type: str
    scope: str
    layer: Optional[str]
    motif: str  # "warmup_transient", "persistent_pressure", "rank_skew_induced", "checkpoint_tail_risk", "unclassified"
    severity: str  # human-readable label for logging
    severity_score: float  # continuous [0.0, 1.0] for gating/scaling
    confidence: float
    trend: TrendEvidence
    contributing_facts: List[Tuple[str, str]]  # list of (fact_type, scope)
    recommendation_bundle: str
    summary: str
    opportunity_tags: List[str] = dc.field(default_factory=list)
    suppresses_tags: List[str] = dc.field(default_factory=list)
    key_metrics: Dict[str, float] = dc.field(default_factory=dict)


@dc.dataclass
class DiagnosisResult:
    flat_view_paths: List[str]
    scored_flat_views: List[pd.DataFrame]
    findings: List[DiagnosisFinding] = dc.field(default_factory=list)
