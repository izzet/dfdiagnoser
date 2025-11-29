import dataclasses as dc
import pandas as pd
from typing import List, Literal


FileOutputFormat = Literal["csv", "json", "parquet"]


@dc.dataclass
class DiagnosisResult:
    flat_view_paths: List[str]
    scored_flat_views: List[pd.DataFrame]
