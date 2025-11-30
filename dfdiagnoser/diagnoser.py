import glob
import json
import os
import pandas as pd


from .scoring import score_metrics
from .types import DiagnosisResult
from .utils.log_utils import console_block


class Diagnoser:
    def __init__(self):
        pass

    def diagnose_checkpoint(self, checkpoint_dir: str, metric_boundaries: dict = {}):
        if not os.path.exists(checkpoint_dir):
            raise FileNotFoundError(
                f"Checkpoint directory {checkpoint_dir} does not exist")
        if not os.path.isdir(checkpoint_dir):
            raise NotADirectoryError(
                f"Checkpoint directory {checkpoint_dir} is not a directory")
        if not os.listdir(checkpoint_dir):
            raise ValueError(f"Checkpoint directory {checkpoint_dir} is empty")
        
        with console_block("Load raw stats"):
            raw_stats_paths = glob.glob(os.path.join(
                checkpoint_dir, "_raw_stats_*.json"))
            if not raw_stats_paths:
                raise ValueError(
                    f"Checkpoint directory {checkpoint_dir} does not contain any raw stats files")
            with open(raw_stats_paths[0], "r") as f:
                raw_stats = json.load(f)
        flat_view_paths = glob.glob(os.path.join(
            checkpoint_dir, "_flat_view_*.parquet"))
        if not flat_view_paths:
            raise ValueError(
                f"Checkpoint directory {checkpoint_dir} does not contain any flat view files")

        with console_block("Score flat views"):
            scored_flat_views = []
            for flat_view_path in flat_view_paths:
                flat_view = pd.read_parquet(flat_view_path)
                scored_flat_view = score_metrics(flat_view, metric_boundaries)
                scored_flat_views.append(scored_flat_view)

        return DiagnosisResult(
            flat_view_paths=flat_view_paths,
            scored_flat_views=scored_flat_views
        )

    def diagnose_zmq(self, address: str):
        pass

    def _diagnose(self, data: dict):
        pass
