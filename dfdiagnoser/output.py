import pandas as pd
from typing import Optional

from .types import DiagnosisResult, FileOutputFormat


class Output:
    def __init__(self):
        pass

    def handle_result(self, result: DiagnosisResult):
        pass


class ConsoleOutput(Output):
    def __init__(self):
        super().__init__()

    def handle_result(self, result: DiagnosisResult):
        pass


class FileOutput(Output):
    def __init__(self, output_dir: Optional[str] = None, output_format: FileOutputFormat = "json"):
        super().__init__()
        self.output_dir = output_dir
        self.output_format = output_format

    def handle_result(self, result: DiagnosisResult):
        for flat_view_path, scored_flat_view in zip(result.flat_view_paths, result.scored_flat_views):
            if self.output_dir:
                output_path = f"{self.output_dir}/{flat_view_path.split('/')[-1].split('.')[0]}_scored.{self.output_format}"
            else:
                output_path = f"{flat_view_path.split('.')[0]}_scored.{self.output_format}"

            if self.output_format == "json":
                scored_flat_view.to_json(output_path, orient="index")
            elif self.output_format == "csv":
                scored_flat_view.to_csv(output_path, index=True)
            elif self.output_format == "parquet":
                scored_flat_view.to_parquet(output_path, index=True)
            else:
                raise ValueError(
                    f"Unsupported output format: {self.output_format}")
