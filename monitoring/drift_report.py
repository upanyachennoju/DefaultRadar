from pathlib import Path

import pandas as pd
from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset


def _resolve_path(path: str) -> Path:
    path_obj = Path(path)
    if path_obj.exists():
        return path_obj

    current = Path.cwd()
    while current != current.parent:
        candidate = current / path_obj
        if candidate.exists():
            return candidate
        current = current.parent

    raise FileNotFoundError(f"Data file not found: {path}")


def generate_drift_report(
    reference_path: str = "data/raw/synthetic_credit_risk.csv",
    current_path: str = "data/raw/current_data.csv",
    output_path: str = "artifacts/reports/drift_report.html"
):
    """Generate a data drift report using Evidently."""
    reference_path = _resolve_path(reference_path)
    current_path = _resolve_path(current_path)
    output_path = Path(output_path)

    reference_data = pd.read_csv(reference_path)
    current_data = pd.read_csv(current_path)

    data_definition = DataDefinition()
    ref = Dataset.from_pandas(reference_data, data_definition)
    curr = Dataset.from_pandas(current_data, data_definition)

    report = Report(metrics=[DataDriftPreset()])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = report.run(reference_data=ref, current_data=curr)
    snapshot.save_html(output_path)

    print(f"Drift report generated and saved to: {output_path}")


if __name__ == "__main__":
    generate_drift_report()
