from evidently import Report, Dataset, DataDefinition
from evidently.presets import DataDriftPreset
import pandas as pd
from pathlib import Path

def generate_drift_report():
    reference_data = pd.read_csv("artifacts/train.csv")
    current_data = pd.read_csv("artifacts/current_data.csv")

    data_definition = DataDefinition()
    ref = Dataset.from_pandas(reference_data, data_definition)
    curr = Dataset.from_pandas(current_data, data_definition)

    report = Report(metrics=[
        DataDriftPreset()
    ])

    Path("monitoring/reports").mkdir(parents=True, exist_ok=True)

    snapshot = report.run(reference_data=ref, current_data=curr)
    snapshot.save_html("artifacts/drift_report.html")
    print("Drift report generated and saved.")