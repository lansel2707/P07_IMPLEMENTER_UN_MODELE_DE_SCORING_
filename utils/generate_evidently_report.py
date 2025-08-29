from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import os

data_path = "data/application_train.csv"
reference_path = "data_processed/application_train_clean.csv"

df = pd.read_csv(data_path)
ref_df = pd.read_csv(reference_path)

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=df)

os.makedirs("reports", exist_ok=True)
report.save_html("reports/evidently_report.html")

print("✅ Evidently report generated at reports/evidently_report.html")

