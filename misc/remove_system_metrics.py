import pandas as pd

metrics = pd.read_csv("data/gaia/test/anomalies/metric_names.csv")
metrics = metrics[~metrics["name"].str.contains("system_")]
metrics.to_csv("data/gaia/test/anomalies/metric_names_no_system.csv", index=False)
