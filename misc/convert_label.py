import pandas as pd

labels = pd.read_csv("data/gaia/test/labels.csv")
labels["anomaly_type"] = labels["anomaly_type"].apply(lambda x: f"[{x}]")
labels.to_csv("data/gaia/test/labels_converted.csv")
