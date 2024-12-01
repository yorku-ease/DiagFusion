import pandas as pd

demo_df = pd.read_csv("data/gaia/demo/demo_1100/demo_train160.csv")
print(demo_df["anomaly_type"].value_counts(dropna=False, sort=False))

df = pd.read_csv("data/gaia/test/labels.csv")
print(df["anomaly_type"].value_counts(dropna=False, sort=False))
