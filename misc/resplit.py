import pandas as pd

df = pd.read_csv("labels.csv")
df["data_type"] = "test"
df.loc[:100, "data_type"] = "train"
df.to_csv("labels_resplit.csv")
