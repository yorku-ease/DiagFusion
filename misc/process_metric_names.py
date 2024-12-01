import os
import re
import pandas as pd

# Directory containing the files
directory = "GAIA-DataSet-main/MicroSS/metric/metric"

# Regular expression pattern to match filenames in the format and capture the metric name
pattern = re.compile(r"^(.+)_2021-07-\d{2}_2021-07-\d{2}\.csv$")

# List to store the metric names
metrics = set()

# Iterate over each file in the directory
for filename in os.listdir(directory):
    # Match the pattern and capture the metric name
    match = pattern.match(filename)
    if match:
        metric_name = match.group(1)
        metrics.add(metric_name)

# Convert the list of metric names into a DataFrame
df_metrics = pd.DataFrame(metrics, columns=["metric_name"])

# Output the metrics to a CSV file
output_path = "data/gaia/test/metric_names.csv"
df_metrics.to_csv(output_path, index=False)

print(f"Metric names have been captured and saved to {output_path}")
