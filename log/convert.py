import time
import os
import pandas as pd
import pytz
import numpy as np
import re

tz = pytz.timezone("Asia/Shanghai")


def time_to_ts(ctime):
    if ctime is None:
        return 0
    try:
        timeArray = time.strptime(ctime, "%Y-%m-%d %H:%M:%S,%f")
    except:
        timeArray = time.strptime(ctime, "%Y-%m-%d")
    return int(time.mktime(timeArray)) * 1000


def extract_timestamp(message):
    if isinstance(message, str):  # Check if message is a string
        match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", message)
        return match.group() if match else None
    return None


if __name__ == "__main__":
    path = "mid/DataSet-main/MicroSS/business/fault_files"
    for file in os.listdir(path):
        df = pd.read_csv(
            f"{path}/{file}",
            sep=",",
            names=[
                "LineId",
                "datetime",
                "service",
                "Content",
                "EventId",
                "EventTemplate",
                "ParameterList",
                "convertion",
            ],
        )
        df["timestamp"] = df["Content"].apply(lambda x: extract_timestamp(x))
        df["datetime"] = df["timestamp"].apply(lambda x: time_to_ts(x))
        output_headers = [
            "LineId",
            "datetime",
            "service",
            "Content",
            "EventId",
            "EventTemplate",
            "ParameterList",
        ]
        print(f"Writing logs_faults/{file}")
        df[output_headers].to_csv(f"logs_faults/{file}")
