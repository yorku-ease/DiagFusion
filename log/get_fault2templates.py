import time
import os
import pandas as pd
import re
from datetime import datetime


def extract_timestamp(message):
    if isinstance(message, str):  # Check if message is a string
        match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", message)
        return match.group() if match else None
    return None


def time_to_ts(ctime):
    if ctime is None:
        return None
    try:
        timeArray = time.strptime(ctime, "%Y-%m-%d %H:%M:%S,%f")
    except:
        timeArray = time.strptime(ctime, "%Y-%m-%d")
    return int(time.mktime(timeArray)) * 1000


def get_fault2templates(fault_file, mid_dir):
    print(f"[get_fault2templates] fault_file: {fault_file}")
    faults = pd.read_csv(fault_file)

    inputs = os.walk(mid_dir)
    paths = []
    for root, dirs, files in inputs:
        for file in files:
            if root.endswith("drain_result") and "structured" in file:
                path = os.path.join(root, file)
                paths.append(path)

    for path in paths:
        cur_csv = pd.read_csv(path)
        print(path)

        if cur_csv.iloc[0]["datetime"] == "datetime":
            cur_csv = cur_csv.drop([0])

        cur_csv["timestamp"] = cur_csv["Content"].apply(lambda x: extract_timestamp(x))
        cur_csv["timestamp"] = cur_csv["timestamp"].apply(lambda x: time_to_ts(x))
        cur_csv = cur_csv[cur_csv["timestamp"] != None]
        cur_csv.sort_values("timestamp", inplace=True)

        if cur_csv.shape[0] < 1:
            continue

        csv_st_time = cur_csv.iloc[0]["timestamp"]
        csv_ed_time = cur_csv.iloc[-1]["timestamp"]

        for idx, row in faults.iterrows():
            _id = row["index"]

            st_time = time_to_ts(row["st_time"])
            ed_time = time_to_ts(row["ed_time"])

            if ed_time < csv_st_time or st_time > csv_ed_time:
                continue

            fault_file_dir = os.path.dirname(fault_file)
            fault2template_file = f"{mid_dir}/fault_files/{_id}.csv"
            print(fault2template_file)
            if not os.path.exists(f"{mid_dir}fault_files/"):
                os.makedirs(f"{mid_dir}fault_files/")
            selected_rows = cur_csv[
                (st_time <= cur_csv["timestamp"]) & (cur_csv["timestamp"] <= ed_time)
            ]
            selected_rows.to_csv(
                fault2template_file, mode="a", index=False, header=False
            )
            print(
                f"[get_fault2templates] execute_time: {st_time} \
            st_time: {st_time} ed_time: {ed_time} fault2template_file: {fault2template_file}"
            )
    print(f"[get_fault2templates done] fault_file: {fault_file}")
