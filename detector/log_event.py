from tqdm.notebook import trange, tqdm
from sklearn.model_selection import train_test_split
import time
from datetime import datetime
from tqdm.notebook import trange, tqdm
import json
import pandas as pd
import numpy as np
import sys
import os
import re


def extract_timestamp(message):
    if isinstance(message, str):  # Check if message is a string
        match = re.search(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3}", message)
        return match.group() if match else None
    return None


def time_to_ts(ctime):
    if ctime is None:
        return 0
    try:
        timeArray = time.strptime(ctime, "%Y-%m-%d %H:%M:%S,%f")
    except:
        timeArray = time.strptime(ctime, "%Y-%m-%d")
    return int(time.mktime(timeArray)) * 1000


class LogEvent:
    # def __init__(self, config, method):
    #     self.config = config
    #     self.method = method
    #     self.methods_list = {'log_scale': self.log_scale,
    #                          'random_sampling': self.random_sampling,
    #                          'stratified_sampling': self.stratified_sampling}
    #     assert self.method in self.methods_list.keys()
    def __init__(self):
        print("begin")

    @staticmethod
    def log_scale(labels, log_files, services, save_path):
        logs_list = []
        for i in tqdm(range(len(labels))):
            id = labels.iloc[i]["index"]
            df = pd.read_csv(f"{log_files}/{id}.csv")
            df["timestamp"] = df["Content"].apply(lambda x: extract_timestamp(x))
            df["timestamp"] = df["timestamp"].apply(lambda x: time_to_ts(x))
            # for i in range(250,475):
            service_list = []
            for service in services:
                service_df = df.loc[df["service"] == service]
                duration = int(labels.iloc[i]["duration"] / 60)
                if duration == 0:
                    duration = 1
                for k in range(duration):
                    st_time = labels.iloc[i]["st_time"]
                    st_array = datetime.strptime(st_time, "%Y-%m-%d %H:%M:%S,%f")
                    st_stamp = int(
                        time.mktime(st_array.timetuple()) * 1000.0
                        + st_array.microsecond / 1000.0
                    )
                    new_stamp = st_stamp + k * 60000 + 30000

                    temp_df = service_df.loc[
                        (service_df["timestamp"] >= (st_stamp + k * 60000))
                        & (service_df["timestamp"] < (st_stamp + (k + 1) * 60000))
                    ]
                    unique_list = np.unique(temp_df["EventId"], return_counts=True)
                    event_id = unique_list[0]
                    cnt = unique_list[1]

                    for t in range(len(event_id)):
                        str_cnt = str(cnt[t])
                        event = event_id[t] + "_" + str_cnt
                        service_list.append([new_stamp, service, event])
            logs_list.append(service_list)
        np.save(save_path, logs_list)

    @staticmethod
    def random_sampling(df, labels, save_path):
        logs_list = []
        for i in range(0, len(labels)):
            service_list = []
            for j in range(len(df)):
                temp_df = df[j].loc[
                    (df[j]["datetime"] >= labels["st_time"][i])
                    & (df[j]["datetime"] <= labels["ed_time"][i])
                ]
                if len(temp_df) == 0:
                    continue
                elif len(temp_df) < 11 and len(temp_df) > 0:
                    temp_df = temp_df
                elif len(temp_df) < 1001 and len(temp_df) >= 11:
                    temp_df = temp_df.sample(n=10)
                elif len(temp_df) < 2000 and len(temp_df) >= 1001:
                    tmp = int(len(temp_df) / 100)
                    temp_df = temp_df.sample(n=tmp)
                else:
                    temp_df = temp_df.sample(n=20)

                print(len(temp_df))
                for _, row in temp_df.iterrows():
                    st_time = row["datetime"]
                    st_array = datetime.strptime(st_time, "%Y-%m-%d %H:%M:%S,%f")
                    st_stamp = int(
                        time.mktime(st_array.timetuple()) * 1000.0
                        + st_array.microsecond / 1000.0
                    )
                    service_list.append([st_stamp, row["Service"], row["EventId"]])
            logs_list.append(service_list)
        np.save(save_path, logs_list)

    @staticmethod
    def stratified_sampling(labels, log_files, services, save_path):
        logs_list = []
        # for i in tqdm(range(len(labels))):
        for i in range(0, len(labels)):
            id = labels.iloc[i]["index"]
            df = pd.read_csv(f"{log_files}/{id}.csv")
            df["datetime"] = df["Content"].apply(lambda x: extract_timestamp(x))
            # for i in range(250,475):
            service_list = []
            for service in services:
                service_df = df.loc[df["service"] == service]
                unique_list = np.unique(service_df["EventId"], return_counts=True)
                event_id = unique_list[0]
                cnt = unique_list[1]
                for k in range(len(cnt)):
                    if cnt[k] == 1:
                        unique_time = service_df["datetime"].loc[
                            service_df["EventId"] == event_id[k]
                        ]
                        unique_array = datetime.strptime(
                            (list(unique_time))[0], "%Y-%m-%d %H:%M:%S,%f"
                        )
                        unique_stamp = int(
                            time.mktime(unique_array.timetuple()) * 1000.0
                            + unique_array.microsecond / 1000.0
                        )
                        service_list.append([unique_stamp, service, event_id[k]])
                        service_df = service_df.loc[
                            service_df["EventId"] != event_id[k]
                        ]
                X = service_df
                y = service_df["EventId"]
                if len(service_df) == 0:
                    continue
                elif len(service_df) < 21 and len(service_df) >= 1:
                    X_test = service_df
                else:
                    tmp = len(event_id)
                    if tmp < 20:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=tmp, stratify=y
                        )
                    else:
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, test_size=20, stratify=y
                        )
                # print(X_test)
                for _, row in X_test.iterrows():
                    st_time = row["datetime"]
                    st_array = datetime.strptime(st_time, "%Y-%m-%d %H:%M:%S,%f")
                    st_stamp = int(
                        time.mktime(st_array.timetuple()) * 1000.0
                        + st_array.microsecond / 1000.0
                    )
                    service_list.append([st_stamp, row["service"], row["EventId"]])
            logs_list.append(service_list)
        np.save(save_path, logs_list)

    def do_lab(self):
        # df_1 = np.load(pf.get_path(self.config['source_data_path'], 'df_1.npy'), allow_pickle=True)
        # df_2 = np.load(pf.get_path(self.config['source_data_path'], 'df_2.npy'), allow_pickle=True)
        # df_3 = np.load(pf.get_path(self.config['source_data_path'], 'df_3.npy'), allow_pickle=True)
        # df = list(df_1) + list(df_2) + list(df_3)
        # labels = pd.read_csv(pf.get_path(config['demo_path'], ))
        services = [
            "dbservice1",
            "mobservice2",
            "logservice1",
            "mobservice1",
            "logservice2",
            "dbservice2",
            "redisservice1",
            "webservice2",
            "webservice1",
            "redisservice2",
        ]
        path = "logs_faults"
        # df = pd.DataFrame()
        # for file in os.listdir(path):
        #     df_local = pd.read_csv(f"{path}/{file}")
        #     df.append(df_local)
        labels = pd.read_csv("data/gaia/test/anomalies/labels_converted.csv")
        # self.log_scale(labels, path, services, "log_scale.npy")
        # print("log_scale_finish")
        # self.random_sampling(df, labels, "random.npy")
        # print("random finish")
        self.stratified_sampling(
            labels, path, services, "data/gaia/test/anomalies/stratification.npy"
        )
        print("stratified finish")


if __name__ == "__main__":
    log = LogEvent()
    log.do_lab()
