from transforms.events import (
    metric_trace_log_parse,
    fasttext_with_DA,
    sententce_embedding,
)
from models import He_DGL
from public_function import deal_config, get_config
import os
import pandas as pd


if __name__ == "__main__":
    label_path = "data/gaia/test/anomalies/labels_converted.csv"
    labels = pd.read_csv(label_path, index_col=0)

    print("[parse]")
    metric_trace_log_parse.run_parse(labels)

    print("[fasttext]")
    fasttext_with_DA.run_fasttext(labels)

    print("[sentence_embedding]")
    sententce_embedding.run_sentence_embedding()

    print("[dgl]")
    lab_id = 9  # 实验唯一编号
    He_DGL.UnircaLab().do_lab(lab_id)
