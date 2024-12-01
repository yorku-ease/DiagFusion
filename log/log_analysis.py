import argparse
from split_big_file import split_file
from drain_work import drain_work
from concat_template import concat_template
from get_fault2templates import get_fault2templates
from get_all_templates_cmdbs import get_all_templates_cmdbs
import pandas as pd


def log_analysis(log_config):
    input_dir = log_config["input_dir"]
    fault_files = log_config["fault_files"]

    # # drain
    # split_file(input_dir, 128 * 1024 * 1024)  # 切分文件

    # drain_work(input_dir, 128 * 1024 * 1024)  # 使用drain提取

    # all_templates
    mid_dir = f"mid/{input_dir[5:]}/"
    # templates_file = f"{mid_dir}all_templates.csv"
    # if log_config["all_templates"] == None:
    #     concat_template(mid_dir, templates_file)
    # else:
    #     print("[read config] all_templates:", log_config["all_templates"])

    # fault2templates
    if log_config["fault2templates"] == None:
        if fault_files is None or len(fault_files) == 0:
            raise Exception("no fault files")
        for fault_file in fault_files:
            get_fault2templates(fault_file, mid_dir)
    # else:
    #     print("[read config] fault2templates:", log_config["fault2templates"])

    # all_templates_cmdbs
    # if log_config["all_templates_cmdbs"] == None:
    #     get_all_templates_cmdbs(mid_dir, f"{mid_dir}all_templates_cmdbs.csv")
    # else:
    #     print("[read config] all_templates_cmdbs:", log_config["all_templates_cmdbs"])


if __name__ == "__main__":
    log_config = {
        "input_dir": "GAIA-DataSet-main/MicroSS/business",
        "fault_files": ["data/gaia/demo/demo_1100/labels.csv"],
        "fault2templates": None,
        "all_templates": None,
        "all_templates_cmdbs": None,
    }
    log_analysis(log_config=log_config)
