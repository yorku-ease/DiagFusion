import os
import pandas as pd
from get_fault2templates import extract_timestamp

def get_all_templates_cmdbs(drain_result_dir, output_file):
    print('[get_all_templates_cmdbs]')
    drain_result = []
    for root, dirs, files in os.walk(drain_result_dir):
        for file in files:
            if root.endswith('drain_result') and 'structured' in file:
                drain_result.append(os.path.join(root, file))

    df = pd.DataFrame(columns=['templates_cmdbs'])
    for file in drain_result:
        print(file)
        t = pd.read_csv(file)
        if t.iloc[0]["Content"] == "message":
            t = t.drop([0])
        t["timestamp"] = t["Content"].apply(lambda x: extract_timestamp(x))
        t["timestamp"] = pd.to_datetime(t["timestamp"], format='%Y-%m-%d %H:%M:%S,%f')
        tdf = pd.DataFrame((t['EventTemplate'] + '_' + t['service']).unique(), columns=['templates_cmdbs'])
        df = df.append(tdf, ignore_index=True)

    ans = pd.DataFrame(df['templates_cmdbs'].unique(), columns=['templates_cmdbs'])
    ans.to_csv(output_file)
    print(f'[get_all_templates_cmdbs done!] output: {output_file}')