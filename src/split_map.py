"""
split the 32k generation into 1k, 2k, 4k, 8k, 16k, 32k
to use the reasoning data sample id to map the base model data
"""
import pandas as pd
import os
import ast

data_dict = {
    "._models_R-Phi4/2025-06-23T01-52-10.258150": "._models_I-Phi4/2025-06-26T21-33-22.888531",
    # llama reasoning to map to llama base
}

length_list = [x * 1024 for x in [1, 2, 4, 8, 16, 32]]
# todo:: current metrics can't save the sample id. We can't use the groudtruth to filter the data.
load_path = "./data/eval/0623_32k/outputs/"
save_path = "./data"
rate = 0.5
seed = 42

def sample_and_save(reasoning_path, target_path):
    timestamp = reasoning_path.split("/")[-1]
    timestamp1 = target_path.split("/")[-1]
    root = os.path.join(load_path, target_path)
    for f in os.listdir(root):
        if not f.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(root, f))
        for split in ["vali", "test", "prepare"]:
            last_length = 0
            for length in length_list:
                split_path = os.path.join(save_path, split, f"{int(last_length/1024)}-{int(length/1024)}k", "outputs", reasoning_path)
                df1 = pd.read_parquet(os.path.join(split_path, f.replace(timestamp1, timestamp)))
                # Merge on sample_id and request_id, but keep the samples from df
                target_df = pd.merge(df, df1[['sample_id', 'request_id']], on=["sample_id", "request_id"], how="inner")
                temp_path = os.path.join(save_path, split, f"{int(last_length/1024)}-{int(length/1024)}k", "outputs", target_path)
                os.makedirs(temp_path, exist_ok=True)
                target_df.to_parquet(os.path.join(temp_path, f))
                last_length = length

        # outputs_mm|aime24|0_2025-06-23T01-52-10.258150.parquet
        task = f[8: -35]
        print(task)
        # task_dict[task] = df

if __name__ == '__main__':
    for k, v in data_dict.items():
        sample_and_save(k, v)