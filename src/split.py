"""
split the 32k generation into 1k, 2k, 4k, 8k, 16k, 32k
"""
import pandas as pd
import os
import ast

data_dict = {
    "phi4": "._models_R-Phi4/2025-06-23T01-52-10.258150",
    "llama3": "._models_R1-Llama3-8B/2025-06-23T10-15-33.465228"
}

length_list = [x * 1024 for x in [1, 2, 4, 8, 16, 32]]
# todo:: current metrics can't save the sample id. We can't use the groudtruth to filter the data.
load_path = "./data/eval/0623_32k/outputs"
save_path = "./data"
rate = 0.5
seed = 42

def sample_and_save(model_name, dir_path):
    root = os.path.join(load_path, dir_path)
    for f in os.listdir(root):
        if not f.endswith(".parquet"):
            continue
        df = pd.read_parquet(os.path.join(root, f))

        df_copy = df.copy(True)
        df["generated_tokens_count"] = df["generated_tokens_count"].apply(lambda x: ast.literal_eval(x)[0])
        df["input_tokens_count"] = df["input_tokens_count"].astype(int)
        df["tokens_count"] = df["generated_tokens_count"] + df["input_tokens_count"]
        
        last_length = 0
        for length in length_list:
            condition = (df["tokens_count"] > last_length) & (df["tokens_count"] <= length)
            filtered_keys = df[condition][["sample_id", "request_id"]]
            # Merge with df_copy to retain all columns from the original dataframe
            target_df = pd.merge(filtered_keys, df_copy, on=["sample_id", "request_id"], how="inner")
            vali_df = target_df.sample(frac=rate, random_state=seed)
            test_df = target_df.drop(vali_df.index)

            vali_path = os.path.join(save_path, "vali", f"{int(last_length/1024)}-{int(length/1024)}k", "outputs", dir_path)
            os.makedirs(vali_path, exist_ok=True)
            vali_df.to_parquet(os.path.join(vali_path, f))

            test_path = os.path.join(save_path, "test", f"{int(last_length/1024)}-{int(length/1024)}k", "outputs", dir_path)
            os.makedirs(test_path, exist_ok=True)
            test_df.to_parquet(os.path.join(test_path, f))

            temp_path = os.path.join(save_path, "prepare", f"{int(last_length/1024)}-{int(length/1024)}k", "outputs", dir_path)
            os.makedirs(temp_path, exist_ok=True)
            target_df.to_parquet(os.path.join(temp_path,  f))

            last_length = length

        # outputs_mm|aime24|0_2025-06-23T01-52-10.258150.parquet
        task = f[8: -35]
        print(task)
        # task_dict[task] = df

if __name__ == '__main__':
    for k, v in data_dict.items():
        sample_and_save(k, v)