import pandas as pd
import os
root="data/eval/0619_16k/outputs/._models_R-Phi4/2025-06-19T17-24-10.004059"
model_name="phi4"
output_path = f"../mllm/raw_data/{model_name}" # re-inference
if not os.path.exists(output_path):
    os.makedirs(output_path)

len_dict = {
	"mm|aime24|0": 0, # loo less
	"mm|arc_challenge|0": 200,
	"mm|arc_easy|0": 200,
	"mm|commonsenseqa|0": 200,
	"mm|gpqa_diamond|0": 0, # too hard
	"mm|gsm8k|0": 100,
	"mm|math_500|0": 100,
	"mm|truthfulqa|0": 200,
	"mm|mmlu_pro|0": 200, # wait to generation
}
# sum: 1000

df = pd.concat([pd.read_parquet(os.path.join(root, f)) for f in os.listdir(root) if f.endswith(".parquet")])
import ast
df["generated_tokens_count"] = df["generated_tokens_count"].apply(lambda x: ast.literal_eval(x)[0])
df["gold"] = df["gold"].apply(lambda x: ast.literal_eval(x)[0])
df["predictions"] = df["predictions"].apply(lambda x: ast.literal_eval(x)[0])
df["input_tokens_count"] = df["input_tokens_count"].astype(int)

df["gold_format"] = df["predictions"].apply(lambda x: x.startswith("<think>") and "</think>" in x)

for task_name, task_df in df.groupby("task_name"):
    print(task_name)
    length = len_dict.get(task_name, 100)
    if length == 0:
        continue
    task_df = task_df[task_df["gold_format"]]
    res = task_df[["full_prompt", "predictions", "generated_tokens_count", "input_tokens_count", "gold", "example"]].sort_values(by="generated_tokens_count", ascending=True).head(length)
    res.to_json(f"{output_path}/{task_name}.json", orient="records", indent=4, force_ascii=False)
    print(f"Saved {len(res)} examples.")