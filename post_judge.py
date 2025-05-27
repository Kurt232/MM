import pandas as pd
import ast
import json
import os

detail_path = "data/eval_judged/0524_4k/details/._models_Qwen2.5-14B/2025-05-25T19-37-04.422266"
data = pd.read_parquet(detail_path)

post_predictions = []
acc_dict = {}

# Iterate through rows in the batch DataFrame
for idx, item in data.iterrows():
    task = item["task"]
    if task not in acc_dict:
        acc_dict[task] = {
            "acc": 0,
            "tot": 0
        }
    gold_str: str = item["gold"]
    judged_str: str = item["judged"]
    predictions_str: str = item["predictions"]
    assert isinstance(gold_str, str) and isinstance(judged_str, str)
    gold: list[str] = ast.literal_eval(gold_str)
    judged: list[str] = ast.literal_eval(judged_str)
    predictions: list[str] = ast.literal_eval(predictions_str)
    # print(f"{idx}: {task}: {gold} -> {predictions} -> {judged}")
    assert len(gold) == 1, len(gold)
    assert len(judged) == len(predictions), f"{idx=}: {len(gold)=}, {len(predictions)=}"
    assert len(gold) <= len(judged), f"{idx=}: {len(gold)=}, {len(judged)=}"
    predictions = []

    acc_flag = 0 #! consider pass@sample
    for i, g in enumerate(gold):
        j = judged[i]
        think_idx = j.find("</think>\n\n")
        try:
            assert think_idx != -1 #! non expected format.
            j = j[think_idx + 10: ].strip()
            pred = json.loads(j)['extracted_option']
        except AssertionError:
            print(f"{idx}: format:{judged_str[-20:]}")
            pred = None
        except:
            print(f"{idx}: json:{judged_str[:20]}")
            pred = None
        g = g.strip()
        if pred:
            pred = pred.strip()
        else:
            pred = "X"
        predictions.append(pred)
        if g == pred:
            acc_flag = 1
    post_predictions.append(str(predictions))
    # print(f"{idx}: {task}: {gold} -> {predictions}")
    acc_dict[task]["acc"] += acc_flag
    acc_dict[task]["tot"] += 1

data["predictions"] = post_predictions
data.drop(columns=["judged"])

acc = {
    "all": [0, 0]
}
for task, group in data.groupby("task"):
    # data.to_parquet(f"{save_dir}/details_{task}_{timestamp}.parquet")
    _acc = acc_dict[task]["acc"]
    _tot = acc_dict[task]["tot"]
    
    acc['all'][0] += _acc
    acc['all'][1] += _tot

    # task = task.split(":")[0]
    acc[task]= [_acc, _tot]

    print(f"{task}: {_acc}/{_tot} = {_acc/_tot:.4f}")
print(f"all: {acc['all'][0]}/{acc['all'][1]} = {acc['all'][0]/acc['all'][1]:.4f}")



    
