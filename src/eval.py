import pandas as pd
import ast
import os
import os.path as osp
from tqdm import tqdm
import logging
import argparse
import glob

# Configure logging to include line number and filename
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s[%(filename)s:%(lineno)d]: %(message)s"
)
logger = logging.getLogger(__name__)

from lighteval.metrics.metrics import PassAtK, SimpleEvalMatch, normalize_math_expression, compare_gold_target, ExactMatches

math_tasks = ["mm|aime24|0", "mm|math_500|0", "mm|gsm8k|0", "mm|aime24_c|0", "mm|math_500_c|0", "mm|gsm8k_c|0"]
mcq_instruct_tasks = ["mm|mmlu_pro|0", "mm|truthfulqa|0", "mm|commonsenseqa|0", "mm|arc_easy|0", "mm|arc_challenge|0", "mm|gpqa_diamond|0"]
mcq_completion_tasks = ["mm|mmlu_pro_c|0", "mm|truthfulqa_c|0", "mm|commonsenseqa_c|0", "mm|arc_easy_c|0", "mm|arc_challenge_c|0", "mm|gpqa_diamond_c|0"]

def eval(task_name: str, golds: list, preds: list) -> pd.DataFrame:
    if task_name in math_tasks:
        metric_func = PassAtK(
            k=1,
            n=1,
            strip_strings=True,
            normalize_gold=normalize_math_expression,
            normalize_pred=normalize_math_expression,
            # Uses sympy for comparison
            sample_scoring_function=compare_gold_target,
        ).compute
    elif task_name in mcq_instruct_tasks:
        metric_func = SimpleEvalMatch().compute
    elif task_name in mcq_completion_tasks:
        metric_func = ExactMatches(strip_strings=True, type_exact_match="prefix").compute
    else:
        raise ValueError(f"Unsupported task type: {task_name}")

    results = []
    for gold, pred in tqdm(zip(golds, preds)):
        results.append(metric_func(gold, pred) == 1)
    
    return results

def eval_task(output_path, file_name):
    logger.info(f"Evaluating {file_name}...")
    file_path = osp.join(output_path, file_name)
    task_name = file_name[8: -35] # "outputs_mm|aime24|0_2025-06-26T21-33-22.888531.parquet"
    
    df = pd.read_parquet(file_path)
    if "eval" in df.columns:
        logger.info(f"Skipping {file_name} as it has already been evaluated.")
        return
    golds = df["gold"].apply(lambda x: ast.literal_eval(x)).tolist()
    preds = df["predictions"].apply(lambda x: ast.literal_eval(x)).tolist()

    try:
        results = eval(task_name, golds, preds)
    except Exception as e:
        logger.error(f"Error evaluating {file_name}: {e}")
        raise e

    df["eval"] = results

    df.to_parquet(file_path, index=False)

def main(args):
    output_dir = args.output_dir
    model_name = args.model_name
    _model_name = model_name.replace("/", "_")
    timestamp = args.timestamp

    path = f"{output_dir}/outputs/{_model_name}/*/"
    timestamps = glob.glob(path)
    if timestamps is None or len(timestamps) == 0:
        logger.error(f"No timestamps found in {path}.")
        return

    if timestamp == "latest":
        timestamp = sorted(timestamps)[-1].split("/")[-2]
        logger.info(f"Latest timestamp: {timestamp}")
    elif not any(timestamp in ts for ts in timestamps):
        logger.error(f"Timestamp {timestamp} not found in {path}.")
        return
    
    output_path = os.path.join(output_dir, "outputs", _model_name, timestamp)

    records = [f for f in os.listdir(output_path) if f.endswith(".parquet")]

    for file_name in records:
        try:
            eval_task(output_path, file_name)
        except Exception as e:
            # Optionally log debug info here if needed.
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on various tasks')
    parser.add_argument('output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('model_name', type=str, help='Name of the model to use')
    parser.add_argument('timestamp', type=str, default='latest',
                        help='Timestamp for output directory (default: current time, "latest" for most recent)')
    args = parser.parse_args()
    main(args)