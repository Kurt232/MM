import os
import collections
import json
import pandas as pd
import ast
import sys
import glob

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.requests import SampleUid
from lighteval.tasks.lighteval_task import LightevalTask
from lighteval.models.model_output import ModelResponse, GenerativeResponse
from lighteval.metrics import MetricCategory
from lighteval.utils.utils import make_results_table

import logging

logger = logging.getLogger(__name__)

def load_responses(response_dir, timestamp, current_task_set):
    logger.debug("--- LOAD RESPONSES ---")
    if not os.path.exists(response_dir):
        raise FileNotFoundError(f"Response directory {response_dir} does not exist.")
    sample_id_to_responses: dict[(SampleUid, MetricCategory), list[ModelResponse]] = collections.defaultdict(list)
    for file_name in os.listdir(response_dir):
        task_name = file_name.replace("outputs_", "").replace(f"_{timestamp}.parquet", "")
        if task_name not in current_task_set:
            continue

        data = pd.read_parquet(os.path.join(response_dir, file_name))
        for _, row in data.iterrows():
            metric_categories: list[str] = ast.literal_eval(row["metric_categories"])
            sample_id = row["sample_id"]
            sample_uid = SampleUid(task_name=task_name, doc_id_seed=sample_id)

            pred = ast.literal_eval(row["predictions"])
            logits = ast.literal_eval(row["logits"]) if row["logits"] is not None else None
            truncated_tokens_count = int(row["truncated_tokens_count"])
            padded_tokens_count = int(row["padded_tokens_count"])

            if logits is None:
                response = ModelResponse(
                    result=pred,
                    truncated_tokens_count=truncated_tokens_count,
                    padded_tokens_count=padded_tokens_count,
                )
            else:
                response = GenerativeResponse(
                    result=pred,
                    logits=logits,
                    truncated_tokens_count=truncated_tokens_count,
                    padded_tokens_count=padded_tokens_count,
                )
            
            for metric_category in metric_categories:
                sample_id_to_responses[(sample_uid, MetricCategory(metric_category))].append(response)
    return sample_id_to_responses

def main():
    assert len(sys.argv) == 4
    output_dir = sys.argv[1]
    model_name = sys.argv[2]
    timestamp = sys.argv[3]

    _model_name = model_name.replace("/", "_")
    if timestamp == "latest":
        path = f"{output_dir}/outputs/{_model_name}/*/"
        timestamps = glob.glob(path)
        timestamp = sorted(timestamps)[-1].split("/")[-2]
        logging.info(f"Latest timestamp: {timestamp}")

    response_dir = f"{output_dir}/outputs/{_model_name}/{timestamp}"
    assert os.path.exists(response_dir), response_dir

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        # use_chat_template=True,
        dataset_loading_processes= 32
    )

    model_config = VLLMModelConfig(
        model_name=model_name,
        # use_chat_template=True,
        dtype="bfloat16",
        max_model_length=4096,
        gpu_memory_utilization=0.8,
        generation_parameters= {
            "max_new_tokens":4096,
        }
    )

    tasks = """\
mm|mmlu_pro_c|0|0,\
helm|truthfulqa|0|0,\
helm|commonsenseqa|0|0,\
mm|arc_c:easy|0|0,\
mm|arc_c:challenge|0|0,\
mm|gpqa_c:diamond|0|0,\
mm|aime24|0|0,\
mm|math_500|0|0,\
mm|gsm8k_c|0|0\
"""
# extended|lcb:codegeneration_release_v6|0|0\

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    ) # init the tasks
    
    date_id = timestamp
    current_task_set = set()
    for task in pipeline.task_names_list:
        for few_shot in pipeline.fewshot_dict[task]:
            current_task_set.add(f"{task}|{few_shot[0]}")

    sample_id_to_responses = load_responses(response_dir, date_id, current_task_set)
    logger.debug("--- COMPUTING METRICS ---")
    task_metric_category_groups = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(list))
    )

    for (sample_id, metric_category), sample_responses in sample_id_to_responses.items():
        task_metric_category_groups[sample_id.task_name][metric_category]["ids"].append(sample_id.doc_id_seed)
        task_metric_category_groups[sample_id.task_name][metric_category]["responses"].append(sample_responses)
        task_metric_category_groups[sample_id.task_name][metric_category]["docs"].append(pipeline.docs[sample_id])

    for task_name, samples_per_metric in task_metric_category_groups.items():
        task: LightevalTask = pipeline._get_task(task_name)

        for metric_category, samples in samples_per_metric.items():
            sample_ids = samples["ids"]
            responses = samples["responses"]
            docs = samples["docs"]
            metric_function = task.get_metric_method_from_category(metric_category=metric_category)
            metric_category_metrics = [metric for metric in task.metrics if metric.category == metric_category]

            outputs = metric_function(
                sample_ids=sample_ids,
                responses=responses,
                formatted_docs=docs,
                metrics=metric_category_metrics,
            )

            for output in outputs:
                pipeline.evaluation_tracker.metrics_logger.log(task_name, output)

    pipeline.evaluation_tracker.metrics_logger.aggregate(pipeline.task_dict)

    results: dict = dict(pipeline.evaluation_tracker.metrics_logger.metric_aggregated)

    versions = pipeline.evaluation_tracker.versions_logger.versions
    results_dict = {"results": results, "versions": versions}
    logger.debug("--- SAVING RESULTS ---")
    save_dir = f"{output_dir}/results/{_model_name}"
    os.makedirs(save_dir, exist_ok=True)
    json.dump(results_dict, open(f"{save_dir}/results_{date_id}.json", "w"), indent=2, ensure_ascii=False)

    pipeline.model.cleanup()

    logger.debug("--- SHOWING RESULTS ---")
    md_table = make_results_table(results_dict)
    print(md_table)

if __name__ == "__main__":
    main()