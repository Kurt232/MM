import os
import sys
import collections
from datetime import datetime
import pandas as pd
import glob

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.requests import SampleUid
from lighteval.models.model_output import ModelResponse, GenerativeResponse

import logging
import argparse

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def run(pipeline: Pipeline, output_path: str, date_id: str):
    logger.info("--- RUNNING MODEL ---")
    task2sample_id_with_responses: dict[str, list[tuple[str, str, list, ModelResponse]]] = collections.defaultdict(list)
    try:
        for request_type, requests in pipeline.requests.items():
            logger.info(f"Running {request_type} requests")
            run_model = pipeline.model.get_method_from_request_type(request_type=request_type)
            responses = run_model(requests)

            for response, request in zip(responses, requests):
                metric_categories = [m.value for m in request.metric_categories]
                task2sample_id_with_responses[request.task_name].append((
                    request.sample_index, request.request_index, 
                    metric_categories, response))
    except Exception as e:
        logger.error(f"Error in inference: {e}")

    logger.info("--- SAVING RESPONSES ---")
    for task_name, sample_id2responses in task2sample_id_with_responses.items():
        response_list = []
        for (sample_id, request_id, metric_categories, response) in sample_id2responses:
            sample_uid = SampleUid(task_name=task_name, doc_id_seed=sample_id)
            doc = pipeline.docs[sample_uid]

            response_list.append(
                {   
                    # request info
                    "sample_id": sample_id, # str
                    "request_id": request_id, # str
                    "task_name": task_name, # str
                    "metric_categories": metric_categories, # list[str]
                    # doc info
                    "example": doc.query,
                    "instruction": doc.instruction,
                    "full_prompt": doc.ctx,
                    "num_effective_few_shots": doc.num_effective_few_shots,
                    "num_asked_few_shots": doc.num_asked_few_shots,
                    "choices": doc.choices,
                    "gold_index": doc.gold_index,
                    "specific": doc.specific,
                    
                    "gold": doc.get_golds(), # list
                    # response info
                    "predictions": response.get_result_for_eval(), # Any
                    "logits": response.logits if isinstance(response, GenerativeResponse) else None,
                    # "input_tokens": response.input_tokens,
                    # "generated_tokens": response.generated_tokens,
                    "input_tokens_count": len(response.input_tokens),
                    "generated_tokens_count": [len(o) for o in response.generated_tokens],
                    "truncated_tokens_count": response.truncated_tokens_count,
                    "padded_tokens_count": response.padded_tokens_count,
                }
            )
        
        df = pd.DataFrame(response_list, dtype=str)
        df = df.sort_values("sample_id")

        df.to_parquet(
            os.path.join(output_path, f"outputs_{task_name}_{date_id}.parquet"),
            index=False
        )

def check_existed_tasks(output_dir: str, _model_name:str, _timestamp:str, tasks: str):
    if _timestamp is None or tasks is None:
        return tasks
    output_dir = os.path.join(output_dir, "outputs", _model_name, _timestamp)
    if not os.path.exists(output_dir):
        return tasks
    tasks = tasks.split(",")
    existing_tasks = set(os.path.basename(f).replace(f"outputs_", "").replace(f"_{_timestamp}.parquet", "") for f in glob.glob(os.path.join(output_dir, "outputs_*.parquet")))
    target_tasks = []
    for task in tasks:
        short_task = task[:-2] # "mm|mmlu_pro_c|0|0" - > "mm|mmlu_pro_c|0"
        if short_task not in existing_tasks:
            target_tasks.append(task)
    
    if not target_tasks:
        return None
    return ",".join(target_tasks)

def main(args):
    output_dir = args.output_dir
    model_name = args.model_name
    _model_name = model_name.replace("/", "_")
    timestamp = args.timestamp

    if timestamp == "latest":
        path = f"{output_dir}/outputs/{_model_name}/*/"
        timestamps = glob.glob(path)
        if timestamps:
            timestamp = sorted(timestamps)[-1].split("/")[-2]
            logging.info(f"Latest timestamp: {timestamp}")
        else:
            logging.warning(f"No existing timestamps found for {_model_name}")
            timestamp = datetime.now().isoformat().replace(":", "-")
    elif timestamp is None:
        timestamp = datetime.now().isoformat().replace(":", "-")
    
    output_path = os.path.join(output_dir, "outputs", _model_name, timestamp)
    os.makedirs(output_path, exist_ok=True)
    date_id = timestamp

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
    )

    model_config = VLLMModelConfig(
        model_name=model_name,
        dtype="bfloat16",
        max_model_length=args.max_length,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        use_chat_template=False,
        generation_parameters= {
            "max_new_tokens": None,
            "temperature": 0.0,
        }
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        dataset_loading_processes=32,
        use_chat_template=False,
    )

    pipeline = Pipeline(
        tasks=None,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    ) # init the tasks

    timestamp = datetime.now()
    # Reasoning task
    tasks = """\
mm|gpqa_diamond_c|0|0,\
mm|aime24_c|0|0,\
mm|math_500_c|0|0,\
mm|gsm8k_c|0|0\
"""
    tasks = check_existed_tasks(output_dir, _model_name, date_id, tasks)
    if tasks:
        logger.info(f"Running inference for tasks: {tasks}")
        # pipeline.pipeline_parameters.max_samples=500
        pipeline.pipeline_parameters.stop_tokens = ["Problem", "problem"],
        pipeline._init_tasks_and_requests(tasks=tasks) # re-initialize the tasks in pipeline
        run(pipeline, output_path, date_id)

    # MCQ greedy search
    tasks = """\
mm|truthfulqa_c|0|0,\
mm|commonsenseqa_c|0|0,\
mm|arc_easy_c|0|0,\
mm|arc_challenge_c|0|0\
"""
    tasks = check_existed_tasks(output_dir, _model_name, date_id, tasks)
    if tasks:
        logger.info(f"Running inference for tasks: {tasks}")
        # pipeline.pipeline_parameters.max_samples=500
        pipeline.pipeline_parameters.stop_tokens = ["\n"]
        pipeline._init_tasks_and_requests(tasks=tasks) # re-initialize the tasks in pipeline
        run(pipeline, output_path, date_id)

    # MMLU Pro
    # tasks = "mm|mmlu_pro_c|0|0"
    tasks = None
    tasks = check_existed_tasks(output_dir, _model_name, date_id, tasks)
    if tasks:
        logger.info(f"Running inference for tasks: {tasks}")
        # pipeline.pipeline_parameters.max_samples=6000
        pipeline.pipeline_parameters.stop_tokens = ["\n"]
        pipeline._init_tasks_and_requests(tasks=tasks) # re-initialize the tasks in pipeline
        run(pipeline, output_path, date_id)
    
    
    time_delta = datetime.now() - timestamp
    # use HH:MM:SS format
    logger.info(f" Time Cost: {time_delta.seconds // 3600}:{(time_delta.seconds // 60) % 60:02}:{time_delta.seconds % 60:02}")
    
    pipeline.model.cleanup()
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference on various tasks')
    parser.add_argument('output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('model_name', type=str, help='Name of the model to use')
    parser.add_argument('--timestamp', type=str, default=None, 
                      help='Timestamp for output directory (default: current time, "latest" for most recent)')
    parser.add_argument('--max_length', type=int, default=4096,
                      help='Maximum length of LLM')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                      help='tensor parallel for vllm, default 1')
    args = parser.parse_args()
    main(args)