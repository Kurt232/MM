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
                    "input_tokens_count": len(response.input_tokens) if response.input_tokens is not None else -1,
                    "generated_tokens_count": len(response.generated_tokens) if response.generated_tokens is not None else -1,
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

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        dataset_loading_processes=32,
        max_samples=100, # for mmlu
        use_chat_template=args.use_chat_template
    )

    model_config = VLLMModelConfig(
        model_name=model_name,
        dtype="bfloat16",
        max_model_length=args.max_length,
        gpu_memory_utilization=0.9,
        use_chat_template=args.use_chat_template,
        generation_parameters= {
            "max_new_tokens":4096,
            "temperature": 0.0,
        }
    )

    timestamp = datetime.now()
    tasks = "helm|mmlu|0|0"
    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    ) # init the tasks

    logger.info(f"Running inference for tasks: {tasks}")
    run(pipeline, output_path, date_id)

    # MCQ greedy search
    tasks = """\
helm|truthfulqa|0|0,\
helm|commonsenseqa|0|0,\
lighteval|arc:easy|0|0,\
leaderboard|arc:challenge|0|0\
"""
    logger.info(f"Running inference for tasks: {tasks}")
    pipeline.pipeline_parameters.max_samples=500
    pipeline._init_tasks_and_requests(tasks=tasks) # re-initialize the tasks in pipeline
    run(pipeline, output_path, date_id)

    # Generative task (Reasoning)
    tasks = """\
lighteval|gpqa:diamond|0|0,\
lighteval|aime24|0|0,\
lighteval|math_500|0|0,\
lighteval|gsm8k|0|0,\
extended|lcb:codegeneration_release_v6|0|0\
"""
    logger.info(f"Running inference for tasks: {tasks}")
    pipeline.pipeline_parameters.max_samples=500
    pipeline._init_tasks_and_requests(tasks=tasks) # re-initialize the tasks in pipeline
    pipeline.model._config.generation_parameters.temperature = 0.6
    pipeline.model._config.generation_parameters.top_p = 0.95
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
    parser.add_argument('--max_length', type=int, default=2048,
                      help='Maximum length of LLM')
    parser.add_argument('--use_chat_template', action='store_true',
                      help='Use chat template for Instruction Models')
    args = parser.parse_args()
    main(args)