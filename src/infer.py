import os
import sys
import collections
from datetime import datetime
import pandas as pd

from lighteval.logging.evaluation_tracker import EvaluationTracker
from lighteval.models.vllm.vllm_model import VLLMModelConfig
from lighteval.pipeline import ParallelismManager, Pipeline, PipelineParameters
from lighteval.tasks.requests import SampleUid
from lighteval.models.model_output import ModelResponse, GenerativeResponse

import logging

logger = logging.getLogger(__name__)

def run(pipeline: Pipeline, output_dir: str, date_id: str):
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
                    "truncated_tokens_count": response.truncated_tokens_count,
                    "padded_tokens_count": response.padded_tokens_count,
                }
            )
        
        df = pd.DataFrame(response_list, dtype=str)
        df = df.sort_values("sample_id")
        output_path = os.path.join(output_dir, "outputs",
            pipeline.model.model_info.model_name,
            date_id,
            f"outputs_{task_name}_{date_id}.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(
            output_path,
            index=False
        )

def main():
    assert len(sys.argv) == 3
    output_dir = sys.argv[1]
    model_name = sys.argv[2]

    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        dataset_loading_processes=32,
        max_samples=100, # for mmlu
    )

    model_config = VLLMModelConfig(
        model_name=model_name,
        dtype="bfloat16",
        max_model_length=4096,
        gpu_memory_utilization=0.9,
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

    date_id = datetime.now().isoformat().replace(":", "-")
    logger.info(f"Running inference for tasks: {tasks}")
    run(pipeline, output_dir, date_id)

    # MCQ greedy search
    tasks = """\
helm|truthfulqa|0|0,\
helm|commonsenseqa|0|0,\
helm|openbookqa|0|0,\
lighteval|arc:easy|0|0,\
leaderboard|arc:challenge|0|0\
"""
    pipeline.pipeline_parameters.max_samples=500
    pipeline._init_tasks_and_requests(tasks=tasks) # re-initialize the tasks in pipeline
    run(pipeline, output_dir, date_id)

    # Generative task (Reasoning)
    tasks = """\
lighteval|gpqa:diamond|0|0,\
lighteval|aime24|0|0,\
lighteval|math_500|0|0,\
lighteval|gsm8k|0|0,\
extended|lcb:codegeneration_release_v6|0|0\
"""
    pipeline.pipeline_parameters.max_samples=500
    pipeline._init_tasks_and_requests(tasks=tasks) # re-initialize the tasks in pipeline
    pipeline.model._config.generation_parameters.temperature = 0.6
    pipeline.model._config.generation_parameters.top_p = 0.95
    run(pipeline, output_dir, date_id)

    
    time_delta = datetime.now() - timestamp
    # use HH:MM:SS format
    logger.info(f" Time Cost: {time_delta.seconds // 3600}:{(time_delta.seconds // 60) % 60:02}:{time_delta.seconds % 60:02}")
    
    pipeline.model.cleanup()
        

if __name__ == "__main__":
    main()