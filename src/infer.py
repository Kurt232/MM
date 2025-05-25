import os
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

def main():
    output_dir = "./results_test"
    evaluation_tracker = EvaluationTracker(
        output_dir=output_dir,
    )

    pipeline_params = PipelineParameters(
        launcher_type=ParallelismManager.VLLM,
        # use_chat_template=True,
    )

    model_config = VLLMModelConfig(
        model_name="./models/Llama3-8B",
        # use_chat_template=True,
        dtype="bfloat16",
        # tensor_parallel_size=2, 
        max_model_length=4096,
        gpu_memory_utilization=0.8,
        generation_parameters= {
            "max_new_tokens":4096,
        }
    )

    tasks = "helm|truthfulqa|0|0"

    pipeline = Pipeline(
        tasks=tasks,
        pipeline_parameters=pipeline_params,
        evaluation_tracker=evaluation_tracker,
        model_config=model_config,
    ) # init the tasks

    logger.info("--- RUNNING MODEL ---")
    task2sample_id_with_responses: dict[str, list[tuple[str, ModelResponse]]] = collections.defaultdict(list)
    try:
        for request_type, requests in pipeline.requests.items():
            logger.info(f"Running {request_type} requests")
            run_model = pipeline.model.get_method_from_request_type(request_type=request_type)
            responses = run_model(requests)

            for response, request in zip(responses, requests):
                task2sample_id_with_responses[request.task_name].append((request.sample_index, response))
    except Exception as e:
        logger.error(f"Error in inference: {e}")
    pipeline.model.cleanup()

    logger.info("--- SAVING RESPONSES ---")
    date_id = datetime.now().isoformat().replace(":", "-")
    for task_name, sample_id2responses in task2sample_id_with_responses.items():
        response_list = []
        for (sample_id, response) in sample_id2responses:
            sample_uid = SampleUid(task_name=task_name, doc_id_seed=sample_id)
            doc = pipeline.docs[sample_uid]

            response_list.append(
                {
                    "sample_id": sample_id, # str
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
                    "input_tokens": response.input_tokens,
                    "generated_tokens": response.generated_tokens,
                    "truncated_tokens_count": response.truncated_tokens_count,
                    "padded_tokens_count": response.padded_tokens_count,
                }
            )
        
        df = pd.DataFrame(response_list, dtype=str)
        df = df.sort_values("sample_id")
        output_path = os.path.join(output_dir, "outputs",
            pipeline.model.model_info.model_name,
            f"outputs_{task_name}_{date_id}.parquet")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_parquet(
            output_path,
            index=False
        )

        

if __name__ == "__main__":
    main()