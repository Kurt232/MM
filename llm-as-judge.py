import os
import sys
import glob
import pandas as pd
from tqdm import tqdm
import ast
import logging

from vllm import LLM, SamplingParams, RequestOutput
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

# output_dir = "results"
# model_name = "._models_Qwen2.5-14B"
# timestamp = "2025-05-16T01-17-57.441000"

assert len(sys.argv) == 4
output_dir = sys.argv[1]
model_name = sys.argv[2]
timestamp = sys.argv[3]

if timestamp == "latest":
    path = f"{output_dir}/details/{model_name}/*/"
    timestamps = glob.glob(path)
    timestamp = sorted(timestamps)[-1].split("/")[-2]
    logging.info(f"Latest timestamp: {timestamp}")

def prepare_data(output_dir, model_name, timestamp):
    details_path = f"{output_dir}/details/{model_name}/{timestamp}/"

    if not os.path.exists(details_path):
        raise ValueError(f"Details path {details_path} does not exist.")

    data_list = []
    for d in os.listdir(details_path):
        data = pd.read_parquet(os.path.join(details_path, d))
        task = d.replace("details_", "").replace(f"_{timestamp}.parquet", "")
        data["task"] = task
        data_list.append(data)

    data = pd.concat(data_list, ignore_index=True)

    return data

class VLLMModel():
    def __init__(self, 
        model_name: str,
        max_model_len: int):
        
        self.max_model_len = max_model_len

        # os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        self.model = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.9,
            tensor_parallel_size=2,
            # enable_reasoning=True,
            # reasoning_parser="qwen3",
            max_model_len=self.max_model_len,
        )

        _tokenizer = get_tokenizer(
                model_name,
                tokenizer_mode="auto",
            )
        _tokenizer.pad_token = _tokenizer.eos_token
        self.tokenizer = _tokenizer

        self.sampling_params = SamplingParams(
            max_tokens=max_model_len,
            # non-thinking
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0
        )
    
    def generate(self, prompts: list[str], max_new_tokens=None, min_new_tokens=None):
        """
        vllm engine can't truncate the prompt correctly
        """
        tokenized = self.tokenizer(prompts, add_special_tokens=True)
        inputs = tokenized["input_ids"]

        if min_new_tokens is not None:
            self.sampling_params.min_tokens = min_new_tokens
        if max_new_tokens is not None:
            self.sampling_params.max_tokens = max_new_tokens

        #! left truncate the inputs to the maximum length
        if max_new_tokens:
            max_context_size = self.max_model_len - max_new_tokens
        elif min_new_tokens:
            max_context_size = self.max_model_len - min_new_tokens
        else:
            max_context_size = self.max_model_len
        
        # Validate context size
        if max_context_size <= 0:
            raise ValueError("Context size <= 0. Adjust max/min_new_tokens or model max length.")

        inputs = [input[-max_context_size:] for input in inputs]

        responses: list[RequestOutput] = self.model.generate(
            prompt_token_ids=inputs,
            sampling_params=self.sampling_params,
            )
        return responses


def prepare_prompt(question, response, enable_thinking=None):
    """
    only support Qwen models
    """
    judge_prompt = f"""<|im_start|>system
You are an expert evaluator for multiple-choice question answering. Your task is to strictly analyze the given Response and determine if it correctly identifies one of the valid options for the multiple-choice question. 
Follow these evaluation rules:
1. Extract the first explicit option choice mentioned in the Response
2. The option must:
   - Appear as a standalone capital letter
   - Be preceded by "A.", "B.", etc. or clearly indicated as "Answer: X"
   - Match exactly one of the provided options
3. If no valid option is found:
   - Check if the Response contains "correct answer is [X]" pattern
   - Check for paraphrased but unambiguous identifications
4. Reject the Response if:
   - It provides multiple conflicting options
   - The selected option isn't among the provided choices
   - The justification contradicts the chosen option
   - No option can be reasonably inferred

Output format (JSON):
{{
  "extracted_option": "X" | null,
  "is_valid": bool,
}}

**Question**:
{question}

**Answer**:
{response}<|im_end|>
<|im_start|>assistant
"""
    return judge_prompt + ("" if enable_thinking is None or enable_thinking else "<think>\n\n</think>\n\n")

def model_generate(model: VLLMModel, batch: pd.DataFrame) -> pd.DataFrame:
    prompts = []
    prompt_idx = 0
    idx_list = []
    
    # Iterate through rows in the batch DataFrame
    for _, item in batch.iterrows():
        question: str = item["example"]
        
        predictions: str = item["predictions"]
        assert isinstance(predictions, str) #? is strange
        # logging.info(predictions)
        predictions = ast.literal_eval(predictions)
        _idx_list = []
        for resp in predictions:
            # logging.info(resp) # has []
            # exit()
            prompts.append(prepare_prompt(question, resp))
            _idx_list.append(prompt_idx)
            prompt_idx += 1
        idx_list.append(_idx_list)

    assert len(prompts) > 0, "No prompts generated"  # Fixed assertion (was checking for empty)

    responses = model.generate(
        prompts=prompts,
        min_new_tokens=100
    )
    # Map responses back to the original batch
    judged_responses = []
    for e in idx_list:
        judged = [[output.text for output in responses[idx].outputs] for idx in e]
        judged_responses.append(str(judged))
    
    batch = batch.copy()  # Avoid SettingWithCopyWarning
    batch["judged"] = judged_responses
    return batch

# model = VLLMModel(model_name="models/Qwen3-30B-A3B", max_model_len=32768)
model = VLLMModel(model_name="models/Qwen2.5-32B-Instruct", max_model_len=32768)
data: pd.DataFrame = prepare_data(output_dir, model_name, timestamp)
bs = 2048

# Process in batches and collect results
batch_list = []
for i in tqdm(range(0, len(data), bs)):
    batch = data.iloc[i:i+bs].copy()  # Explicit copy to avoid warnings
    processed_batch = model_generate(model, batch)
    batch_list.append(processed_batch)

# Combine results and save
result_df = pd.concat(batch_list)
save_path = f"{output_dir}_judged/{model_name}"
os.makedirs(save_path, exist_ok=True)
result_df.to_parquet(f"{save_path}/{timestamp}.parquet")
