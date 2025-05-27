import os
import sys
import pandas as pd
from tqdm import tqdm
import ast
import logging
import glob

from vllm import LLM, SamplingParams, RequestOutput
from vllm.transformers_utils.tokenizer import get_tokenizer

logger = logging.getLogger(__name__)

assert len(sys.argv) == 4
tasks = sys.argv[1]
output_dir = sys.argv[2]
model_name = sys.argv[3]
timestamp = sys.argv[4]

_model_name = model_name.replace("/", "_")
if timestamp == "latest":
    path = f"{output_dir}/outputs/{_model_name}/*/"
    timestamps = glob.glob(path)
    timestamp = sorted(timestamps)[-1].split("/")[-2]
    logging.info(f"Latest timestamp: {timestamp}")

saved_obj = "details" # "outputs"
details_path = f"{output_dir}/{_model_name}/{timestamp}"

if not os.path.exists(details_path):
    raise ValueError(f"Details path {details_path} does not exist.")

task_list = []
for task in tasks.split(","):
    task = task.strip()
    if task:
        temp = task.split('|')
        if (len(temp) == 3 or len(temp) == 4) and temp[2].isdigit(): # helm|mmlu:abstract_algebra|0 or helm|mmlu:abstract_algebra|0|0
            # todo:: to check the num_sample number
            # task_list.append(('|'.join(temp[0:2]), temp[2])) # helm|mmlu:abstract_algebra
            task_list.append('|'.join(temp[0:2]))
task_dict: dict = {task: False for task in task_list}

def prepare_data(details_path, timestamp, task_dict):
    if not os.path.exists(details_path):
        raise ValueError(f"Details path {details_path} does not exist.")

    data_list = []
    for d in os.listdir(details_path):
        data = pd.read_parquet(os.path.join(details_path, d))
        task = d.replace(f"{saved_obj}_", "").replace(f"_{timestamp}.parquet", "")
        data["task"] = task
        for task_name in task_dict.keys():
            if task.startswith(task_name):
                task_dict[task_name] = True
                data_list.append(data)
                break

    data = pd.concat(data_list, ignore_index=True)
    not_found_tasks = [k for k, v in task_dict.items() if not v]
    if not_found_tasks:
        logger.warning(f"Not found tasks: {', '.join(not_found_tasks)}")
    return data

judge_model_name="models/Qwen3-30B-A3B"
max_model_len=8192

model = LLM(
    model=judge_model_name,
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=max_model_len,
)

_tokenizer = get_tokenizer(
        judge_model_name,
        tokenizer_mode="auto",
    )
_tokenizer.pad_token = _tokenizer.eos_token
tokenizer = _tokenizer

sampling_params = SamplingParams(
    max_tokens=max_model_len,
    # # non-thinking
    # temperature=0.7,
    # top_p=0.8,
    # top_k=20,
    # min_p=0,
    # thinking,
    temperature=0.6,
    top_p=0.95,
    top_k=20,
    min_p=0,
)

def prepare_prompt(options, response, enable_thinking=None):
    """
    only support Qwen models
    """
    judge_prompt = f"""<|im_start|>system
You are an expert evaluator for multiple-choice question answering.<|im_end|>
<|im_start|>user
Your task is to strictly analyze the given Response and determine if it correctly identifies one of the valid options for the multiple-choice question. 
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

**Options**:
{options}

**Response**:
{response}<|im_end|>
<|im_start|>assistant
"""
    return judge_prompt + ("" if enable_thinking is None or enable_thinking else "<think>\n\n</think>\n\n")

def model_generate(model: LLM, batch: pd.DataFrame) -> pd.DataFrame:
    """
    only for helm MMLU
    """
    prompts = []
    prompt_idx = 0
    idx_list = []
    
    # Iterate through rows in the batch DataFrame
    for _, item in batch.iterrows():
        question: str = item["example"]
        predictions: str = item["predictions"]

        option_start = question.find("A.")
        option_end = question.find("Answer:", option_start)
        assert option_start != -1 and option_end != -1, f"Invalid question format: {question}"
        options = question[option_start:option_end]

        # logging.info(predictions)
        predictions = ast.literal_eval(predictions)
        _idx_list = []
        for resp in predictions:
            # assume the resp too long
            length = len(resp)
            if length > max_model_len:
                resp = resp[-max_model_len:]
            prompts.append(prepare_prompt(options, resp))
            _idx_list.append(prompt_idx)
            prompt_idx += 1
        idx_list.append(_idx_list)

    assert len(prompts) > 0, "No prompts generated"  # Fixed assertion (was checking for empty)

    responses: list[RequestOutput] = model.generate(
        prompts=prompts,
        sampling_params=sampling_params,
        )

    # Map responses back to the original batch
    judged_responses = []
    for e in idx_list:
        judged = [responses[idx].outputs[0].text for idx in e]
        judged_responses.append(str(judged))
    
    batch = batch.copy()  # Avoid SettingWithCopyWarning
    batch["judged"] = judged_responses
    return batch

data: pd.DataFrame = prepare_data(details_path, timestamp, task_dict)
result_df = model_generate(model, data)

output_path = os.path.join(output_dir, f"{saved_obj}_judge", model_name, timestamp)
if not os.path.exists(output_path):
    os.makedirs(output_path)

# save results per task
for task, data in tqdm(result_df.groupby("task")):
    data.to_parquet(os.path.join(output_path, f"{saved_obj}_{task}_{timestamp}.parquet"))
