# infer from output data.
import pandas as pd
import os
import glob

from vllm import LLM, SamplingParams

import logging

def argparse():
    import argparse
    parser = argparse.ArgumentParser(description="Inference script for LLMs.")
    parser.add_argument('output_dir', type=str, help='Directory to save outputs')
    parser.add_argument('model_name', type=str, help='Name of the model to load')
    parser.add_argument('target_model_name', type=str, help='Name of the model to use')
    parser.add_argument('--save_dir', type=str, default=None,
                      help='Directory to save outputs (default: output_dir)')
    parser.add_argument('timestamp', type=str, 
                      help='Timestamp for output directory (default: current time, "latest" for most recent)')
    parser.add_argument('--max_length', type=int, default=4096,
                      help='Maximum length of LLM')
    parser.add_argument('--tensor_parallel_size', type=int, default=1,
                      help='tensor parallel for vllm, default 1')
    return parser.parse_args()

if __name__ == "__main__":
    args = argparse()

    output_dir = args.output_dir
    model_name = args.model_name
    target_model_name = args.target_model_name
    max_length = args.max_length
    if args.save_dir is not None:
        save_dir = args.save_dir
    else:
        save_dir = output_dir

    _model_name = model_name.replace("/", "_")
    _target_model_name = target_model_name.replace("/", "_")

    timestamp = args.timestamp
    if timestamp == "latest":
        path = f"{output_dir}/outputs/{_model_name}/*/"
        timestamps = glob.glob(path)
        if timestamps:
            timestamp = sorted(timestamps)[-1].split("/")[-2]
            logging.info(f"Latest timestamp: {timestamp}")
        else:
            raise ValueError(f"No existing timestamps found for {_model_name}")
    elif timestamp is None:
        raise ValueError(f"No existing timestamps found for {_model_name}")

    load_path = os.path.join(output_dir, "outputs", _model_name, timestamp)
    save_path = os.path.join(save_dir, "outputs", _target_model_name, timestamp)
    if os.path.exists(save_path):
        raise FileExistsError(f"Output directory {save_path} already exists.")
    df = pd.concat([pd.read_parquet(os.path.join(load_path, f)) for f in os.listdir(load_path) if f.endswith(".parquet")])
    # ["sample_id", "request_id", "task_name"] unique defination
    # ! all are greedy generation and no likelihoods metrics

    df.reset_index(drop=True, inplace=True)
    df["full_prompt"] = df["full_prompt"].str.cat(["<think>"]*len(df))
    prompts = df["full_prompt"].tolist()

    llm = LLM(
        model=target_model_name, 
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=0.9,
        max_model_len=max_length)

    # sampling params
    sampling_params = SamplingParams(
        max_tokens=None,
        temperature=0,
    )

    # inference
    outputs = llm.generate(
        prompts=prompts,
        sampling_params=sampling_params,
    )

    predictions = [str([o.text for o in output.outputs]) for output in outputs]
    generated_tokens_count = [str([len(o.token_ids) for o in output.outputs]) for output in outputs]
    #! not update logits

    df["predictions"] = predictions
    df["generated_tokens_count"] = generated_tokens_count

    os.makedirs(save_path)
    for task_name, group in df.groupby("task_name"):
        group.to_parquet(
            os.path.join(save_path, f"outputs_{task_name}_{timestamp}.parquet"),
            index=False
        )