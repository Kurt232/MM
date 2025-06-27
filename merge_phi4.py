from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import yaml

output_path = "../mm1/merged"
cfg_root = "configs/merge"

def merge(model_name, base_model_name, output_dir, target):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
    )
    state_dict = model.model.state_dict()

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=True,
        device_map="cpu",
        torch_dtype="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=True,
    )
    base_state_dict = base_model.model.state_dict()

    assert state_dict.keys() == base_state_dict.keys()

    targeted_module = {}
    for key in base_state_dict.keys():
        for module in target:
            if module in key:
                if module not in targeted_module:
                    targeted_module[module] = []
                targeted_module[module].append(key)
                break

    for module, keys in targeted_module.items():
        for key in keys:
            base_state_dict[key] = state_dict[key]

    base_model.model.load_state_dict(base_state_dict, strict=True)
    base_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

cfg_files = [f for f in os.listdir(cfg_root) if f.endswith(".yaml")]
for cfg_file in cfg_files:
    with open(os.path.join(cfg_root, cfg_file), 'r') as file:
        cfg = yaml.safe_load(file)
    model_name = cfg['model_name']
    base_model_name = cfg['base_model_name']
    output_name = cfg['name']
    output_dir = os.path.join(output_path, output_name)
    if os.path.exists(output_dir):
        print(f"Skipping {output_name}, output directory already exists.")
        continue
    else:
        os.makedirs(output_dir)
    print(f"Processing {output_name}...")
    target = cfg['target']
    merge(model_name, base_model_name, output_dir, target)