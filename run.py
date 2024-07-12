import json
import os
import uuid
import argparse
import yaml
import logging
import transformers
import torch
import anthropic


# run model, called from main
def run_model(config: dict, data_path: str = 'data_ids/ids.jsonl'):

    temperature = config['temperature']
    mod_name = config['mod_name']
    summary = config['summary']
    print(temperature, mod_name)

    # can use these paths or change to own model paths
    if mod_name == 'llama-8b': 
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/8b-instruct"
        output = llama_run(model_id, summary)

    elif mod_name == 'llama-70b':
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/70b-instruct"
        output = llama_run(model_id, summary)

    if mod_name == 'claude-opus': 
        model_id = "claude-3-opus-20240229"
        key_path = config['claude_key']
        with open(key_path, 'r') as file: key = file.read()
        output = claude_run(key, model_id, summary)

    elif mod_name == 'claude-sonnet': 
        model_id = "claude-3-sonnet-20240229"
        key_path = config['claude_key']
        with open(key_path, 'r') as file: key = file.read()
        output = claude_run(key, model_id, summary)

    return config, data_path

# run a llama model
def llama_run(model_id: str, summary: bool):

# run a claude model
def claude_run(key: str, model_id: str, summary: bool):




def main():

    parser = argparse.ArgumentParser(description="Run Models")

    parser.add_argument(
        "--data_path", required=False, 
        help="Path to jsonl files with ids",
        default='data_ids/ids.jsonl'
    )

    # the default directory and default ids may already have associated files, so make sure to check before accidentally replacing and change either path or ids
    parser.add_argument(
        "--output_dir", required=False, 
        help="Directory to save diarized transcript files", 
        default='/archive/shared/sim_center/shared/annie/testing_scripts/'
    )
    parser.add_argument("--config",required=True, 
                        help="File path to a config.yaml")

    args = parser.parse_args()

    # read the config as a dictionary from a yaml
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # specify path to jsonl file with ids
    data_path = args.data_path

    output = run_model(config, data_path=data_path)
    print(output)

    # seeing main output to debug
    with open('out.txt', 'w') as f:
        for element in output:
            f.write(str(element) + '\n')

    
if __name__ == "__main__":
    main()


