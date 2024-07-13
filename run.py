import json
import os
import uuid
import argparse
import yaml
import logging
import transformers
import torch
import anthropic
import pandas as pd
from typing import List

# run model, called from main
def run_model(config: dict, data_path: str = 'helper_files/ids.jsonl'):

    temperature = config['temperature']
    mod_name = config['mod_name']
    chunk_num = config['chunk_num']
    summary = config['summary']

    # extract transcript ids from file
    # Load data from JSON file into list of ids
    with open(data_path, 'r') as f:
        ids = list(f)
    id_list = []
    for id in ids:
        id = json.loads(id)
        id_list.append(id['id'])
    ids = id_list
    
    # can use these paths or change to own model paths
    if mod_name == 'llama-8b': 
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/8b-instruct"
        output = llama_run(model_id, summary, chunk_num, ids)

    elif mod_name == 'llama-70b':
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/70b-instruct"
        output = llama_run(model_id, summary, chunk_num, ids)

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

    return output


def read_transcript_from_id(transcript_id: str, chunk_num: int=1)->List[str]:

    path_to_data_folder = '/archive/shared/sim_center/shared/ameer/'
    # path_to_data_folder = '/archive/shared/sim_center/shared/annie/GPT4 3-chunk/'
    # lookinto this dictionary to find the path
    # can also manually create the path and it would be faster but not by much

    merged_lookup = pd.read_csv(path_to_data_folder + 'grade_lookupv5.csv')

    path = merged_lookup[merged_lookup.id == transcript_id].path.iloc[0]

    path = path[:-4] + '.json'

    # Opening JSON file
    f = open(path)

    # returns JSON object as 
    # a dictionary
    json_transcript = json.load(f)

    transcript = []
    transcript_txt = ''

    lines = json_transcript
    
    if chunk_num == 1: 
        for line in lines:
            if line['text'] != '\n':
                tok_line = line['text'].split(' ')
                for i in range(len(tok_line)):
                    transcript_txt += ' ' + tok_line[i]
        transcript.append(transcript_txt)
    
    else:
        transcript_chunks = []
        # for each chunk
        for n in range(chunk_num):
            transcript = ''
            # get the relevant lines
            start = n*int(len(lines)/chunk_num)
            end = (n+1)*int(len(lines)/chunk_num)
            if n == chunk_num-1: end = len(lines)

            for line in lines[start: end]:
                if line['text'] != '\n':
                    tok_line = line['text'].split(' ')
                    for i in range(len(tok_line)):
                        transcript += ' ' + tok_line[i]
            #append to transcript
            transcript_chunks.append(transcript)
        
        transcript = transcript_chunks

    return transcript

def summary_prompt(path: str)->str:

    with open(path, 'r') as f:
        prompt = f.read()
    return prompt

# summarize helper
def llama_summarize(transcript: list, pipeline: transformers.pipeline, 
                    summary_path='helper_files/summary_prompt.txt')->List[str]:

    prompt = summary_prompt(summary_path)

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    s = []
    
    for chunk in transcript:
        
        messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": chunk},
        ]

        outputs = pipeline(
                messages,
                max_new_tokens=256,
                eos_token_id=terminators,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                )

        s.append(outputs[0]["generated_text"][-1])

    # list of summaries from each chunk
    return s

def diarize_prompt(path: str)->str:

    with open(path, 'r') as f:
        prompt = f.read()
    return prompt

# diarizer
def llama_diarize(transcript: list, pipeline: transformers.pipeline, diarize_path='helper_files/diarize_prompt_w_summary.txt', summary=None)->List[str]:
 
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    diarized = []

    for chunk in transcript:

        if summary: 
            messages = [
        {"role": "system", "content": diarize_prompt(diarize_path)},
        {"role": "user", "content": 'summary: ' + summary["content"] 
         + '\n\ntranscript: ' + chunk},
        ]

        else:
            messages = [
        {"role": "system", "content": diarize_prompt(diarize_path)},
        {"role": "user", "content": 'transcript: ' + chunk}      
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=10000,
            eos_token_id=terminators,
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
        )

        diarized_chunk = outputs[0]["generated_text"][-1]
        diarized.append(diarized_chunk['content'])

    return diarized
   

# run a llama model
def llama_run(model_id: str, summary: bool, chunk_num: int, ids: list):

    pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

    for id in ids: 
        print('diarizing: ' + id)
        transcript = read_transcript_from_id(id, chunk_num=chunk_num)
        if summary: 
            s = llama_summarize(transcript, pipeline)
            diarized = llama_diarize(transcript, summary=s)
        else:
            diarized = llama_diarize(transcript)

    final = ''
    for chunk in diarized:
        final += chunk + '\n'

    return final

# run a claude model
def claude_run(key: str, model_id: str, summary: bool):

    return 'hi'




def main():

    parser = argparse.ArgumentParser(description="Run Models")

    parser.add_argument(
        "--data_path", required=False, 
        help="Path to jsonl files with ids",
        default='helper_files/ids.jsonl'
    )

    parser.add_argument(
        "--summary_path", required=False, 
        help="Path to .txt file with summary prompt",
        default='helper_files/summary_prompt.txt'
    )

    parser.add_argument(
        "--diarize_path", required=False, 
        help="Path to .txt file with diarize prompt",
        default='helper_files/diarize_prompt_w_summary.txt'
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
    summary_path = args.summary_prompt
    diarize_path = args.diarize_prompt

    output = run_model(config, data_path=data_path, summary_path=summary_path, 
                       diarize_path=diarize_path)
    print(output)

    # seeing main output to debug
    with open('out.txt', 'w') as f:
        for element in output:
            f.write(str(element) + '\n')

    
if __name__ == "__main__":
    main()


