import json
import argparse
import yaml
import transformers
import torch
import anthropic
import pandas as pd
from typing import List


# run model, called from main
def run_model(config: dict, data_path: str = 'helper_files/ids.jsonl', 
              summary_path: str = 'helper_files/summary_prompt.txt', 
              diarize_path: str = 'helper_files/diarize_prompt_w_summary.txt',
              output_dir: str='temp/'):


    # extract transcript ids from file
    # Load data from JSON file into list of ids
    with open(data_path, 'r') as f:
        ids = list(f)
    id_list = []
    for id in ids:
        id = json.loads(id)
        id_list.append(id['id'].split('.')[0])
    ids = id_list

    # some relevant configs
    chunk_num = config['chunk_num']
    summary = config['summary']
    mod_name_summary = config['mod_name_summary']
    mod_name_diarize = config['mod_name_diarize']
    
    output = []
    # run pipeline
    for id in ids:
        print('diarizing: ' + id)

        # extract transcript (chunks)
        transcript = read_transcript_from_id(id, chunk_num=chunk_num)

        # if summary is True, extract summar(ies)
        if summary:
            if 'llama' in mod_name_summary: 
                s = llama_summarize(transcript, config, 
                                    summary_path=summary_path)
            elif 'claude' in mod_name_summary:
                s = claude_summarize(transcript, config, 
                                     summary_path=summary_path)
            else: s = None

        if 'llama' in mod_name_diarize:
            diarized = llama_diarize(transcript, config, 
                                     diarize_path=diarize_path, summary_list=s)
        
        elif 'claude' in mod_name_diarize:
            diarized = claude_diarize(transcript, config, 
                                      diarize_path=diarize_path, summary_list=s)
        
        # return chunks if parameter true
        
        output.append(diarized)

        with open(output_dir + id + '.txt', 'w') as f:
            j = 0
            for chunk in diarized:
                if config['return_chunked']:
                    f.write('chunk ' + str(j) + ':\n\n' + str(chunk) + '\n\n')
                else: 
                    f.write(str(chunk) + '\n\n')
                j += 1
    
    return ids, output


def read_transcript_from_id(transcript_id: str, chunk_num: int)->List[str]:

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


def diarize_prompt(path: str)->str:

    with open(path, 'r') as f:
        prompt = f.read()
    return prompt


# summarize helper
def llama_summarize(transcript: list, config: dict,
                    summary_path: str='helper_files/summary_prompt.txt')->List[str]:
    
    temperature = config['temperature_summary']
    max_new_tokens = config['max_new_tokens_summary']
    top_p = config['top_p_summary']
    mod_name = config['mod_name_summary']
    do_sample = config['do_sample_summary']

    prompt = summary_prompt(summary_path)

    # change these paths if you wish to use llama instances stored elsewhere
    if mod_name == 'llama-8b': 
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/8b-instruct"

    elif mod_name == 'llama-70b':
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/70b-instruct"

    pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
    
    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    s = []

    # if summarizing entire transcript instead of each chunk
    if not config['summary_chunking']:
        
        transcript = ''
        for chunk in transcript: transcript += chunk + '\n\n'

        messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": transcript},
        ]

        outputs = pipeline(
                messages,
                max_new_tokens=max_new_tokens,
                eos_token_id=terminators,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                )

        summary_chunk = outputs[0]["generated_text"][-1]
        s.append(summary_chunk['content'])
    
    # summarize each chunk
    else: 
        for chunk in transcript:
        
            messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": chunk},
            ]

            outputs = pipeline(
                    messages,
                    max_new_tokens=max_new_tokens,
                    eos_token_id=terminators,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    )

            summary_chunk = outputs[0]["generated_text"][-1]
            s.append(summary_chunk['content'])

    # list of summaries from each chunk
    return s


# diarizer
def llama_diarize(transcript: list, config: dict,
                  diarize_path: str='helper_files/diarize_prompt_w_summary.txt', 
                  summary_list: list=None)->List[str]:
    
    temperature = config['temperature_diarize']
    max_new_tokens = config['max_new_tokens_diarize']
    top_p = config['top_p_diarize']
    do_sample = config['do_sample_diarize']
    mod_name = config['mod_name_diarize']
    summary = config['summary']


    prompt = diarize_prompt(diarize_path)

    # change these paths if you wish to use llama instances stored elsewhere
    if mod_name == 'llama-8b': 
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/8b-instruct"

    elif mod_name == 'llama-70b':
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/70b-instruct"

    pipeline = transformers.pipeline(
                "text-generation",
                model=model_id,
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    diarized = []

    for i in range(len(transcript)):

        if summary: 
            messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": 'summary: ' + summary_list[i] 
         + '\n\ntranscript: ' + transcript[i]},
        ]

        else:
            messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": 'transcript: ' + transcript[i]}      
        ]

        outputs = pipeline(
            messages,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        diarized_chunk = outputs[0]["generated_text"][-1]
        diarized.append(diarized_chunk['content'])

    # list of diarized chunks
    return diarized
   


# summarize using claude
def claude_summarize(transcript, config: dict, summary_path: str='helper_files/summary_prompt.txt')->List[str]:

    key_path = config['claude_key']
    with open(key_path, 'r') as file: 
        key = file.read()

    mod_name = config['mod_name_summary']
    max_new_tokens = config['max_new_tokens_summary']
    temperature = config['temperature_summary']
    prompt = summary_prompt(summary_path)


    if mod_name == 'claude-opus': 
        model_id = "claude-3-opus-20240229"
    

    elif mod_name == 'claude-sonnet': 
        model_id = "claude-3-sonnet-20240229"
    
    client = anthropic.Anthropic(api_key=key,)

    s = []
    for chunk in transcript:
        summary = client.messages.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system=prompt,
            messages=[
                {"role": "user", "content": chunk}
            ]
        )
        
        s.append(summary.content[0].text)
    
    return s


# summarize using claude
def claude_diarize(transcript, config: dict, diarize_path: str='helper_files/diarize_prompt_w_summary.txt', summary_list=None)->List[str]:

    key_path = config['claude_key']
    with open(key_path, 'r') as file: key = file.read()

    mod_name = config['mod_name_diarize']
    max_new_tokens = config['max_new_tokens_diarize']
    temperature = config['temperature_diarize']
    summary = config['summary']
    prompt = diarize_prompt(diarize_path)

    if mod_name == 'claude-opus': 
        model_id = "claude-3-opus-20240229"
    
    elif mod_name == 'claude-sonnet': 
        model_id = "claude-3-sonnet-20240229"
    
    client = anthropic.Anthropic(api_key=key,)

    diarized = []

    for i in range(len(transcript)):

        if summary: 
            diarization = client.messages.create(
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
                system=prompt,
                messages=[
                    {"role": "user", "content": 'transcript: ' + transcript[i]}
                ]
            )
        
        else: 
            diarization = client.messages.create(
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
                system=prompt,
                messages=[
                    {"role": "user", "content": 'summary: ' + summary_list[i] + '\n\ntranscript: ' + transcript[i]}
                ]
            )
        
        diarized.append(diarization.content[0].text)
    
    return diarized


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
        default='temp/'
    )
    parser.add_argument("--config",required=True, 
                        help="File path to a config.yaml")

    args = parser.parse_args()

    # read the config as a dictionary from a yaml
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # specify path to jsonl file with ids
    data_path = args.data_path
    summary_path = args.summary_path
    diarize_path = args.diarize_path
    output_dir = args.output_dir

    ids, output = run_model(config, data_path=data_path, 
                            summary_path=summary_path, 
                            diarize_path=diarize_path,
                            output_dir=output_dir)
    print(ids, output)

    # seeing main output to debug
    i = 0
    with open('out.txt', 'w') as f:
        for element in output:
            f.write(ids[i])
            i += 1
            f.write(str(element) + '\n')

    
if __name__ == "__main__":
    main()


