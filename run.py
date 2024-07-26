import json
import argparse
import yaml
import transformers
import torch
import anthropic
import pandas as pd
import os
from typing import List
from openai import AzureOpenAI


def run_model(
        config: dict,
        data_path: str = "helper_files/ids.jsonl",
        summary_path: str = "helper_files/summary_prompt.txt",
        diarize_path: str = "helper_files/diarize_prompt_w_summary.txt",
        output_dir: str = "temp/"
) -> tuple[List[str], List[str]]:

    # extract transcript ids from file
    # Load data from JSON file into list of ids
    with open(data_path, "r") as f:
        ids = list(f)
    id_list = []
    for id in ids:
        id = json.loads(id)
        id_list.append(id["id"].split(".")[0])
    ids = id_list

    # some relevant configs
    chunk_num = config["chunk_num"]
    summary = config["summary"]
    mod_name_summary = config["mod_name_summary"]
    mod_name_diarize = config["mod_name_diarize"]
    return_json = config["return_json"]
    json_and_txt = config["json_and_txt"]

    output = []

    # run pipeline
    raw_transcript = config["raw_transcript"]  # check if user wanted to return only raw transcripts

    for id in ids:

        # extract transcript (chunks)
        transcript = read_transcript_from_id(id, chunk_num=chunk_num)

        if raw_transcript:
            output.append(transcript)

            if return_json or json_and_txt:
                json_transcript = read_transcript_from_id(id, chunk_num=chunk_num, return_json=True)

        print("getting: " + id)

        # if summary is True, extract summar(ies)
        if (not raw_transcript) and summary:

            print("summarizing: " + id)

            if "llama" in mod_name_summary:
                s = llama_summarize(transcript, config, summary_path=summary_path)
            elif "claude" in mod_name_summary:
                s = claude_summarize(transcript, config, summary_path=summary_path)
            elif "azure" in mod_name_summary:
                s = azure_summarize(transcript, config, summary_path=summary_path)
            else:
                s = None
                print("check summary model name")

        if not raw_transcript:

            if not summary:
                s = None

            print("diarizing: " + id)

            if "llama" in mod_name_diarize:
                diarized = llama_diarize(
                    transcript, config, diarize_path=diarize_path, summary_list=s
                )

            elif "claude" in mod_name_diarize:
                diarized = claude_diarize(
                    transcript, config, diarize_path=diarize_path, summary_list=s
                )

            elif "azure" in mod_name_diarize:
                diarized = azure_diarize(
                    transcript, config, diarize_path=diarize_path, summary_list=s
                )

            else:
                diarized = None
                print("check diarization model name")

            output.append(diarized)

        # set file name
        run_name = config["run_name"]

        if run_name is not None:
            filename = run_name + '_' + id

        else:
            filename = id

        # create subfolders if returning both txt and json transcripts
        if json_and_txt:
            if not os.path.isdir(output_dir + '/json'):
                os.mkdir(output_dir + '/json')
            if not os.path.isdir(output_dir + '/text'):
                os.mkdir(output_dir + '/text')

        # write to txt file
        if (not return_json) or (json_and_txt):

            if json_and_txt:
                thisfile = output_dir + 'text/' + filename + '.txt'
            else:
                thisfile = output_dir + filename + '.txt'

            with open(thisfile, "w") as f:
                j = 0
                for chunk in output[-1]:
                    if config["return_chunked"]:
                        f.write("chunk " + str(j) + ":\n\n" + str(chunk) + "\n\n")
                    else:
                        f.write(str(chunk) + "\n\n")
                    j += 1

        # construct json transcript and write to .json file
        json_transcript = json_construct(diarized)

        if (return_json) or (json_and_txt):

            if json_and_txt:
                thisfile = output_dir + 'json/' + filename + '.json'
            else:
                thisfile = output_dir + filename + '.json'

            with open(thisfile, "w") as f:
                json.dump(json_transcript, f)

    return ids, output


def read_transcript_from_id(transcript_id: str, chunk_num: int, return_json: bool = False) -> List[str]:

    path_to_data_folder = "/archive/shared/sim_center/shared/ameer/"
    # path_to_data_folder = '/archive/shared/sim_center/shared/annie/GPT4 3-chunk/'
    # lookinto this dictionary to find the path
    # can also manually create the path and it would be faster but not by much

    merged_lookup = pd.read_csv(path_to_data_folder + "grade_lookupv5.csv")

    path = merged_lookup[merged_lookup.id == transcript_id].path.iloc[0]

    path = path[:-4] + ".json"

    # Opening JSON file
    f = open(path)

    # returns JSON object as
    # a dictionary
    json_transcript = json.load(f)

    transcript = []
    transcript_txt = ""

    lines = json_transcript

    if chunk_num == 1:

        if return_json:
            return json_transcript

        for line in lines:
            if line["text"] != "\n":
                tok_line = line["text"].split(" ")
                for i in range(len(tok_line)):
                    transcript_txt += " " + tok_line[i]
        transcript.append(transcript_txt)

    else:

        if return_json:

            json_transcript = {}

            for n in range(chunk_num):

                start = n * int(len(lines) / chunk_num)
                end = (n + 1) * int(len(lines) / chunk_num)
                if n == chunk_num - 1:
                    end = len(lines)

                json_transcript["chunk " + str(n)] = lines[start:end]

            return json_transcript

        transcript_chunks = []
        # for each chunk
        for n in range(chunk_num):
            transcript = ""
            # get the relevant lines
            start = n * int(len(lines) / chunk_num)
            end = (n + 1) * int(len(lines) / chunk_num)
            if n == chunk_num - 1:
                end = len(lines)

            for line in lines[start:end]:
                if line["text"] != "\n":
                    tok_line = line["text"].split(" ")
                    for i in range(len(tok_line)):
                        transcript += " " + tok_line[i]
            # append to transcript
            transcript_chunks.append(transcript)

        transcript = transcript_chunks

    return transcript


# helper function to construct json from text diarization output
def json_construct(diarized: List[str]):

    out = []

    i = 0
    for chunk in diarized:
        lines = chunk.split('\n')

        for line in lines:
            if line.find(':') == -1:
                continue
            temp = {}
            temp['chunk'] = i
            temp['speaker'] = line[:line.find(':')]
            temp['text'] = line[line.find(':') + 1: line.find('(')]
            temp['timestamp'] = []
            if (line.find('(') != -1) and ('?' not in line[line.find('('): line.find(')')]):
                temp['timestamp'].append(float(line[line.find('(') + 1: line.rfind(',')]))
                temp['timestamp'].append(float(line[line.rfind(',') + 2: line.find(')')]))

            out.append(temp)

        i += 1

    return out


def summary_prompt(path: str) -> str:

    with open(path, "r") as f:
        prompt = f.read()
    return prompt


def diarize_prompt(path: str) -> str:

    with open(path, "r") as f:
        prompt = f.read()
    return prompt


# summarize helper
def llama_summarize(
    transcript: list,
    config: dict,
    summary_path: str = "helper_files/summary_prompt.txt",
) -> List[str]:

    temperature = config["temperature_summary"]
    max_new_tokens = config["max_new_tokens_summary"]
    top_p = config["top_p_summary"]
    mod_name = config["mod_name_summary"]
    do_sample = config["do_sample_summary"]

    prompt = summary_prompt(summary_path)

    # change these paths if you wish to use llama instances stored elsewhere
    if mod_name == "llama-8b":
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/8b-instruct"

    elif mod_name == "llama-70b":
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/70b-instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    s = []

    # if summarizing entire transcript instead of each chunk
    if not config["summary_chunking"]:

        transcript = ""
        for chunk in transcript:
            transcript += chunk + "\n\n"

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
        s.append(summary_chunk["content"])

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
            s.append(summary_chunk["content"])

    # list of summaries from each chunk
    return s


# diarizer
def llama_diarize(
    transcript: list,
    config: dict,
    diarize_path: str = "helper_files/diarize_prompt_w_summary.txt",
    summary_list: list = None,
) -> List[str]:

    temperature = config["temperature_diarize"]
    max_new_tokens = config["max_new_tokens_diarize"]
    top_p = config["top_p_diarize"]
    do_sample = config["do_sample_diarize"]
    mod_name = config["mod_name_diarize"]
    summary = config["summary"]

    prompt = diarize_prompt(diarize_path)

    # change these paths if you wish to use llama instances stored elsewhere
    if mod_name == "llama-8b":
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/8b-instruct"

    elif mod_name == "llama-70b":
        model_id = "/archive/shared/sim_center/shared/annie/hf_models/70b-instruct"

    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        pipeline.tokenizer.eos_token_id,
        pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    diarized = []

    for i in range(len(transcript)):

        if summary:
            messages = [
                {"role": "system", "content": prompt},
                {
                    "role": "user",
                    "content": "summary: " + summary_list[i] + "\n\ntranscript: " + transcript[i],
                },
            ]

        else:
            messages = [
                {"role": "system", "content": prompt},
                {"role": "user", "content": "transcript: " + transcript[i]},
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
        diarized.append(diarized_chunk["content"])

    # list of diarized chunks
    return diarized


# summarize using claude
def claude_summarize(
    transcript: str, config: dict, summary_path: str = "helper_files/summary_prompt.txt"
) -> List[str]:

    key_path = config["claude_key"]
    with open(key_path, "r") as file:
        key = file.read()

    mod_name = config["mod_name_summary"]
    max_new_tokens = config["max_new_tokens_summary"]
    temperature = config["temperature_summary"]
    prompt = summary_prompt(summary_path)

    if mod_name == "claude-opus-3":
        model_id = "claude-3-opus-20240229"

    elif mod_name == "claude-sonnet-3":
        model_id = "claude-3-sonnet-20240229"

    elif mod_name == "claude-sonnet-3.5":
        model_id = "claude-3-5-sonnet-20240620"

    elif mod_name == "claude-haiku-3":
        model_id = "claude-3-haiku-20240307"

    client = anthropic.Anthropic(
        api_key=key,
    )

    s = []
    for chunk in transcript:
        summary = client.messages.create(
            model=model_id,
            max_tokens=max_new_tokens,
            temperature=temperature,
            system=prompt,
            messages=[{"role": "user", "content": chunk}],
        )

        s.append(summary.content[0].text)

    return s


# summarize using claude
def claude_diarize(
    transcript,
    config: dict,
    diarize_path: str = "helper_files/diarize_prompt_w_summary.txt",
    summary_list=None,
) -> List[str]:

    key_path = config["claude_key"]
    with open(key_path, "r") as file:
        key = file.read()

    mod_name = config["mod_name_diarize"]
    max_new_tokens = config["max_new_tokens_diarize"]
    temperature = config["temperature_diarize"]
    summary = config["summary"]
    prompt = diarize_prompt(diarize_path)

    if mod_name == "claude-opus-3":
        model_id = "claude-3-opus-20240229"

    elif mod_name == "claude-sonnet-3":
        model_id = "claude-3-sonnet-20240229"

    elif mod_name == "claude-sonnet-3.5":
        model_id = "claude-3-5-sonnet-20240620"

    elif mod_name == "claude-haiku-3":
        model_id = "claude-3-haiku-20240307"

    client = anthropic.Anthropic(
        api_key=key,
    )

    diarized = []

    for i in range(len(transcript)):

        if not summary:
            diarization = client.messages.create(
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
                system=prompt,
                messages=[{"role": "user", "content": "transcript: " + transcript[i]}],
            )

        elif summary:
            diarization = client.messages.create(
                model=model_id,
                max_tokens=max_new_tokens,
                temperature=temperature,
                system=prompt,
                messages=[
                    {
                        "role": "user",
                        "content": "summary: " + summary_list[i] + "\n\ntranscript: " + transcript[i],
                    }
                ],
            )

        diarized.append(diarization.content[0].text)

    return diarized


def azure_summarize(
        transcript: list,
        config: dict,
        summary_path: str = "helper_files/summary_prompt.txt") -> List[str]:

    '''print(
        os.environ.get("OPENAI_API_VERSION"),
        os.environ.get("AZURE_OPENAI_ENDPOINT"),
        os.environ.get("AZURE_OPENAI_API_KEY"))  # for debugging if connection error'''

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

    deployment = config["azure_deployment_summary"]
    prompt = summary_prompt(summary_path)

    s = []
    for chunk in transcript:

        messages = [
            {
                "role": "user",
                "content": prompt + "\n\n" + "transcript: \n\n" + chunk
            }]

        completion = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=config["max_new_tokens_summary"],
            temperature=config["temperature_summary"],
            top_p=config["top_p_summary"],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        s.append(json.loads(completion.to_json())["choices"][0]["message"]["content"])

    return s


def azure_diarize(
        transcript: list,
        config: dict,
        diarize_path: str = "helper_files/diarize_prompt.txt",
        summary_list: list = None) -> List[str]:

    '''print(
        os.environ.get("OPENAI_API_VERSION"),
        os.environ.get("AZURE_OPENAI_ENDPOINT"),
        os.environ.get("AZURE_OPENAI_API_KEY"))  # for debugging if connection error'''

    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("OPENAI_API_VERSION"),
    )

    deployment = config["azure_deployment_diarize"]
    prompt = diarize_prompt(diarize_path)
    summary = config["summary"]

    diarized = []

    i = 0
    for chunk in transcript:

        if summary:

            messages = [
                {
                    "role": "user",
                    "content": prompt + "\n\nsummary:" + summary_list[i] + "\n\ntranscript: " + chunk
                }]

        else:

            messages = [
                {
                    "role": "user",
                    "content": prompt + "\n\ntranscript: " + chunk
                }]

        completion = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=config["max_new_tokens_diarize"],
            temperature=config["temperature_diarize"],
            top_p=config["top_p_diarize"],
            frequency_penalty=0,
            presence_penalty=0,
            stop=None,
            stream=False
        )

        diarized.append(json.loads(completion.to_json())["choices"][0]["message"]["content"])
        i += 1

    return diarized


def main():

    parser = argparse.ArgumentParser(description="Run Models")

    parser.add_argument(
        "--data_path",
        required=False,
        help="Path to jsonl files with ids",
        default="helper_files/test-id.jsonl",
    )

    parser.add_argument(
        "--summary_path",
        required=False,
        help="Path to .txt file with summary prompt",
        default="helper_files/summary_prompt.txt",
    )

    parser.add_argument(
        "--diarize_path",
        required=False,
        help="Path to .txt file with diarize prompt",
        default="helper_files/diarize_prompt_w_summary.txt",
    )

    # the default directory and default ids may already have associated files, so make sure to check before accidentally replacing and change either path or ids
    parser.add_argument(
        "--output_dir",
        required=False,
        help="Directory to save diarized transcript files",
        default="temp/",
    )
    parser.add_argument("--config", required=False, help="File path to a config.yaml", default="config.yaml")

    args = parser.parse_args()

    # read the config as a dictionary from a yaml
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # specify path to jsonl file with ids
    data_path = args.data_path
    summary_path = args.summary_path
    diarize_path = args.diarize_path
    output_dir = args.output_dir

    ids, output = run_model(
        config,
        data_path=data_path,
        summary_path=summary_path,
        diarize_path=diarize_path,
        output_dir=output_dir,
    )
    print(ids, output)

    # seeing main output to debug
    i = 0
    with open("out.txt", "w") as f:
        for element in output:
            f.write(ids[i])
            i += 1
            f.write(str(element) + "\n")


if __name__ == "__main__":
    main()
