import argparse
import yaml
import pandas as pd
import os
from typing import List
import json
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import difflib
import transformers
import torch
import anthropic
from openai import AzureOpenAI
from openai import OpenAI


def run_eval(
        config: dict,
        data_path: str,
        diarized_path_txt: str,
        diarized_path_json: str,
        gold_path: str,
        input_dir: str,
) -> tuple[List[str], List[str]]:
    
    # extract transcript ids from file
    # Load data from JSON file into list of ids
    with open(data_path, "r") as f:
        ids = list(f)
    id_list = []
    for id in ids:
        id = json.loads(id)
        id_list.append(id["id"])
    ids = id_list

    preservation = config["preservation"]
    accuracy = config["accuracy"]
    input_json = config["input_json"]
    input_json_and_txt = config["input_json_and_txt"]

    if input_json and not input_json_and_txt:
        diarized_path_txt = None
    
    if not input_json and not input_json_and_txt:
        diarized_path_json = None

    scores = {}

    if preservation:
        scores['preservation'] = calc_preservation(config, ids, diar_path_txt=diarized_path_txt, diar_path_json=diarized_path_json, input_dir=input_dir)
    if accuracy:
        scores['accuracy'], scores['nolabel_baseline'] = calc_accuracy(config, ids, diar_path_txt=diarized_path_txt, diar_path_json=diarized_path_json, gold_path=gold_path)

    return ids, scores

    
def read_transcript_from_id(transcript_id: str, segments: int, input_dir: str='input/') -> List[str]:

    path_to_data_folder = input_dir
    # path_to_data_folder = '/archive/shared/sim_center/shared/annie/GPT4 3-chunk/'
    # lookinto this dictionary to find the path
    # can also manually create the path and it would be faster but not by much

    path = path_to_data_folder + str("/") + id + ".json"

    # Opening JSON file
    f = open(path)

    # returns JSON object as
    # a dictionary
    json_transcript = json.load(f)

    transcript = []
    transcript_txt = ""

    lines = json_transcript

    if segments == 1:

        for line in lines:
            if line["text"] != "\n":
                tok_line = line["text"].split(" ")
                for i in range(len(tok_line)):
                    transcript_txt += " " + tok_line[i]
        transcript.append(transcript_txt)

    else:

        transcript_chunks = []
        # for each chunk
        for n in range(segments):
            transcript = ""
            # get the relevant lines
            start = n * int(len(lines) / segments)
            end = (n + 1) * int(len(lines) / segments)
            if n == segments - 1:
                end = len(lines)

            for line in lines[start:end]:
                if line["text"] != "\n":
                    tok_line = line["text"].split(" ")
                    for i in range(len(tok_line)):
                        transcript += " " + tok_line[i]

            transcript = " ".join(transcript.split())  # clean up spaces and newline
            # append to transcript
            transcript_chunks.append(transcript)

        transcript = transcript_chunks

    return transcript


def reconstruct_transcript(path: str, id: str, segments: int) -> List[str]:

    transcript = ''
    path = path + str("/") + id + '.txt'
    with open(path, 'r') as file:
        lines = file.readlines()

    transcript_chunks = []

    # for each chunk
    for n in range(segments):
        transcript = ""
        # get the relevant lines
        start = n * int(len(lines) / segments)
        end = (n + 1) * int(len(lines) / segments)
        if n == segments - 1:
            end = len(lines)

        for line in lines[start:end]:
            if line != '\n':
                tok_line = line.split(' ')

                # add all tokens except for diarization label
                for i in range(1, len(tok_line)):
                    transcript += ' ' + tok_line[i]

        # clean up any line breaks
        resid_lines = transcript.split('\n')
        transcript = ''
        for line in resid_lines:
            transcript += line
        transcript = " ".join(transcript.split())  # clean up spaces and newline
        # append to transcript
        transcript_chunks.append(transcript)

    transcript = transcript_chunks

    return transcript


def reconstruct_transcript_from_json(path: str, id: str, segments: int) -> List[str]:

    path = path + "/" + id + '.json'
    with open(path, 'r') as file:
        lines = json.load(file)

    transcript_chunks = []
    for n in range(segments):
        transcript = ''
        # get the relevant lines
        start = n * int(len(lines) / segments)
        end = (n + 1) * int(len(lines) / segments)
        if n == segments - 1:
            end = len(lines)
        for line in lines[start: end]:
            transcript += line['text']

        transcript = " ".join(transcript.split())  # clean up spaces and newline

        transcript_chunks.append(transcript)

    transcript = transcript_chunks

    return transcript


# reconstruct diarized version of transcript and consolidate any repeated utterances (e.g. "Patient: Hello  \n\n Patient: How are you? \n\n" -> "Patient: Hello How are you? \n\n")
def consolidate_transcript(path: str, id: str, segments: int):

    path = path + str("/") + id + '.txt'
    with open(path, 'r') as file:
        lines = file.readlines()

    diar_transcript = []
    for n in range(segments):
        # get the relevant lines
        start = n * int(len(lines) / segments)
        end = (n + 1) * int(len(lines) / segments)
        if n == segments - 1:
            end = len(lines)

        out = []
        for line in lines[start:end]:
            temp = {}
            if line.find(':') == -1:
                temp['speaker'] = None
                temp['text'] = line
            temp['speaker'] = line[:line.find(':')]

            # handle for extra formatting tokens like bold
            if 'Patient' or 'patient' in temp['speaker']: 
                temp['speaker'] = 'Patient'
            if 'Student' or 'student' in temp['speaker']:
                temp['speaker'] = 'Student'

            temp['text'] = line[line.find(':') + 1:]
            out.append(temp)

        new = []
        new.append({'speaker': out[0]['speaker'], 'text': out[0]['text']})

        for i in range(1, len(out)):
            this = {}
            if out[i]['speaker'] == out[i - 1]['speaker']:
                new[-1]['text'] += out[i]['text']
            else:
                this['speaker'] = out[i]['speaker']
                this['text'] = out[i]['text']
                new.append(this)

        diar = ''
        for line in new:
            diar += line['speaker'] + ":" + line['text']
        diar = " ".join(diar.split())  # clean up spaces and newline
        diar_transcript.append(diar)

    return diar_transcript


def consolidate_transcript_from_json(path: str, id: str, segments: int):

    path = path + str("/") + id + '.json'
    with open(path, 'r') as file:
        lines = json.load(file)

    diar_transcript = []

    for n in range(segments):
        # get the relevant lines
        start = n * int(len(lines) / segments)
        end = (n + 1) * int(len(lines) / segments)
        if n == segments - 1:
            end = len(lines)

        new = []

        if 'Patient' or 'patient' in lines[start]['speaker']: 
            lines[start]['speaker'] = 'Patient'
        if 'Student' or 'student' in lines[start]['speaker']:
            lines[start]['speaker'] = 'Student'
        
        new.append({'speaker': lines[start]['speaker'], 'text': lines[start]['text']})

        for i in range(start + 1, end):
            this = {}

            if 'Patient' or 'patient' in lines[i]['speaker']: 
                lines[i]['speaker'] = 'Patient'
            if 'Student' or 'student' in lines[i]['speaker']:
                lines[i]['speaker'] = 'Student'
            
            if lines[i]['speaker'] == lines[i - 1]['speaker']:
                new[-1]['text'] += lines[i]['text']
            else:
                this['speaker'] = lines[i]['speaker']
                this['text'] = lines[i]['text']
                new.append(this)

        diar = ''
        for line in new:
            diar += line['speaker'] + ":" + line['text']

        diar = " ".join(diar.split())  # clean up spaces and newline
        diar_transcript.append(diar)

    return diar_transcript


def calc_preservation(config: dict, ids: List[str], diar_path_txt: str = None, diar_path_json: str = None):

    these_scores = {}
    segments = config["segments"]
    input_json = config["input_json"]
    input_json_and_txt = config["input_json_and_txt"]
    levenshtein = config['levenshtein']
    diff = config['diff']

    if levenshtein:
        these_scores["levenshtein"] = {}
    if diff:
        these_scores["diff"] = {}

    for id in ids:
        print("calculating preservation: ", id)
        transcript = read_transcript_from_id(id, segments=segments)

        if input_json and not input_json_and_txt:
            diar_transcript = reconstruct_transcript_from_json(diar_path_json, id, segments=segments)
        else:
            diar_transcript = reconstruct_transcript(diar_path_txt, id, segments=segments)

        if levenshtein:
            these_scores["levenshtein"][id] = {}
        if diff:
            these_scores["diff"][id] = {}

        for i in range(segments):
            # indexed by id and then chunk number
            if levenshtein:
                these_scores["levenshtein"][id][i] = NormalizedLevenshtein().similarity(transcript[i], diar_transcript[i])
            if diff:
                these_scores["diff"][id][i] = difflib.SequenceMatcher(None, transcript[i], diar_transcript[i]).ratio()

    return these_scores


# calculate baseline score between a transcript with no labels and the gold transcript
def nolabel_baseline_score(config: dict, ids: List[str], diar_path_txt: str = None, diar_path_json: str = None, gold_path: str = '/archive/shared/sim_center/shared/annie/gold-standards/gpt4-gold-standard-diarized/'):

    segments = config["segments"]
    levenshtein = config["levenshtein"]
    diff = config["diff"]
    input_json = config["input_json"]
    input_json_and_txt = config["input_json_and_txt"]
    gold_json = config["gold_json"]

    scores = {}
    scores["levenshtein"] = {}
    scores["diff"] = {}

    for id in ids:

        if input_json and not input_json_and_txt:
            transcript = reconstruct_transcript_from_json(diar_path_json, id, segments)
        else:
            transcript = reconstruct_transcript(diar_path_txt, id, segments)
        if gold_json:
            gold = consolidate_transcript_from_json(gold_path, id, segments)
        else:
            gold = consolidate_transcript(gold_path, id, segments)

        scores["levenshtein"][id] = {}
        scores["diff"][id] = {}

        for n in range(segments):

            if levenshtein:
                similarity = NormalizedLevenshtein().similarity(transcript[n], gold[n])
                scores["levenshtein"][id][n] = similarity

            if diff:
                similarity = difflib.SequenceMatcher(None, transcript[n], gold[n]).ratio()
                scores["diff"][id][n] = similarity

    return scores


def calc_accuracy(config: dict, ids: List[str], diar_path_txt: str = None, diar_path_json: str = None, gold_path: str = '/archive/shared/sim_center/shared/annie/gold-standards/gpt4-gold-standard-diarized/'):

    segments = config["segments"]
    input_json = config["input_json"]
    input_json_and_txt = config["input_json_and_txt"]
    gold_json = config["gold_json"]
    levenshtein = config['levenshtein']
    diff = config['diff']

    these_scores = {}

    if levenshtein:
        these_scores["levenshtein"] = {}
        these_scores["levenshtein_baseline_normed"] = {}
    if diff:
        these_scores["diff"] = {}
        these_scores["diff_baseline_normed"] = {}

    nolabel_baselines = nolabel_baseline_score(config, ids, diar_path_txt=diar_path_txt, gold_path=gold_path)

    for id in ids:

        print("calculating accuracy: ", id)

        if gold_json:
            transcript = consolidate_transcript_from_json(gold_path, id, segments=segments)

        else:
            transcript = consolidate_transcript(gold_path, id, segments=segments)

        if input_json and not input_json_and_txt:
            diar_transcript = consolidate_transcript_from_json(diar_path_json, id, segments=segments)

        else:
            diar_transcript = consolidate_transcript(diar_path_txt, id, segments=segments)

        if levenshtein:
            these_scores["levenshtein"][id] = {}
            these_scores["levenshtein_baseline_normed"][id] = {}
        if diff:
            these_scores["diff"][id] = {}
            these_scores["diff_baseline_normed"][id] = {}

        for i in range(segments):
            # indexed by id and then chunk number
            if levenshtein:
                these_scores["levenshtein"][id][i] = NormalizedLevenshtein().similarity(transcript[i], diar_transcript[i])
                these_scores["levenshtein_baseline_normed"][id][i] = (these_scores["levenshtein"][id][i] - nolabel_baselines["levenshtein"][id][i]) / (1 - nolabel_baselines["levenshtein"][id][i])
            if diff:
                these_scores["diff"][id][i] = difflib.SequenceMatcher(None, transcript[i], diar_transcript[i]).ratio()
                these_scores["diff_baseline_normed"][id][i] = (these_scores["diff"][id][i] - nolabel_baselines["diff"][id][i]) / (1 - nolabel_baselines["diff"][id][i])

    return these_scores, nolabel_baselines


def llm_judge(config: dict, ids: List[str], diar_path_txt: str = None, diar_path_json: str = None):
    return None


def llm_compare(config: dict, ids: List[str], diar_path_txt: str = None, diar_path_json: str = None, gold_path: str = '/archive/shared/sim_center/shared/annie/gold-standards/gpt4-gold-standard-diarized/'):
    return None


def run_llm(config: dict, ids: List[str], diar_path_txt: str = None, diar_path_json: str = None, gold_path: str = '/archive/shared/sim_center/shared/annie/gold-standards/gpt4-gold-standard-diarized/'):

    llama_running = config["llama_running"]
    mod_name = config["mod_name"]
    llm_judge = config["llm_judge"]
    llm_compare = config["llm_compare"]

    if "llama" in mod_name and not llama_running:

        if mod_name == "llama-8b":
            model_id = "/archive/shared/sim_center/shared/annie/hf_models/8b-instruct"
        if mod_name == "llama-70b":
            model_id = "/archive/shared/sim_center/shared/annie/hf_models/70b-instruct"

        pipeline = transformers.pipeline(
            "text-generation",
            model=model_id,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=3,
        )

        terminators = [
            pipeline.tokenizer.eos_token_id,
            pipeline.tokenizer.convert_tokens_to_ids("<|eot_id|>"),
        ]
    
    else:
        pipeline = None
        terminators = None

    output = []

    for id in ids:

        # extract transcript (chunks)
        transcript = read_transcript_from_id(id, segments=1)

        if llm_judge:

            print("judging: " + id)
            out = llm_judge()

        if llm_compare:

            print("comparing: " + id)
            out = llm_compare()

        output.append(out)
        
        return output


def main():

    parser = argparse.ArgumentParser(description="Run Models")

    parser.add_argument(
        "--run_name",
        required=False,
        help="name to save scores to",
        default="evaluation",
    )

    parser.add_argument(
        "--diarized_path_txt",
        required=False,
        help="Path to diarized files in .txt form",
        default="/archive/shared/sim_center/shared/annie/diarized/gpt4-all/text/",
    )

    parser.add_argument(
        "--diarized_path_json",
        required=False,
        help="Path to diarized files in .json form",
        default="/archive/shared/sim_center/shared/annie/diarized/gpt4-all/json/",
    )

    parser.add_argument(
        "--data_path",
        required=False,
        help="Path to .jsonl file with ids",
        default="helper_files/test-id.jsonl",
    )

    parser.add_argument(
        "--segments",
        required=False,
        help="number of segments to eval (overrides value in config)",
        default=None,
    )

    parser.add_argument(
        "--gold_path",
        required=False,
        help="Path to folder with gold standards",
        default="/archive/shared/sim_center/shared/annie/gold-standards/gpt4-gold-standard-diarized/",
    )

    # the default directory and default ids may already have associated files, so make sure to check before accidentally replacing and change either path or ids
    parser.add_argument(
        "--input_dir",
        required=False,
        help="Directory with raw transcripts or text input",
        default="input/",
    )

    # the default directory and default ids may already have associated files, so make sure to check before accidentally replacing and change either path or ids
    parser.add_argument(
        "--output_dir",
        required=False,
        help="Directory to save diarized transcript files",
        default="temp/",
    )
    parser.add_argument("--config", required=False, help="File path to a config.yaml", default="eval_config.yaml")

    args = parser.parse_args()

    # read the config as a dictionary from a yaml
    with open(args.config, "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    # specify path to jsonl file with ids
    data_path = args.data_path
    diarized_path_txt = args.diarized_path_txt
    diarized_path_json = args.diarized_path_json
    gold_path = args.gold_path
    input_dir = args.input_dir
    output_dir = args.output_dir
    run_name = args.run_name

    if args.segments is not None:
        config['segments'] = int(args.segments)

    output = run_eval(config, data_path, diarized_path_txt, diarized_path_json, gold_path, input_dir, output_dir)
    
    with open(output_dir + str("/") + run_name + '.json', "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    main()