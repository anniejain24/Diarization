import argparse
import yaml
import pandas as pd
import os
from typing import List
import json
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import difflib


def run_eval(
        config: dict,
        data_path: str,
        diarized_path_txt: str,
        diarized_path_json: str,
        output_dir: str
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

    levenshtein = config["levenshtein"]
    diff = config["diff"]
    preservation = config["preservation"]
    accuracy_basic = config["accuracy_basic"]
    accuracy_baseline_normed = config["accuracy_baseline_normed"]
    accuracy_opposite_normed = config["accuracy_opposite_normed"]
    segments = config["segments"]
    input_json = config["input_json"]
    input_json_and_txt = config["input_json_and_txt"]

    if input_json and not input_json_and_txt:
        diarized_path_txt = None
    
    if not input_json and not input_json_and_txt:
        diarized_path_json = None

    scores = {}

    if preservation:
        scores['preservation'] = calc_preservation(config, ids, diar_path_txt=diarized_path_txt, diar_path_json=diarized_path_json)

    return ids, scores

    

def read_transcript_from_id(transcript_id: str, segments: int, return_json: bool = False) -> List[str]:

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

    if segments == 1:

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

            for n in range(segments):

                start = n * int(len(lines) / segments)
                end = (n + 1) * int(len(lines) / segments)
                if n == segments - 1:
                    end = len(lines)

                json_transcript["chunk " + str(n)] = lines[start:end]

            return json_transcript

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
            # append to transcript
            transcript_chunks.append(transcript)

        transcript = transcript_chunks

    return transcript


def reconstruct_transcript(path: str, id: str, segments: int) -> List[str]:

    transcript = ''
    path = path + id + '.txt'
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

            # append to transcript
        transcript_chunks.append(transcript)

    transcript = transcript_chunks

    return transcript

def reconstruct_transcript_from_json(path: str, id: str, segments: int) -> List[str]:

    path = path + id + '.json'
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

        # clean up any line breaks
        resid_lines = transcript.split('\n')
        transcript = ''
        for line in resid_lines:
            transcript += line
            
        transcript_chunks.append(transcript)
    
    transcript = transcript_chunks
    
    return transcript


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

def main():

    parser = argparse.ArgumentParser(description="Run Models")

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
    output_dir = args.output_dir

    print(run_eval(config, data_path, diarized_path_txt, diarized_path_json, output_dir))


if __name__ == "__main__":
    main()