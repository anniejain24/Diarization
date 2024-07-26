import argparse
import yaml
import pandas as pd
import os
from typing import List
import json


def run_eval(
        config: dict,
        data_path: str,
        diarized_path_txt: str,
        diarize_path_json: str,
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
    chunk_num = config["chunk_num"]
    

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


def main():

    parser = argparse.ArgumentParser(description="Run Models")

    parser.add_argument(
        "--diarized_path_txt",
        required=False,
        help="Path to diarized files in .txt form",
        default="/archive/shared/sim_center/shared/annie/diarized/gpt4-all/text",
    )

    parser.add_argument(
        "--diarized_path_json",
        required=False,
        help="Path to diarized files in .json form",
        default="/archive/shared/sim_center/shared/annie/diarized/gpt4-all/json",
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

    # seeing main output to debug
    i = 0
    with open("out.txt", "w") as f:
        for element in output:
            f.write(ids[i])
            i += 1
            f.write(str(element) + "\n")


if __name__ == "__main__":
    main()