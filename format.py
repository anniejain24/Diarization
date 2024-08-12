import argparse
import json
import argparse
import yaml
import pandas as pd
import os
import nltk
from nltk.tokenize import sent_tokenize
from typing import List

def format(
        data_path: str,
        input_dir: str,
        output_dir: str,
) -> tuple[List[str], List[str]]:

    nltk.download('punkt_tab')
    
    # extract transcript ids from file
    # Load data from JSON file into list of ids
    with open(data_path, "r") as f:
        ids = list(f)
    id_list = []
    for id in ids:
        id = json.loads(id)
        id_list.append(id["id"])
    ids = id_list

    for id in ids:
        path = input_dir + str("/") + id + ".txt"
        
        with open(path, 'r') as file:
            transcript = file.read()
        
        sents = sent_tokenize(transcript)

        json_sents = []
        for sent in sents:
            this = {}
            this['text'] = sent
            json_sents.append(this)
        
        save_file(json_sents, output_dir, id)

    return 'reformatting done, new files in ' + output_dir

def save_file(json_sents: List[dict], output_dir: str, id: str) -> str:

    with open(output_dir + str("/") + id + '.json', 'w') as file:
        json.dump(json_sents, file)
    
    return None

def main():

    parser = argparse.ArgumentParser(description="Run Models")

    parser.add_argument(
        "--data_path",
        required=False,
        help="path to list of ids/file names",
        default="helper_files/test-ids.jsonl",
    )

    parser.add_argument(
        "--input_dir",
        required=False,
        help="path with .txt transcript inputs",
        default="input_txt/",
    )

    parser.add_argument(
        "--output_dir",
        required=False,
        help="path to save new json formatted transcripts",
        default="temp/",
    )

    args = parser.parse_args()

    data_path = args.data_path
    input_dir = args.input_dir
    output_dir = args.output_dir

    print(format(data_path, input_dir, output_dir))


if __name__ == "__main__":
    main()