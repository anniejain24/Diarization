import pandas as pd
from promptflow import tool
import json

@tool
def read_transcript_from_id(input1: str) -> str:

    input1 = input1.split('.')[0]
    path_to_data_folder = '/archive/shared/sim_center/shared/ameer/'
    # lookinto this dictionary to find the path
    # can also manually create the path and it would be faster but not by much
    merged_lookup = pd.read_csv(path_to_data_folder + 'grade_lookupv5.csv')

    path = merged_lookup[merged_lookup.id == input1].path.iloc[0]

    path = path[:-4] + '.json'

    # Opening JSON file
    f = open(path)

    # returns JSON object as 
    # a dictionary
    json_transcript = json.load(f)

    #debugging line, change as needed:
    #json_transcript = json_transcript[:8*int(len(json_transcript)/9)]

    # uncomment as necessary for segmenting the transcript in thirds:

    #json_transcript = json_transcript[:int(len(json_transcript)/2)]
    #json_transcript = json_transcript[int(len(json_transcript)/3):2*int(len(json_transcript)/3)]
    #json_transcript = json_transcript[2*int(len(json_transcript)/3):]

    # segmenting the transcript in sixths:

    #json_transcript = json_transcript[:int(len(json_transcript)/6)]
    #json_transcript = json_transcript[int(len(json_transcript)/6):2*int(len(json_transcript)/6)]
    #json_transcript = json_transcript[2*int(len(json_transcript)/6):3*int(len(json_transcript)/6)]
    #json_transcript = json_transcript[3*int(len(json_transcript)/6):4*int(len(json_transcript)/6)]
    #json_transcript = json_transcript[4*int(len(json_transcript)/6):5*int(len(json_transcript)/6)]
    #json_transcript = json_transcript[5*int(len(json_transcript)/6):]

    # segmenting the transcript in ninths:

    #json_transcript = json_transcript[:int(len(json_transcript)/9)]
    #json_transcript = json_transcript[int(len(json_transcript)/9):2*int(len(json_transcript)/9)]
    #json_transcript = json_transcript[2*int(len(json_transcript)/9):3*int(len(json_transcript)/9)]
    #json_transcript = json_transcript[3*int(len(json_transcript)/9):4*int(len(json_transcript)/9)]
    #json_transcript = json_transcript[4*int(len(json_transcript)/9):5*int(len(json_transcript)/9)]
    #json_transcript = json_transcript[5*int(len(json_transcript)/9):6*int(len(json_transcript)/9)]
    #json_transcript = json_transcript[6*int(len(json_transcript)/9):7*int(len(json_transcript)/9)]  
    #json_transcript = json_transcript[7*int(len(json_transcript)/9):8*int(len(json_transcript)/9)] 
    #json_transcript = json_transcript[8*int(len(json_transcript)/9):]  

    transcript = ''
    for line in json_transcript:
        transcript += line['text']

    return transcript
    # return input1
