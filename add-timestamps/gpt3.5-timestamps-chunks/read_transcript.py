import pandas as pd
from promptflow import tool
import json

@tool
def read_transcript_from_id(input1: str, input2: bool) -> str:

    input1 = input1.split('.')[0]
    diarized = input2
    
    if not diarized: path_to_data_folder = '/archive/shared/sim_center/shared/ameer/'
    if diarized: path_to_data_folder = '/archive/shared/sim_center/shared/annie/GPT4 3-chunk/'
    # lookinto this dictionary to find the path
    # can also manually create the path and it would be faster but not by much

    if not diarized: 
        merged_lookup = pd.read_csv(path_to_data_folder + 'grade_lookupv5.csv')

        path = merged_lookup[merged_lookup.id == input1].path.iloc[0]

        path = path[:-4] + '.json'

        # Opening JSON file
        f = open(path)

        # returns JSON object as 
        # a dictionary
        json_transcript = json.load(f)

        transcript = ''
        for line in json_transcript:
            transcript += line['text'] + ' (' + str(line['timestamp'][0]) + ', ' + str(line['timestamp'][1]) + ')' + '\n'
    
    if diarized:
        path = path_to_data_folder + input1 + '.txt'

        with open(path, 'r') as file:
            lines = file.readlines()
        
        transcript1 = ''
        for line in lines[:int(len(lines)/4)]:
            if line == '\n': continue
            transcript1 +=  line[:-1] + ' (??, ??)' + '\n' 
        
        transcript2 = ''
        for line in lines[int(len(lines)/2):3*int(len(lines)/4)]:
            if line == '\n': continue
            transcript2 +=  line[:-1] + ' (??, ??)' + '\n' 
        
        transcript = [transcript1, transcript2]
        
        
        


    return transcript
    # return input1