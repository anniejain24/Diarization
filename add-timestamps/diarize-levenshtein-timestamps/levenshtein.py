import pandas as pd
import json
from strsimpy.normalized_levenshtein import NormalizedLevenshtein
import math
from promptflow import tool

@tool
def timestamp(diar: str, id: str) -> list:
    id = id.split('.')[0]

    def read_transcript(id):

        path_to_data_folder = '/archive/shared/sim_center/shared/ameer/'
        # lookinto this dictionary to find the path
        # can also manually create the path and it would be faster but not by much
        merged_lookup = pd.read_csv(path_to_data_folder + 'grade_lookupv5.csv')

        path = merged_lookup[merged_lookup.id == id].path.iloc[0]

        path = path[:-4] + '.json'

        # Opening JSON file
        f = open(path)

        # returns JSON object as 
        # a dictionary
        json_transcript = json.load(f)
        
        f.close()

        return json_transcript

    def diarized_lines(diar):
        lines = diar.split('\n')
        for line in lines:
            if line == '': lines.remove(line)
        return lines

    transcript = read_transcript(id)
    diar = diarized_lines(diar)

    normalized_levenshtein = NormalizedLevenshtein() 
    new_transcript = []

    for line in diar:
        if line == '\n': continue
        max_sim = -math.inf
        timestamp = ''
        for line2 in transcript:
            this_sim = normalized_levenshtein.similarity(line, line2['text'])
            if this_sim > max_sim:
                max_sim = this_sim
                timestamp = line2['timestamp']
        new_transcript.append({'text': line, 'timestamp': timestamp})
        #print({'text': line, 'timestamp': timestamp}, max_sim)
    
    path = '/archive/shared/sim_center/shared/annie/diarized-levenshteined-3chunk/'
    with open(path + id + ".json", "w") as outfile: 
        json.dump(new_transcript, outfile)

    return new_transcript

