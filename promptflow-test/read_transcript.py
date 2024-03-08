import pandas as pd
from promptflow import tool

@tool
def read_transcript_from_id(input1: str) -> dict:

    input1 = input1.split('.')[0]
    path_to_data_folder = '/archive/shared/sim_center/shared/ameer/'
    # lookinto this dictionary to find the path
    # can also manually create the path and it would be faster but not by much
    merged_lookup = pd.read_csv(path_to_data_folder + 'grade_lookupv5.csv')

    path = merged_lookup[merged_lookup.id == input1].path.iloc[0]

    with open(path, 'r') as file:
        transcript = file.read()

    return transcript[20:]
    # return input1
