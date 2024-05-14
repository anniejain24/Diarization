import pandas as pd
from promptflow import tool
import json

@tool
def get_timestamps(input1: list) -> str:
    out = ''
    for line in input1:
        out += line['text'] + ' (' + str(line['timestamp'][0]) + ', ' + str(line['timestamp'][1]) + ')' + '\n'

    return out