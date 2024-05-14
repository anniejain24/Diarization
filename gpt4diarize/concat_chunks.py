import pandas as pd
from promptflow import tool
import json

@tool
def concatenate(input1: str, input2: str, input3: str) -> str:
    transcript = input1 + '\n\n' + input2 + '\n\n' + input3

    out = ''
    for line in transcript.split('\n\n'):
        if line == '\n': continue
        out += line + ' (??, ??)' + '\n' 
    
    return out
