import pandas as pd
from promptflow import tool
import json

@tool
def concatenate(input1: str, input2: str, input3: str) -> str:
    return input1 + '\n\n' + input2 + '\n\n' + input3
