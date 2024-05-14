
from promptflow import tool
import json


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def save_file(input1: str, input2: str) -> str:
    lines = input1.split('\n') 
    out = []
    
    for line in lines:
        if line.find(':')==-1: continue
        temp = {}
        temp['speaker'] = line[:line.find(':')]
        temp['text'] = line[line.find(':') + 1: line.find('(')]
        temp['timestamp'] = []
        if (line.find('(') != -1) and ('?' not in line[line.find('('): line.find(')')]):
            temp['timestamp'].append(float(line[line.find('(')+1: line.rfind(',')]))
            temp['timestamp'].append(float(line[line.rfind(',')+2: line.find(')')]))
        out.append(temp)
    
    path = '/archive/shared/sim_center/shared/annie/new-diarized-gpt35t-timestamped/'
    with open(path + input2.split('.')[0] + ".json", "w") as outfile:
        json.dump(out, outfile)
    
    return out
        
        


    
