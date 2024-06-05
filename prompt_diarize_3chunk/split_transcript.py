
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: list, input2: int) -> list:
    out = ''
    
    if input2==1:
        for line in input1[:int(len(input1)/3)]:
            out += line['text']
    if input2==2: 
        for line in input1[int(len(input1)/3):2*int(len(input1)/3)]:
            out += line['text']
    if input2==3: 
        for line in input1[2*int(len(input1)/3):]:
            out += line['text']

    return out
