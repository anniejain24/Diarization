
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: list, input2: int) -> list:
    out = ''

    for line in input1:
        out += line['text'] 
    
    if input2==1: return out[:int(len(out)/3)]
    if input2==2: return out[int(len(out)/3):2*int(len(out)/3)]
    if input2==3: return out[2*int(len(out)/3):]

    return out
