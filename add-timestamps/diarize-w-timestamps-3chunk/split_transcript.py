
from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def my_python_tool(input1: list, input2: int) -> list:
    transcript = ''

    if input2==1: transcript = input1[:int(len(input1)/5)]
    if input2==2: transcript = input1[int(len(input1)/4):2*int(len(input1)/4)]
    if input2==3: transcript = input1[2*int(len(input1)/4):3*int(len(input1)/4)]
    if input2==4: transcript = input1[3*int(len(input1)/4):]


    return transcript
