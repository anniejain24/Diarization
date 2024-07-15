from promptflow import tool


# The inputs section will change based on the arguments of the tool function, after you save the code
# Adding type to arguments and return value will help the system show the types properly
# Please update the function name/signature per need
@tool
def save_file(input1: str, input2: str, input3: str) -> str:
    
    path = input3
    with open(path + input2.split('.')[0] + ".txt", "w") as outfile:
        outfile.write('ID: ' + input2.split('.')[0] + '\n\n' + input1)
    
    return input1