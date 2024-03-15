data = {
   "instruction":"Improve the following piece of code that throws multiple errors. You should also write a function test to make sure your code runs correctly.\n\nInitial code:\ndef my_function(x, y, z):\n    a = x + 5y - z\n    print x, y, z, a\n    return a, b, c, d\n\nYour task is not only to resolve the errors but also to improve the code, making it efficient and well-structured. End your task by writing a function test to ensure your code is running as expected. \n\nNote: The context of this coding exercise is language-agnostic. Therefore, be aware of potential differences in syntax and functionalities among different programming languages, and adjust your answer with respect to the programming language of your choice.",
   "output":"The initial code contains some errors which include:\n\n1. The syntax for the arithmetic operation is wrong, it should be 5*y instead of 5y.\n2. Variables \\'b\\',\\'c\\', and \\'d\\' are used without being declared.\n3. Using commas in print function may cause errors in some programming languages.\n\nBelow is the corrected code in Python:\n\n```python\ndef my_function(x, y, z):\n    a = x + 5*y - z\n    print (x, y, z, a)\n    return a\n```\n\nThe above code will calculate the expression x + 5*y - z and return the result as \\'a\\'. Then it will print the values of x, y, z, and a.\n\nThe function test could look something like this:\n\n```python\ndef test_function():\n    result = my_function(1, 2, 3)\n    assert(result == -2), \"The values seem to be incorrect.\"\n    print(\"my_function passed.\")\n```\n\nAbove code will test my_function using the inputs (1, 2, 3). If the output is not -2, it will print \"The values seem to be incorrect.\" If the output is correct, it will print \"my_function passed.\"\n\nSo, when we run the test function:\n\n```python\ntest_function()\n```\n\nIt will test our my_function to see if it\\'s working as expected. The \"assert\" statement is used to let the program test if the condition is true. If the condition is false, the program throws an AssertionError exception.\n\nNote: The above codes are in Python, if these are converted in other language, kindly do the necessary syntax adjustments.",
   "prompt":"[Instruction]:\nImprove the following piece of code that throws multiple errors. You should also write a function test to make sure your code runs correctly.\n\nInitial code:\ndef my_function(x, y, z):\n    a = x + 5y - z\n    print x, y, z, a\n    return a, b, c, d\n\nYour task is not only to resolve the errors but also to improve the code, making it efficient and well-structured. End your task by writing a function test to ensure your code is running as expected. \n\nNote: The context of this coding exercise is language-agnostic. Therefore, be aware of potential differences in syntax and functionalities among different programming languages, and adjust your answer with respect to the programming language of your choice.\n\n[Response]:",
   "text":"[Instruction]:\nImprove the following piece of code that throws multiple errors. You should also write a function test to make sure your code runs correctly.\n\nInitial code:\ndef my_function(x, y, z):\n    a = x + 5y - z\n    print x, y, z, a\n    return a, b, c, d\n\nYour task is not only to resolve the errors but also to improve the code, making it efficient and well-structured. End your task by writing a function test to ensure your code is running as expected. \n\nNote: The context of this coding exercise is language-agnostic. Therefore, be aware of potential differences in syntax and functionalities among different programming languages, and adjust your answer with respect to the programming language of your choice.\n\n[Response]:The initial code contains some errors which include:\n\n1. The syntax for the arithmetic operation is wrong, it should be 5*y instead of 5y.\n2. Variables \\'b\\',\\'c\\', and \\'d\\' are used without being declared.\n3. Using commas in print function may cause errors in some programming languages.\n\nBelow is the corrected code in Python:\n\n```python\ndef my_function(x, y, z):\n    a = x + 5*y - z\n    print (x, y, z, a)\n    return a\n```\n\nThe above code will calculate the expression x + 5*y - z and return the result as \\'a\\'. Then it will print the values of x, y, z, and a.\n\nThe function test could look something like this:\n\n```python\ndef test_function():\n    result = my_function(1, 2, 3)\n    assert(result == -2), \"The values seem to be incorrect.\"\n    print(\"my_function passed.\")\n```\n\nAbove code will test my_function using the inputs (1, 2, 3). If the output is not -2, it will print \"The values seem to be incorrect.\" If the output is correct, it will print \"my_function passed.\"\n\nSo, when we run the test function:\n\n```python\ntest_function()\n```\n\nIt will test our my_function to see if it\\'s working as expected. The \"assert\" statement is used to let the program test if the condition is true. If the condition is false, the program throws an AssertionError exception.\n\nNote: The above codes are in Python, if these are converted in other language, kindly do the necessary syntax adjustments."
}

prompt = "[fooo]:\nname me some cold blooded animals\n\n[Response]:\n"
bs = "jdlsaj;fdsa dls;ajfd;a dfjdaklfd jsak; fsajfiajeiga sdkga idgdhangia n"

prompts = prompt + bs + prompt + bs


def mask_prompt(target: str):
   if not target.startswith("["):
      target = "[" + target
   if not target.endswith("]"):
      target += "]"
      
         
def generate_mask(string:str, prompt: str, censor: str):
    mask = ['1'] * len(string)
    is_prompt = True
    counter = 0
    for i, char in enumerate(string):
        if not is_prompt:
            mask[i] = "0"

        if counter > 0:
            if not is_prompt and char == prompt[counter]:
                counter += 1
            elif  is_prompt and char == censor[counter]:
                counter += 1
            else:
                counter = 0
        
        if counter == len(prompt) and not is_prompt:
            mask[i-counter+1:i+1] = ["1"] * counter
            counter = 0
            is_prompt = True
            
        if counter == len(censor) and is_prompt:
            mask[i-counter+1:i+1] = ["0"] * counter
            counter = 0
            is_prompt = False
            
        if prompt.startswith(char) or censor.startswith(char):
            counter = 1

    return ''.join(mask)
 
def filter_string(mask:str, string:str):
    filtered_list = []
    current_chunk = ''
    for char, m in zip(string, mask):
        if m == '1':
            current_chunk += char
        elif current_chunk:
            filtered_list.append(current_chunk)
            current_chunk = ''
    if current_chunk:
        filtered_list.append(current_chunk)
    return filtered_list
 
# Example usage:
string = "[prompt] hey there\n\n[censor]hello back"
string += string
prompt = "[prompt]"
censor = "[censor]"
mask = generate_mask(string, prompt, censor)
filtered_list = filter_string(mask, string)

