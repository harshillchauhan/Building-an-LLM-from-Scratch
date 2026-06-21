# The os command is used to change the directory of our python file so that we can access the text.
import os
import re
os.chdir(r"c:\Bulding my own LLM\Tokenzation of TEXT")


with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read();

print("Total Number of Characters : ", len(raw_text))
print(raw_text[:99])

'''
GOAL : Our Goal is to tokenize this 20,479 character short story into individual words and special characters that we can turn into embeddings for LLM training.
'''

#? This is an example used to learn how we can tokenize text on a simple sentence.
text = "Hello, world. Is this-- a test?"

#? This is a basic tokenizer that seperates each word and special characters
result = re.split(r'([,.:;?_!"()\']|--|\s)',text)
#? This line is added to remove all the empty string that are reuturned previously
result = [item.strip() for item in result if item.strip()]
print(result)

#! Note : The 'r' in line 21 is used to create a raw string. In a raw string, backslashes are treated literally — they’re not interpreted as escape characters. Meaning : print(r"Hello\nWorld")  raw string → prints Hello\nWorld

'''
Tokenizing the entire Edith Wharton's short story!
'''

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed =  [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))    #? This return 4690 which is the number of tokens in this text

print(preprocessed[:30])