import tiktoken
import os 
os.chdir(r"c:\Bulding my own LLM\Tokenzation of TEXT")

#? Instantiating the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

#! GOAL : To generate the input-target pairs required for training an LLM

#? LLM's are pre-trained by predicting the next word in a text

with open("the-verdict.txt","r", encoding = "utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))     #?Returns 5145, the total number of tokens in the training set after applying BPE tokenizer

#?Returns all token ID's for the training set
# print(ids)

#? Removing first 50 tokens from the dataset for demonstration purposes. [50:] --> starts from 51 to the end.
enc_sample = enc_text[50:]

#? One of the easiest ways to create input-target pairs for next word prediction task is to create two variables x and y.
#? x contains the input tokens 
#? y contains the targets, which are input shifted by 1

#? The context size determines how many tokens are included in the input. 
#! NOTE : For LLM's like GPT-3, the context-size is much larger than 4(in thousands).
context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1: context_size+1]

print(f"x: {x}")
print(f"y:      {y}")

#^ Output : (For visualizing) 
#? x: [290, 4920, 2241, 287]
#? y:      [4920, 2241, 287, 257]
#? token 290 predicts 4920 | token 290,4920 predicts 2241 | and so on 

for i in range (1 , context_size):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(context, "---->", desired)

#^ OUTPUT: (For visualizing)
#? [290] ----> 4920
#? [290, 4920] ----> 2241
#? [290, 4920, 2241] ----> 287

for i in range (1 , context_size):
    context = enc_sample[:i]
    desired = enc_sample[i]

    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))

#^ OUTPUT : (This outputs the same result as of before but shoes the text instead of token ID's)
#?  and ---->  established
#?  and established ---->  himself
#?  and established himself ---->  in




