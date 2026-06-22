import re
import os 
os.chdir(r'C:\Bulding my own LLM\Tokenzation of TEXT')

#! GOAL : To implement a complete tokenizer class in python with an encode method that splits the text into tokens and carries out the string-to-integer mapping to praduce tokken ID's via the vocabulary. In addidtion, we implement a decode method that carries out the reverse integer-to-string mapping to convert the token ID's back into text.

text = """"It's the last he painted, you know," Mrs. Gisburn said with pardonable pride."""

#? We have implemented the simple tokenizer onto a simple text. We can also apply it onto the whole story as well using the code below. But for keeping the project simple, we will use a shorter text.

# with open('the-verdict.txt','r',encoding="utf-8") as f:
#     text = f.read()

#? The code bellow is used to tokenize the entire text
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
preprocessed =  [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))    

print(preprocessed[:30])    #? Verifying if the tokenization is correct

#? sorted() is used to sort all the tokens in an alphabetical order while set() is used to store only unique tokens into all_words.
all_words = sorted(set(preprocessed))
vocab_size = len(all_words)
print(vocab_size)   #? Returns the number of unique tokens

#? Creating a variable vocab and printing the first 51 entries for illustration purposes.
vocab = { token : integer for integer, token in enumerate(all_words)}
for i, item in enumerate(vocab.items()):
    print(item)
    if i >= 50 :
        break

class SimpleTokenizerV1:
    def __init__(self, vocab):
        #? self.str_to_int = vocab --> Stores the voacbulary as a class attribute for access in the encode and decode methods
        self.str_to_int = vocab

        #? self.int_to_str = {i:s for s,i in vocab.items()} --> Creates an inverse vocabulary that maps token ID's back to the original text tokens
        self.int_to_str = {i:s for s,i in vocab.items()}
    
    #? def encode(self, text): --> This processes the input text into token ID's
    def encode(self, text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
                                
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    #? def decode(self, ids): --> This converts token ID's back into text     
    def decode(self, ids):
        text = " ".join([self.int_to_str[i] for i in ids])
        #? Replace spaces before the specified punctuations
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
tokenizer = SimpleTokenizerV1(vocab)

ids = tokenizer.encode(text)
print(ids)

print( tokenizer.decode(ids))

#? The Error in the follwing code is KeyError: 'Hello' and is caused due to the fact that the word 'Hello' never occured in the  “The Verdict” short story.
# text = "Hello, do yoy like tea?"
# print(tokenizer.encode(text))

#! The ERROR above highlights the need to consider large and diverse training sets to extend the vocabulary when working on LLM's.