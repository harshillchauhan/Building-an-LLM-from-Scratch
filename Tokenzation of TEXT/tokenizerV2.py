import re
import os
os.chdir(r'C:\Bulding my own LLM\Tokenzation of TEXT')

#! GOAL : Modifying the tokenizer to handle unknown words.

#? We will be adding special tokens to a vocabulary to deal with certain contexts.
#? We will add an <|unk|> token to represent new and unknown words that were not part of the training data and thus not part of the existing vocabulary.
#? We will add an <|endoftext|> token that we can use to seperate two unrelated text sources.

with open("the-verdict.txt","r",encoding="utf-8") as f:
    raw_text = f.read();

print("Total Number of Characters : ", len(raw_text))

preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', raw_text)
preprocessed =  [item.strip() for item in preprocessed if item.strip()]
print(len(preprocessed))    #? This return 4690 which is the number of tokens in this text

all_tokens = sorted(list(set(preprocessed)))
all_tokens.extend(["<|endoftext|>", "<|unk|>"])
vocab = {token : integer for integer, token in enumerate(all_tokens)}

print(len(vocab.items()))   #? This prints the new vocab size of 1132 which was previously 1130 as we have added two new tokens <|endoftext|> and <|unk|>.

for i , item in enumerate(list(vocab.items())[-5:]) :
    print(item)

'''
The above code snippet prints : 
('younger', 1127)
('your', 1128)
('yourself', 1129)
('<|endoftext|>', 1130)
('<|unk|>', 1131)
Hence, verifying the addition of the two new tokens.
'''

#? Creating a tokenizer class that replaces unknown words with <|unk|> tokens

class SimpleTokenizerV2:
    def __init__(self,vocab):
        self.str_to_int = vocab
        self.int_to_str = { i : s for s, i in vocab.items()}

    def encode(self,text):
        preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        #? The line below replaces unknown words with <|unk|> tokens
        preprocessed = [item if item in self.str_to_int
                        else "<|unk|>" for item in preprocessed]
        
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids

    def decode(self,ids):
        text = " ".join([self.int_to_str[i] for i in ids])

        text = re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text
    
#! NOTE : Compared to the SimpleTokenizerV1, the new SimpleTokenizerv2 replaces unknown words with <|unk|> tokens.

text1 = "Hello, do you like tea?"
text2 = "In the sunlit terraces of the palave."
text = " <|endoftext|> ".join((text1, text2))
print(text)

tokenizer = SimpleTokenizerV2(vocab)
print(tokenizer.encode(text))   #? Tokenizing the string text and printing the token ID's

#^ OUTPUT : [1131, 5, 355, 1126, 628, 975, 10, 1130, 55, 988, 956, 984, 722, 988, 1131, 7]

print(tokenizer.decode(tokenizer.encode(text))) #? Detokenizing the string text

#^ OUTPUT : <|unk|>, do you like tea? <|endoftext|> In the sunlit terraces of the <|unk|>.

#! NOTE : The shortcoming of using <|unk|> for unknown tokens is that when we want to detokenize our text from the token ID's, the unknown words will be replaced by <|unk|>.