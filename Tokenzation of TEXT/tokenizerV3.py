import tiktoken

#! The previous two tokenizers (SimpleTokeinzerV1 and SimpleTokenizerV2) had a major issue. The issue occurs when the LLM encounters a word that was not a part of the training dataset, the tokenizer were not able to handle them entirely (SimpleTokeinzerV1) or properly (SimpleTokenizerV2). 

#? To deal with this issue we can use an encoding method called as BPE or Byte Pair Encoding.
#? It allows the LLM to breakdown words that aren't in the predefined vocabulary into smaller subwords or even individual characters, enabling it to hadle out-of-vocabulary words.

#?For instance, if GPT-2's vocabulary doesn't have the word "unfamiliarword," it might tokenize it as ["unfam", "iliar", "word"] or some other subword breakdown, depending on its trained BPE merges


#? Instantiating the BPE tokenizer from tiktoken
tokenizer = tiktoken.get_encoding("gpt2")

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces""of someunknownPlace.")
#? The reason for using allowed_special is to specify not to treat <|endoftext|> as an unknown or invalid token rather allow it to remain in the vocabulary and encode it properly.
ids = tokenizer.encode(text,  allowed_special={"<|endoftext|>"})
print(ids)

#? Checking the token ID assigned to <|endoftext|> token which is 50256
# ids = tokenizer.encode("<|endoftext|>", allowed_special={"<|endoftext|>"})
# print(ids)

#? Converting the token ID's back into text using decode method
strings = tokenizer.decode(ids)
print(strings)

#? Testing the Byte Pair Encoding Tokenizer on arbitary text
text = "fahfahfaffhkasfbeghAJKBJC VN"
ids = tokenizer.encode(text)
print(ids)

#? Output : [69, 993, 69, 993, 69, 2001, 71, 42749, 69, 1350, 456, 32, 41, 22764, 34382, 569, 45]

#? The above output showcases that we the model in now succesfully able to handle text that does not pre exist in the training data set (vocabulary)

#! The ability to break down unknown words into individual characters ensures that the tokenizer and the LLM that is trained with it can process any text, even if cotaines words that are not present in the training data.