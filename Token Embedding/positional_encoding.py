import torch
import tiktoken
from torch.utils.data import Dataset, DataLoader
import os
os.chdir(r"C:\Bulding my own LLM\Token Embedding")

#! GOAL : Adding positional information to token vectors so the model knows the order of the words.

#? Token embeddings are suitable input for an LLM, however their self-attention mechanism doesn't know the position or order of tokens in a sequence.
#? This means that the model can understand the meaning of each token, but not where it appears in the sentence.

#? Although positional independent embeddings of token ID's is good for repraducability purposes. However since self-attention mechanism of LLM's itself is POSITIONAL AGNOSTIC, it is helpfull to inject additional positional information into the LLM.

#? To achieve POSITIONAL Embedding, we can use two borad categories of positional-aware embeddings:
#? 1) Absolute Positional Embeddings --> They are directly associated with specific positions in a sequence.
#? 2) Relative Positional Embeddings --> They encode the distance between tokens. This means that the model learns the relationship in terms of "how far apart" rather than "at which exact position".

#? GPTDatasetV1 class prepares training samples for a GPT-style language model.
class GPTDatasetV1(Dataset) : 
    def __init__ (self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        #? Tokenizes the entire text
        token_ids = tokenizer.encode(txt)

        #? Uses a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i : i + max_length]
            target_chunk = token_ids[ i + 1 : i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    #? Returns the total number of rows in the dataset
    def __len__ (self):
        return len(self.input_ids)
    
    #? Retuerns a single row from the dataset
    def __getitem__ (self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, batch_size = 4, max_length = 256, stride = 128, shuffle = True, drop_last = True, num_workers = 0):
    #? Initializes the tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    #? Creates dataset
    dataset = GPTDatasetV1(txt,tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size = batch_size,
        shuffle = shuffle,
        #? If drop_last = True, then it will drop the last batch if it is shorter than the specified batch_size to prevent loss spikes during training.
        drop_last = drop_last,
        #? The number of CPU processes to use for preprocessing
        num_workers = num_workers
    )
    return dataloader

#? Loading all textual data of "the-verdict.txt" into raw_text
with open("the-verdict.txt","r",encoding = "utf-8") as f: 
    raw_text = f.read()

#? Encoding 50257 input tokens into a 256-dimension vector representation
vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size , output_dim)

max_length = 4 
dataloader = create_dataloader_v1(
    raw_text, batch_size = 8, max_length = max_length, stride = max_length, shuffle = False
)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Token ID's:\n", inputs)
print("\nInputs shape:\n", inputs.shape)
#? OUTPUT : 
#? Token ID's:
#?  tensor([[   40,   367,  2885,  1464],
#?         [ 1807,  3619,   402,   271],
#?         [10899,  2138,   257,  7026],
#?         [15632,   438,  2016,   257],
#?         [  922,  5891,  1576,   438],
#?         [  568,   340,   373,   645],
#?         [ 1049,  5975,   284,   502],
#?         [  284,  3285,   326,    11]])
#? Inputs shape:
#?  torch.Size([8, 4])

#! Token ID tensor is 8 x 4 dimensional, meaning that the data batch consists of eight text samples with four tokens each. stride = max , assigns stride a value of 4 i.e the batch_size.

#? Embedding thses token ID's into 256-dimensioanl vector
token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)
#? OUTPUT : torch.Size([8, 4, 256])

#? For implementing absolute embedding approach, we just need to create another embedding layer that has the same embedding dimension as the token_embedding_layer.

context_length = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)
#? OUTPUT : torch.Size([4, 256])

#? Adding positional embeddings to the token embeddings
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
#? OUTPUT : torch.Size([8, 4, 256])

#! NOTE : Token embedding stores the vector that represents the actual meaning of token itself while positional embedding stores the vector that represents the position of a token in a sequence.