import tiktoken
import torch 
from torch.utils.data import Dataset, DataLoader
import os 
os.chdir(r"c:\Bulding my own LLM\Tokenzation of TEXT")

#!GOAL : Implementing an effcient DATA LOADER that iterates over the input dataset and returns the inputs and targerts as PyTorch tensors.

#? PyTorch tensors can be thought od as multi-dimensioanal arrays.

#? We are interested in returning two tensors. 
#? Input Tensor : Containg the text that the LLM sees
#? Target Tensor : Containing the targets for the LLM to predict

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
    
#? The GPTDatasetV1 class is based on PyTorch Dataset class and defines how individual rows are fetched from datase, where each row consists of a number of token ID's (based on max_length) assigned to an input_chunk tensor.The target_chunk tensor contains the corresponding targets.


#? Creating a data loader to generate batches with input-with pairs
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


#? Testing the dataloader with a batch size of 1 for an LLM with an context size of 4
with open("the-verdict.txt","r",encoding = "utf-8") as f: 
    raw_text = f.read()

dataloader = create_dataloader_v1(raw_text, batch_size = 1, max_length = 4, stride = 1, shuffle = False)
data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

#? OUTPUT : [tensor([[  40,  367, 2885, 1464]]), tensor([[ 367, 2885, 1464, 1807]])]
#? The first_batch variable contains two tensors : 
#? First tensor stores the input token ID's
#? Second tensor stores the target token ID's

#! NOTE : The input size of 4 is quite small and is chosen only for simplicity. It is common to train LLM's with input sizes of at least 256.

#? Explaining stride
second_batch = next(data_iter)
print(second_batch)
#? OUTPUT : [tensor([[ 367, 2885, 1464, 1807]]), tensor([[2885, 1464, 1807, 3619]])]

#? If we compare first and second batches, we can see that the second batch's token ID's are shifted by one position.
#! The stride setting determines the number of positios the input shifts across batches, emulatind a sliding window approach.

#? Using data loader to sample with a batch size greater than 1.

dataloader = create_dataloader_v1(raw_text, batch_size = 8, max_length = 4, stride = 4, shuffle = False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("Inputs:\n", inputs)
print("\nTargets:\n",targets)

#? This prints : 
#? Inputs:
#?  tensor([[   40,   367,  2885,  1464],
#?         [ 1807,  3619,   402,   271],
#?         [10899,  2138,   257,  7026],
#?         [15632,   438,  2016,   257],
#?         [  922,  5891,  1576,   438],
#?         [  568,   340,   373,   645],
#?         [ 1049,  5975,   284,   502],
#?         [  284,  3285,   326,    11]])
#?
#? Targets:
#?  tensor([[  367,  2885,  1464,  1807],
#?         [ 3619,   402,   271, 10899],
#?         [ 2138,   257,  7026, 15632],
#?         [  438,  2016,   257,   922],
#?         [ 5891,  1576,   438,   568],
#?         [  340,   373,   645,  1049],
#?         [ 5975,   284,   502,   284],
#?         [ 3285,   326,    11,   287]])

#! NOTE : The batch size is a tradeoff and a hyperparameter to experiment when training LLM's

#?Small batch sizes require less memory during training but lead to more noisy model updates.