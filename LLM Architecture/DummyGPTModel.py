import torch 
import torch.nn as nn
import tiktoken

#! GOAL : Implement a placeholder GPT model architecture class

#? GPT-2 Model configuration 
GPT_CONFIG_124M = {
    "vocab_size" : 50257,           #? Vocabulary Size
    "context_length" : 1024,        #? Context Length
    "emb_dim" : 768,                #? Embedding Dimension
    "n_heads" : 12,                 #? Number of attention heads
    "n_layers" : 12,                #? Number of layers
    "drop_rate" : 0.1,              #? Droput rate
    "qkv_bias" : False              #? Query-Key-Value bias
}

#^ Configuration information
#^ vocab_size : Refers to a vocabulary of 50,257 words, as used in the BPE Tokenizer.
#^ context_length : Denotes the maximum number of input tokens that the model can hanle via positional embeddings.
#^ emb_dim : Represents the embedding size, transforming each token into a 768-dimensional vector.
#^ n_heads : Indicates the count of attention heads in the multi-head attention mechanism.
#^ n_layers : Specifies the number of transformer blocks in the model.
#^ drop_rate : Indicates the intensity of the dropout mechanism.
#^ akv_bias : Determines wether to include a bias vector in the Linear layers of the multi-head attenton query, key and value computations.

#? Implementing GPT Place holder architecture
class DummyGPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"],cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[DummyTransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = DummyLayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias = False)

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device = in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits
    
class DummyTransformerBlock(nn.Module):
    def __init__(seslf, cfg):
        super().__init__()

    def forward(self, x):
        return x
    
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5):
        super().__init__()
    
    def forward(self, x):
        return x

#? The DummyGPTModel consists of : 
#? --> Token and positional embeddings 
#? --> Dropout mechanism
#? --> A series of transformer blocks (DummyTransformerBlock)
#? --> A final layer normalization (DummyLayerNorm)
#? --> A linear output layer

#! NOTE : For the timebeing we are using placeholders (DummyLayerNorm and DummyTransformerBlock) for the transformer block and layer normalization. We will develop them later.

#? Tokenizing a batch consisiting of two inputs for the GPT model using toktoken tokenizer

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim = 0)
print(batch)
#? OUTPUT : 
#? tensor([[6109, 3626, 6100,  345],
#?         [6109, 1110, 6622,  257]])

#? Initializing a new 124-million paramter DummyGPTModel instance and feed it in the tokenized batch
torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M)
logits = model(batch)
print("Output shape:", logits.shape)
print(logits)       #? Raw, unnomrmalized scores for each token in the vocabulary at every position in the sequence.
#? OUTPUT : 
#? Output shape: torch.Size([2, 4, 50257])
#? tensor([[[-1.2034,  0.3201, -0.7130,  ..., -1.5548, -0.2390, -0.4667],
#?          [-0.1192,  0.4539, -0.4432,  ...,  0.2392,  1.3469,  1.2430],
#?          [ 0.5307,  1.6720, -0.4695,  ...,  1.1966,  0.0111,  0.5835],
#?          [ 0.0139,  1.6754, -0.3388,  ...,  1.1586, -0.0435, -1.0400]],
#?
#?         [[-1.0908,  0.1798, -0.9484,  ..., -1.6047,  0.2439, -0.4530],
#?          [-0.7860,  0.5581, -0.0610,  ...,  0.4835, -0.0077,  1.6621],
#?          [ 0.3567,  1.2698, -0.6398,  ..., -0.0162, -0.1296,  0.3717],
#?          [-0.2407, -0.7349, -0.5102,  ...,  2.0057, -0.3694,  0.1814]]],
#?        grad_fn=<UnsafeViewBackward0>)

#! The output tensor has two rows corresponding to the two text sample (txt1 and txt2). Each text sample consists of four tokens and each token is a 50,257 dimensional vector, which matches the size of the tokenizer's vocabulary.