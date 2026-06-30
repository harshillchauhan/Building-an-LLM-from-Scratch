import torch
import torch.nn as nn

#! GOAL : To implement a compact casual self attention class 

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33],    
   [0.77, 0.25, 0.10],    
   [0.05, 0.80, 0.55]]   
)

d_in = inputs.shape[1]
d_out = 2

#? Ensuring that the code can handle batches consisting of more than one input
batch = torch.stack((inputs , inputs), dim = 0)
print(batch.shape)
#? OUTPUT : torch.Size([2, 6, 3])

#? Implementing a casual self attention class
class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length , context_length), diagonal = 1))
        #? The above line creates a mask ensuring each token can only attend to itself and the past tokens.
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        #? # Applying causal mask: blocks future tokens by setting scores to -∞
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec 
    
#? Using the CasualAttention class
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, 0.0)      #? 0.0 --> Dropout is set to ineffective throughout
context_vecs = ca(batch)
print("context_vecs.shape:", context_vecs.shape)
#? OUTPUT : context_vecs.shape: torch.Size([2, 6, 2])

#! The resulting context vector is a three-dimensianal tensor, where each token is now represented by a two-dimensioanl embedding