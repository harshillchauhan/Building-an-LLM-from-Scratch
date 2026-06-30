import torch
import torch.nn as nn

#! GOAL : To implement a multi-head attention mechanism

#? Multi-Head refers to dividing the attention mechanism into multiple "heads", each operating independently. In practical terms, implementing multi-head attention involves creating multiple instances of the self-attention mechanism, each with its own weights, and then combining their outputs.

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33],    
   [0.77, 0.25, 0.10],    
   [0.05, 0.80, 0.55]]   
)

batch = torch.stack((inputs , inputs), dim = 0)

class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias = False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length , context_length), diagonal = 1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2)
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec 
    
#? Implementing a wrapper class for multi-head attention
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(d_in, d_out, context_length, dropout, qkv_bias = False)for _ in range(num_heads)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim = -1)
        
#? Using the MultiHeadAttentionWrapper class
torch.manual_seed(123)
context_length = batch.shape[1]
d_in, d_out = 3,2
mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

#? OUTPUT :
#? tensor([[[-0.4519,  0.2216,  0.4772,  0.1063],
#?          [-0.5874,  0.0058,  0.5891,  0.3257],
#?          [-0.6300, -0.0632,  0.6202,  0.3860],
#?          [-0.5675, -0.0843,  0.5478,  0.3589],
#?          [-0.5526, -0.0981,  0.5321,  0.3428],
#?          [-0.5299, -0.1081,  0.5077,  0.3493]],
#?
#?         [[-0.4519,  0.2216,  0.4772,  0.1063],
#?          [-0.5874,  0.0058,  0.5891,  0.3257],
#?          [-0.6300, -0.0632,  0.6202,  0.3860],
#?          [-0.5675, -0.0843,  0.5478,  0.3589],
#?          [-0.5526, -0.0981,  0.5321,  0.3428],
#?          [-0.5299, -0.1081,  0.5077,  0.3493]]], grad_fn=<CatBackward0>)
#? context_vecs.shape: torch.Size([2, 6, 4])

#^ #? context_vecs.shape: torch.Size([2, 6, 4])
#^ First Dimension : Refers to the two input texts(num_head = 2)
#^ Second Dimension : Referes to the 6 tokens in each input
#^ Third Dimension : Refers to the 4-dimensional embedding of each token

#! NOTE : Up to this point, we have implemented a MultiHeadAttentionWrapper that combined multiple single-head attention modules. However, these are processed sequentially in the forward method. 
#* We can improves this implementation by processing heads in parallel.

