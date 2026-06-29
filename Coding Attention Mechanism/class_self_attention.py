import torch
import torch.nn as nn

#! GOAL : To implement a compact self-attention python class

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

#? Implementing a compact self-attention python class
class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):                                #? Initializes training weight matrices
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys = x @ self.W_key
        queries = x @ self.W_query
        values = x @ self.W_value                                                   
        attn_scores = queries @ keys.T                                              #? Computing attention scores
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)   #? Normalizing these scores
        context_vec = attn_weights @ values                                         #? Computing the contect vectors
        return context_vec
    
#? Using the python class SelfAttention_v1
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
print(sa_v1(inputs))
#? OUTPUT : 
#? tensor([[0.2996, 0.8053],
#?         [0.3061, 0.8210],
#?         [0.3058, 0.8203],
#?         [0.2948, 0.7939],
#?         [0.2927, 0.7891],
#?         [0.2990, 0.8040]], grad_fn=<MmBackward0>)

#! NOTE : To improve SelfAttention_v1 implementation further, we can use PyTorch's nn.Linear layers.
#* nn.Linear layers effectively performs matrix multiplication when the bias units are disabled.
#* A significant advantage of using nn.Linear instead of manually implementing nn.Parameter(torch.rand()) is that nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training.

#? Implementing a slef-attention class using PyTorch's Linear layers
class selfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias = False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
        context_vec = attn_weights @ values
        return context_vec
    
torch.manual_seed(789)
sa_v2 = selfAttention_v2(d_in, d_out)
print(sa_v2(inputs))
#? OUTPUT : 
#? tensor([[-0.0739,  0.0713],
#?         [-0.0748,  0.0703],
#?         [-0.0749,  0.0702],
#?         [-0.0760,  0.0685],
#?         [-0.0763,  0.0679],
#?         [-0.0754,  0.0693]], grad_fn=<MmBackward0>)

#! NOTE : selfAttention_v1 and selfAttention_v2 give different outputs because they use different initial weights for the weight matrices since nn.Linear uses a more SOPHESTICATED weight initialization scheme.

