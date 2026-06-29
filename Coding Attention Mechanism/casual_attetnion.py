import torch
import torch.nn as nn

#! GOAL : To implement a casual-attention mechanism also called as masked attention mechanism.

#? For many LLM's, we want the self-attention mechanism to consider only the tokens that appear prior to the current position when PREDICTING the next token in the sequence.
#* Casual attention restricts a model to only consider previous and current inputs in a sequence when processing any given token when computing attention scores.

#^ Casual attention mechanism is a contrast to self-attention mechanism, as the self attention mechanism allows access to the entire input sequence at once.

#! NOTE : To achieve casual-attention, for each token processed, we mask out the furutre tokens, which come after the current token in the input text. We mask out the attention weights above the diagnal, and we normalize the non-masked attention weights such that the attention weights sum to 1 in each row.

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

#? First Step : Compute the attention weights using the softmax function
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries @ keys.T
attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)
print(attn_weights)

#? OUTPUT : 
#? tensor([[0.1921, 0.1646, 0.1652, 0.1550, 0.1721, 0.1510],
#?         [0.2041, 0.1659, 0.1662, 0.1496, 0.1665, 0.1477],
#?         [0.2036, 0.1659, 0.1662, 0.1498, 0.1664, 0.1480],
#?         [0.1869, 0.1667, 0.1668, 0.1571, 0.1661, 0.1564],
#?         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.1585],
#?         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#?        grad_fn=<SoftmaxBackward0>)

#? Second Step : Using the PyTorch's tril function to create a mask where the values above the diagonal are zero.
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)

#? OUTPUT :
#? tensor([[1., 0., 0., 0., 0., 0.],
#?         [1., 1., 0., 0., 0., 0.],
#?         [1., 1., 1., 0., 0., 0.],
#?         [1., 1., 1., 1., 0., 0.],
#?         [1., 1., 1., 1., 1., 0.],
#?         [1., 1., 1., 1., 1., 1.]])

#? Third Step : Multiply this masked matrix with the attention weights to zero-out the values above the diagnal
masked_simple = attn_weights * mask_simple
print(masked_simple)

#? OUTPUT : 
#? tensor([[0.1921, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#?         [0.2041, 0.1659, 0.0000, 0.0000, 0.0000, 0.0000],
#?         [0.2036, 0.1659, 0.1662, 0.0000, 0.0000, 0.0000],
#?         [0.1869, 0.1667, 0.1668, 0.1571, 0.0000, 0.0000],
#?         [0.1830, 0.1669, 0.1670, 0.1588, 0.1658, 0.0000],
#?         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#?        grad_fn=<MulBackward0>)

#! NOTE : Wr used * instad of @ in 'masked_simple = attn_weights * mask_simple' as we want to perform element wise multiplication and not matrix multiplication.

#? Fourth Step : Renormalize the attention weights to sum up to 1 again in each row.
rows_sums = masked_simple.sum(dim=-1, keepdim = True)
masked_simple_norm = masked_simple / rows_sums
print(masked_simple_norm)

#? OUTPUT : 
#? tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#?         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#?         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#?         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#?         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#?         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#?        grad_fn=<DivBackward0>)

#! Information leakage
#* When we apply a mask then normalize the attention weights, it might initially appear that information from future tokens (which we intend to mask) could still influence the current token becase their values are a part of the softmax calculation.
#* But after masking and renormalization, the distribution of attention weights is as if it was calculated only among the unmasked positions to begin with. This ensures there is no information leakage from future or otherwise masked tokens as we intended.

#? Implmenting the computation of masked attention weights in a more efficient manner and fewer steps.
mask = torch.triu(torch.ones(context_length, context_length), diagonal = 1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print(masked)

#? OUTPUT : 
#? tensor([[0.2899,   -inf,   -inf,   -inf,   -inf,   -inf],
#?         [0.4656, 0.1723,   -inf,   -inf,   -inf,   -inf],
#?         [0.4594, 0.1703, 0.1731,   -inf,   -inf,   -inf],
#?         [0.2642, 0.1024, 0.1036, 0.0186,   -inf,   -inf],
#?         [0.2183, 0.0874, 0.0882, 0.0177, 0.0786,   -inf],
#?         [0.3408, 0.1270, 0.1290, 0.0198, 0.1290, 0.0078]],
#?        grad_fn=<MaskedFillBackward0>)

#! Why the above code is more efficient?
#* This approcah is efficient because it uses built-in-tensor operations that run in parallel on GPU/CPU, avoiding slow Python LOOPS and ensuring masked positions are handeled cleanly for attention.

#? Applying softmax function to the masked results calulated with the more efficient method
attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim = 1)
print(attn_weights)

#? OUTPUT : 
#? tensor([[1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#?         [0.5517, 0.4483, 0.0000, 0.0000, 0.0000, 0.0000],
#?         [0.3800, 0.3097, 0.3103, 0.0000, 0.0000, 0.0000],
#?         [0.2758, 0.2460, 0.2462, 0.2319, 0.0000, 0.0000],
#?         [0.2175, 0.1983, 0.1984, 0.1888, 0.1971, 0.0000],
#?         [0.1935, 0.1663, 0.1666, 0.1542, 0.1666, 0.1529]],
#?        grad_fn=<SoftmaxBackward0>)

#! NOTE : Before calculating the context vectors for these attention weights, we will perform a minor tweak called dropout to the casual attention mechanism that is useful for reducing overfitting when training LLM's

#? DROPUT : It is technique wehre radnomly selected hidden layer units are ignored during traning, effectively dropping them out.
#* This method helps prevent overfitting by ensuring that a model does not become overly realiant on any specific set of hidden layer units.

#? Applying dropout mask for illustration
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)     #? Specifying dropout rate to 50%
example = torch.ones(6,6)           #? Creating a 6 x 6 matrix of 1's
print(dropout(example))             #? Applying the dropout mask

#? OUTPUT : 
#? tensor([[2., 2., 0., 2., 2., 0.],
#?         [0., 0., 0., 2., 0., 2.],
#?         [2., 2., 2., 2., 0., 2.],
#?         [0., 2., 2., 0., 0., 2.],
#?         [0., 2., 0., 2., 0., 2.],
#?         [0., 2., 2., 2., 2., 0.]])

#! When applying dropout to an attention weight matrix with a rate of 50%, half of the elements in the matrix are randomly set to zero. To compensate for this reduction in active elements, the values of the remaining elements in the matrix are scaled up by a factor of 1/0.5 = 2.
#* This scaling is crucial to metain the overall balance of the attention weights, ensuring that the average influence of the attention mechanism remains consistent during both the training and inference phases.

#? Applying dropout to the attention weight matrix
torch.manual_seed(123)
print(dropout(attn_weights))

#? OUTPUT : 
#? tensor([[2.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#?         [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
#?         [0.7599, 0.6194, 0.6206, 0.0000, 0.0000, 0.0000],
#?         [0.0000, 0.4921, 0.4925, 0.0000, 0.0000, 0.0000],
#?         [0.0000, 0.3966, 0.0000, 0.3775, 0.0000, 0.0000],
#?         [0.0000, 0.3327, 0.3331, 0.3084, 0.3331, 0.0000]],
#?        grad_fn=<MulBackward0>)

#! NOTE :  It is important to emphasize that droput is only used during traning and is disabled afterwards.

