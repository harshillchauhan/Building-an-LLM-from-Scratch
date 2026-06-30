import torch
import torch.nn as nn

#! GOAL : To implement a multi-head class which combines CasualAttention and MultiHeadAttentionWrapper

#? Previously, we have implemented the multi-head attention mechanism using two seperate classes, MultiHeadAttentionWrapper and CasualAttention.
#* Instead of mentaining two seperate classes, we can combine these concepts into a single MultiHeadAttention class.

#? MultiHeadAttentionWrapper vs MultiHeadAttention
#? Wrapper approch (MultiHeadAttentionWrapper) : Implements multi-head attention by creating multiple independent CasualAttention modules and concatenating their outputs.
#? Integrated Approach (MultiHeadAttention) : Implements multi-head attention inside a single class by reshaping Q, K, V into multiple heads, computing attention in parallel, and ten recombining the results.

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33],    
   [0.77, 0.25, 0.10],    
   [0.05, 0.80, 0.55]]   
)

batch = torch.stack((inputs , inputs), dim = 0)

#? Implementing MultiHeadAttetnion class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias = False):
        super().__init__()
        
        #? Safety check to make sure model dimensions are valid for multi-head attention
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"    
        
        self.d_out = d_out
        self.num_heads = num_heads
        
        #? Reducing the projection dimension to match the desired output dimension
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)

        #? Creating a final linear projection layer for the multi-head attention output
        self.out_proj = nn.Linear(d_out, d_out)     #? Purpose : To combine head outputs
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal = 1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape       #? Defines the tensors shape structure 
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        #? Reshaping query, key and value tensors so they can be processed by multiple attention heads in parallel
        keys    = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values  = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        #? Rearranging the dimensions of the tensors so that they are in the right format for multi-head attention
        keys    = keys.transpose(1,2)
        queries = queries.transpose(1,2)
        values  = values.transpose(1,2)

        attn_scores = queries @ keys.transpose(2,3)                 #? Computes dot product for each head
        mask_bool = self.mask.bool()[:num_tokens, : num_tokens]     #? Masks the truncated number of tokens

        attn_scores.masked_fill_(mask_bool, -torch.inf)             #? Uses the mask to fill the attention scores

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim = -1)   #? Apllying Normalization
        attn_weights = self.dropout(attn_weights)       #? Applying dropout to regular weights for regularization

        #? Computing weights sum of values and reordering dimensions
        context_vec = (attn_weights @ values).transpose(1,2)        

        #? Flattens head back into full output dimensions
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)  

        #? Final linear projection to combine the head output 
        context_vec = self.out_proj(context_vec)
        return context_vec

#! Important : Working of MultiHeadAttetnion class 
#* The MultiHeadAttention class takes an integrated approach. It starts with a multi-head layer and then INTERNALLY SPLITS this layer into individual attention heads.
#* Splitting Q, K, V → Achieved by reshaping and transposing tensors with .view and .transpose.
#* Key idea → Divide d_out into num_heads × head_dim..
#* Splitting is then achieved using the '.view' method : a tensor of dimensions (b, num_tokens, d_out) is reshaped to dimension (b, num_tokens, num_heads, head_dim).
#* The tensors are then transposed to bring the num_heads dimension before the num_tokens dimension, resulting in a shape of (b, num_heads, num_tokens, head_dim).
#* Purpose of transpose → Align queries, keys, and values per head for efficient batched matrix multiplication.
#* After computing the attention weights and context vectors, the context vector is transposed back.
#* Transpose back → Context vectors reshaped to (b, num_tokens, num_heads, head_dim)
#* These vectors are then reshaped (flattened) into the shape (b, num_tokens, d_out), effectively combining the outputs from all the heads.

#? Using the MultiHeadAttention class
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out = 2
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads = 2)
context_vecs = mha(batch)
print(context_vecs)
print("context_vecs.shape:", context_vecs.shape)

#? OUTPUT : 
#? tensor([[[0.3190, 0.4858],
#?          [0.2943, 0.3897],
#?          [0.2856, 0.3593],
#?          [0.2693, 0.3873],
#?          [0.2639, 0.3928],
#?          [0.2575, 0.4028]],
#?
#?         [[0.3190, 0.4858],
#?          [0.2943, 0.3897],
#?          [0.2856, 0.3593],
#?          [0.2693, 0.3873],
#?          [0.2639, 0.3928],
#?          [0.2575, 0.4028]]], grad_fn=<ViewBackward0>)
#? context_vecs.shape: torch.Size([2, 6, 2])

#! NOTE : Even though the MultiHeadAttention class looks more complicated than the MultiHeadAttentionWrapper due to additional reshaping and transposition of the tensors, it is more efficient.
#* The reason is that we only need one matrix multiplication to compute keys, for instance, keys - self.W_key(x) (the same is true for queries and values).