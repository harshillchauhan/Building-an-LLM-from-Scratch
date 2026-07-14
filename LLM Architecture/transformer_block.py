import torch
import torch.nn as nn

#! GOAL : To implement a complete transformer block which includes all its functions.

#? Importing Multi-Head attention class we created earlier
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

#? Importing LayerNorm class we created earlier
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5                                 #? Small constant added to prevent division by 0
        self.scale = nn.Parameter(torch.ones(emb_dim)) #? Learnable parameter, starts as 1, scales normalized values
        self.shift = nn.Parameter(torch.zeros(emb_dim)) #? Learnable paramter, starts at 0, shifts normalized values
        #! NOTE : Together scale and shift let the model 'undo' normalization if needed giving flexibility.
    
    def forward(self, x):
        mean = x.mean( dim = -1, keepdim = True)
        var = x.var( dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift
    
#? Importing GELU activation function we created earlier
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))))
    
#? Importing FeedForward class we created earlier
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        #? First nn.Linear layer --> Expands the embedding dimension by a factor of 4 (768 -> 3072)
        #? GELU() --> Adds non-linearity 
        #? Second nn.Linear layer --> Projects back down to the original embedding dimesnion (3072 -> 768)
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"],4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
    
    def forward(self,x):
        return self.layers(x)
    

    
#? Implementing a transformer block
class TransformerBlock(nn.Module):
    def __init__ (self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(
        d_in = cfg["emb_dim"],
        d_out = cfg["emb_dim"],
        context_length = cfg["context_length"],
        num_heads = cfg["n_heads"],
        dropout = cfg["drop_rate"],
        qkv_bias = cfg["qkv_bias"]
    )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = xx = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
    

#^ The TransformerBlock class includes a multi-head attention mechanism (MultiHeadAttention) and a feed forward network (FeedFroward), both configured based on a provided configuration dictionary (cfg), such as GPT_CONFIG_124M.

#^ Layer Normalization (;ayerNorm) is applied before each of these two components, and dropout is applied after them to regularize the model and prevent overfitting. This is also called as Pre-Layer Norm.

#^ The class also implements the forward pass, where each component is followed by a shortcut connection that adds the input of the block to its output. This critical feature helps gradients flow through the network during training and improves the learning of deep models.

#? GPT-2 Model Configuration
GPT_CONFIG_124M = {
    "vocab_size" : 50257,           #? Vocabulary Size
    "context_length" : 1024,        #? Context Length
    "emb_dim" : 768,                #? Embedding Dimension
    "n_heads" : 12,                 #? Number of attention heads
    "n_layers" : 12,                #? Number of layers
    "drop_rate" : 0.1,              #? Droput rate
    "qkv_bias" : False              #? Query-Key-Value bias
}

#? Instantiating a transformer block and feeding it some sample data
torch.manual_seed(123)
x = torch.rand(2,4,768)
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)

print("Input shape :", x.shape)
print("Output shape :", output.shape)
#? OUTPUT : 
#? Input shape : torch.Size([2, 4, 768])
#? Output shape : torch.Size([2, 4, 768])

#^ The transformer block mentains the input dimensions in its output, indicating that the transformer architecture processes sequences of data without altering their shape throughout the network.
#^ However, the output is a context vector that encapsulates information from the entore input sequence. 
#^ This means that while the physical dimensions of the sequence remain unchanged as it passes through the transformer bloack, the content of each output vector is re-encoded to integrate contextual informaton across the entire input sequence.

#! CONCLUSION : The transformer block combines layer normalization, the feed forward network, GELU activation and shortcut connection.
