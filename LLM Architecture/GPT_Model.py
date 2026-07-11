import tiktoken
import torch
import torch.nn as nn

#! GOAL : To implement a complete GPTModel class and computing the memory requirements for implementing it.

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

#? Implementing the transformer block we created earlier
class TransformerBlock(nn.Module):
    def __init__ (self,cfg):
        super().__init__()
        self.att = MultiHeadAttention(
        d_in = cfg["emb_dim"],
        d_out = cfg["emb_dim"],
        context_length = cfg["n_heads"],
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
    
#? Creating the complete GPT Model class
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
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
    
#! class GPTModel 
#? __init__ constructor
#^ --> The __init__ contructor of the GPTModel class initializes the token and positional embedding layers using the configuration passed via the python dictionary.
#^ --> These embedding layers are responsible for converting input token indices into dense vectors and adding positional information.
#^ --> Next, the __init__ constructor creates a sequential stack of TransformerBlock modules equal to the number of layers specified cfg.
#^ --> Following the transformer blocks, a LayerNorm layer is applied, standardizing the outputs from the transformer blocks to stabilize the learining process.
#^ Finally, a linear output head without bias is defined, which projects the transformer's output into the vocabulary space of the tokenizer to generate logits for each token in the vocabulary.
#? forward method
#^ --> The forward method takes a batch of input token indices, coomputes their embeddings, applies the positional embeddings, passes the sequence through the transformer blocks, normalizes the final output, and then computes the logits, representing the next token's unnormalized probabilities.

    
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

#? Tokenizing a batch consisiting of two inputs for the GPT model using toktoken tokenizer
tokenizer = tiktoken.get_encoding("gpt2")
batch = []
txt1 = "Every effort moves you"
txt2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(txt1)))
batch.append(torch.tensor(tokenizer.encode(txt2)))
batch = torch.stack(batch, dim = 0)

#? Initializing the 124-million-paramteter GPT model
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input batch:\n", batch)
print("\nOutput shape:", out.shape)
print(out)
#? OUTPUT : 
#? Input batch:
#?  tensor([[6109, 3626, 6100,  345],
#?         [6109, 1110, 6622,  257]])
#?
#? Output shape: torch.Size([2, 4, 50257])
#? tensor([[[-0.0965,  0.0260, -0.2693,  ...,  0.7199,  0.5663, -0.7289],
#?          [-0.8844, -0.4454, -0.9419,  ...,  0.2716,  0.9435, -0.9142],
#?          [ 0.2222, -0.1973,  0.5040,  ..., -0.1703,  0.2346, -0.4732],
#?          [-1.0163,  0.2056, -0.0822,  ...,  0.5353,  0.5620, -0.1189]],
#?
#?         [[-0.5499,  0.3848, -0.0855,  ...,  0.0603,  0.4987, -0.5448],
#?          [-0.1834,  0.3969, -0.3001,  ...,  0.3497,  0.8520, -0.2309],
#?          [ 0.2048,  0.9418, -0.3673,  ...,  0.2878,  0.5798, -0.5170],
#?          [-0.5806,  0.4568,  0.1560,  ...,  0.7012,  0.0325, -0.3301]]],
#?        grad_fn=<UnsafeViewBackward0>)

#! NOTE : The output tensor has the shape [2,4,50257], since we passed in two input texts with four tokens each. The last dimension 50257, corresponds to the vocabulary size of the tokenizer.

#? Analyzing the size of the model architecture 
total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters: {total_params:,}")
#? OUTPUT : Total number of parameters: 163,009,536

#! NOTE : Earlier we talked about initializing 124-million parameter GPT model,but the actual number of paramteres is 163-million.
#* REASON : A concept called weight typing, used in the original GPT-2 architecture. It means that the original GPT-2 architecture resuses the weights from the token embedding layer in its output layer.

print("Token embedding layer shape :", model.tok_emb.weight.shape)
print("Output layer shape:", model.out_head.weight.shape)
#? OUTPUT : 
#? Token embedding layer shape : torch.Size([50257, 768])
#? Output layer shape: torch.Size([50257, 768])

#! NOTE : Without weight typing, the model allocates a new matrix for the output layer. But if we use weight typing, it reuses the same tensor that was created for the token embedding layer.

#? Showcasing model parameters with weight typing
total_params_gpt2 = (
    total_params - sum(p.numel()
    for p in model.out_head.parameters())
)
print(f"Number of trainable paramters considering weight typing: {total_params_gpt2:,} ")
#?OUTPUT : Number of trainable paramters Considering weight typing: 124,412,160

#? Computing memory requirements of the 163 million parameters in out GPTModel object
total_size_bytes = total_params * 4
total_size_mb = total_size_bytes / (1024 * 1024)
print(f"Total size of the model: {total_size_mb: .2f} MB")
#? OUTPUT : Total size of the model:  621.83 MB

#! NOTE : Total size of the model is calcuted by assuming each parameter is 32-bit float taking up 4 bytes