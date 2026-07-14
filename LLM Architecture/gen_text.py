import torch 
import torch.nn as nn
import tiktoken

#! GOAL : Convert tensor outputs of the GPT model back into text.

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
    
#? Importing the complete GPT Model class we created earlier
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

model = GPTModel(GPT_CONFIG_124M)

#^ The process by which a GPT model goes from the output tensors to generated text include decoding the output tensors, selecting the tokens based on probability distribution, and converting these tokens into human readable text.

#^In each step the model outputs a matrix with vectors representing potential next tokens.
#^ The vector coresponding to the next token is extracted and converted into a probability distribution via a softmax funtion.
#^ Within the vector containg the resulting probability scores, the index of the highest value is located, which translates to the token ID.
#^ This token ID is then decoded back into text, praducing the next token in the sequence.
#^ Finally this token is appended to the previous inputs, forming new input sequence for the subsequent iteration.

#! NOTE : This step-by-step process enables the model to generate text sequentially, building coherent phrases and sentences from the initial input context. This process is repeated over many iterations until we reach a user-specified number of generated tokens.

#? A function of the GPT model to generate text
def generate_text_simple(model, idx, max_new_tokens, context_size):
    #? idx is (batch, n_tokens) array of indices in the current context. 
    for _ in range(max_new_tokens):
        
        #? Crops current context if it exceeds the supported context size
        idx_cond = idx[:, -context_size:]

        #? Get the prediction 
        with torch.no_grad():
            logits = model(idx_cond)

        #? Focus only on the last time step
        #? (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
        logits = logits[:, -1, :]

        #? Apply softmax to get probabilities
        probas = torch.softmax(logits, dim = -1)

        #? Get the idx of the vocab entry with the highest probability value
        idx_next = torch.argmax(probas, dim = -1, keepdim = True)

        #? Append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim =1)

    return idx

#^ The above function code demonstrates a generative loop that iterates for a specified number of new tokens to be generated, crops the current context to fit the model's maximum context size, computes the predictions, and then selects the next token based on the highest probability prediction.

#? Trying the generate_text_simple function with "Hello, I am" context as model input 

#? Encoding the input context into token ID's
start_context = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)
#? OUTPUT: 
#? encoded: [15496, 11, 314, 716]
#? encoded_tensor.shape: torch.Size([1, 4])

#? Putting the model into .eval() mode
model.eval()
out = generate_text_simple(
    model = model,
    idx = encoded_tensor,
    max_new_tokens = 6,
    context_size = GPT_CONFIG_124M["context_length"]
)
print("Output:", out)
print("Output length:", len(out[0]))
#? OUTPUT : 
#? Output: tensor([[15496,    11,   314,   716, 15125, 43101, 41958, 43208, 17670, 14241]])
#? Output length: 10

#! NOTE : We put the model into .eval() mode to disable random components which are only used during training, and use the generate_text_simple function on the encoded input tensor.

#? Converting the token ID's back into text using .decode method 
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
#? OUTPUT : Hello, I am reefsdid modifiersamus �ickets

#! NOTE : The above output that the model generated is complete giberish, which is not like the coherent text we expected. 
#* The reason the model is unable to praduce coherent text is that we haven't trained it yet.