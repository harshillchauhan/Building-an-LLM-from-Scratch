import torch
import torch.nn as nn
import tiktoken

#! GOAL : To calculate the cross-entropy loss for two sample inputs

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

#? Modified GPT-2 Model Configuration 
GPT_CONFIG_124M = {
    "vocab_size" : 50257,           #? Vocabulary Size
    "context_length" : 256,         #? Context Length
    "emb_dim" : 768,                #? Embedding Dimension
    "n_heads" : 12,                 #? Number of attention heads
    "n_layers" : 12,                #? Number of layers
    "drop_rate" : 0.1,              #? Droput rate
    "qkv_bias" : False              #? Query-Key-Value bias
}

#? Initializing the GPT model we created earlier
model = GPTModel(GPT_CONFIG_124M)

#? Importing Utility function for text-to-token ID conversion we created earlier
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special = {'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

#? Importing Utility function for token ID to text conversion we created earlier
def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

#? Instantiating the tokenizer 
tokenizer = tiktoken.get_encoding("gpt2")

#? Token ID's of two input examples
inputs = torch.tensor([[16833, 3626, 6100],         #? ["every effort moves"]
                        [40, 1107, 588]])           #? ["I really like"]

#? Target token ID's
targets = torch.tensor([[3626, 6100, 345  ],         #? ["effort moves you"]
                       [1107, 588, 11311]])         #? ["really like chocolate"]

#! NOTE : Targets are the inputs but shifted to one position forward. This shifting stratergy is crucial for teaching the model to predict the next token in the sequence.

#^ Steps for calculating the loss for the probabilty scores
#^ STEP 1: Model outputs logits --> After passing input token ID's through the GPT model, we get logits (raw scores)
#^ STEP 2: Converting logits into probabilty --> We apply softmax to calculate probabilities across the vocabulary.
#^ STEP 3: Align with target tokens --> We compare the predicted probabilites with the target tensor (the actual next tokens from the training data)
#^ STEP 4: Take the logarith --> We apply lof() to those probabilities inorder to penalize wrong predictions more confidently.
#^ STEP 5: Compute negative log probabilty --> This is the loss contribution for each token.
#^ STEP 6: Average across tokens --> Sum the losses for all tokens in the sequence and divide by the number of tokens to get the mean loss per token.
#^ STEP 7 : Final scalar loss --> The is what backpropogation uses to update the model's weight

#? Fedding the inputs into the model to calculate logits vectors for the two input example
with torch.no_grad():
    logits = model(inputs)
probas = torch.softmax(logits, dim = -1)
print(probas.shape)
#? OUTPUT : torch.Size([2, 3, 50257])   --> probas tensor containing the probability scores
#? Structure --> 2 is the batch size, 3 is the number of input tokens in each row, 50257 is the embedding dimensionality determined by the voacbulary size. 

#? Applying argmax function to the probabilty scores to obtain the corresponding token ID's 
token_ids = torch.argmax(probas, dim = -1, keepdim = True)
print("Token IDs:\n", token_ids)
#? OUTPUT : 
#? Token IDs:
#?   tensor([[[15047],
#?          [16694],
#?          [30971]],,
#?
#?         [[14206],
#?          [16644],
#?          [18666]]])

#^ Following the conversion from logits to probabilities via the softmax function, the generate_text_simple then converts the probabilty scores back into text.

#? Converting the token ID's back into text
print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
print(f"Output batch 1:"
      f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")
#? OUTPUT : 
#? Targets batch 1:  effort moves you
#? Output batch 1: Lightgre archaeological

#! NOTE : We can observer that the output tokens are quite different from the target tokens we want to generate.

#^ Since the output is not as per our requirements, we want to evaluate the performance of the model's generated text numerically via a loss. Not only is this usefull for measuring the quality of the generated text, but it is also a building block for implementing the training function, which we will use to update the model's weight to improve the generated text.

#? Printing the Initital softmax probability scores corresponding to the target tokens 
text_idx = 0
target_probas_1 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 1:", target_probas_1)
#? OUTPUT : Text 1: tensor([4.6496e-05, 2.2865e-05, 3.1090e-05])

text_idx = 1
target_probas_2 = probas[text_idx, [0,1,2], targets[text_idx]]
print("Text 2:", target_probas_2)
#? OUTPUT : Text 2: tensor([5.4702e-06, 1.5064e-05, 3.8039e-05])

#! NOTE : The main goal of an LLM is to maximize the likelihood of the correct tokken, which involves increasing the probability relative to the other tokens. This ensures the LLM consistently picks the target token - essentially the next word in the sentence - as the next token it generates.

#? Applying logarith to the probabilty scores 
log_probas = torch.log(torch.cat((target_probas_1, target_probas_2)))
print(log_probas)
#? OUTPUT : tensor([ -9.9762, -10.6859, -10.3786, -12.1162, -11.1032, -10.1769])

#? Calculating the average log probabilty
avg_log_probas = torch.mean(log_probas)
print(avg_log_probas)
#? OUTPUT : tensor(-10.7395)

#! NOTE : The main goal here is to get the average log probabilty as close to 0 as possible by updating the model weights as part of the training proceess.

#? Calculating the negative average log probabilty
neg_avg_log_probas = avg_log_probas * -1
print(neg_avg_log_probas)
#? OUTPUT : tensor(10.7395) --> Called as cross entropy loss

#^ Cross entropy loss is popular measure in machine learning and deep learning that measures the difference between two probabilty distributions - typically the true distribution labels (here, token in dataset) and the predicted distribution from a model (for instance, the token probabilities generated by an LLM).

#? Inspecting the shape of the logits
print("Logits shape:", logits.shape)
print("Target shape:", targets.shape)
#? OUTPUT : 
#? Logits shape: torch.Size([2, 3, 50257])
#? Target shape: torch.Size([2, 3])

#! NOTE : PyTorch has a built in cross_entropy() loss function that performs all the six steps we performed earlier but we must flatten our tensors over the batch dimension before applying the function.

#? Flattening the tensors
logits_flat = logits.flatten(0,1)       #? Flattens the logit tensor from [2,3,50275] to [6,50275]
targets_flat = targets.flatten()
print("Flattened logits:", logits_flat.shape)
print("Flattened targets:", targets_flat.shape)
#? OUTPUT : 
#? Flattened logits: torch.Size([6, 50257])
#? Flattened targets: torch.Size([6]

#? Applying PyTorch's cross_entropy() function
loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
print(loss)
#? OUTPUT : tensor(10.7395)

#^ Perplexity is a measure often used alongside cross entropy to evaluate the performance of models in tasks like language modeling. It is the measure of how well the probability distribution predicted by the model matches the actual distribution of the words in the dataset. Similar to entropy loss, lower perplexity indicates that the model prdictions are closer to the actual distribution.

#? Calcuating Perplexity of the previously calculated loss
perplexity = torch.exp(loss)
print(perplexity)
#? OUTPUT : tensor(46142.5352)

#! CONCLUSION : Calculated the loss for two small text inputs for illustration purposes.