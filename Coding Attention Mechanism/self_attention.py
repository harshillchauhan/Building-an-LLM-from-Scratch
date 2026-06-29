import torch

#! GOAL : TO implement a self-attention mechanism with trainable weights for a single input token.

#? The self-attention mechanism used in the original transformer architecture, the GPT models, and most other popular LLM's is called SCALED DOT PRODUCTION ATTENTION.

#^ We will implement the self-attention mechanism step-by-step by introducing three trainable weight metrices :
#^ Wq, Wk, and Wv. These three metrices are used to project the embedded input tokens x^i into QUERY, KEY and Value vectors.

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33],    
   [0.77, 0.25, 0.10],    
   [0.05, 0.80, 0.55]]   
)

x_2 = inputs[1]             #? Specigying to use the second token's embedding to compute query, key and value.
d_in = inputs.shape[1]      #? Input embedding dimension.
d_out = 2                   #? Specifying to project the output into 2‑dimensional query/key/value vectors.

#? Initianlizig the three weights metrices Wq, Wk, Wv
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad = False)

#! NOTE : We set 'requires_grad=Fasle' to reduce clutter in outputs, but if we were to use the weights metrices for MODEL TRAINING, we would set 'requires_grad=True' to update these matecies during model training.

#? Computing the query, key and value vectors
query_2 = x_2 @ W_query     #? Praduces Query Vector for token 2    
key_2 = x_2 @ W_key         #? Praduces key Vector for token 2 
value_2 = x_2 @ W_value     #? Praduces value Vector for token 2 
print(query_2)
#? OUTPUT : tensor([0.4306, 1.4551])

#! WEIGHT PARAMETERS vs ATTENTION WEIGHTS 
#? Weights Parameters are the fixed, trainable, numbers inside the model (like W_query), that gets updated during the training.
#? Attention Weights are dynamic values computed on the fly for each input, showing how much one token should focus on another in that special context.

#^ Even though we are temporialy focused on computing the context vector X^2, we still require the key and value vectors for all input elements as they are involved in computing the attention weights with respect to query q^2

#? Obtaing all keys and values via matrix multiplication
keys = inputs @ W_key
values = inputs @ W_value 
print("keys.shape:", keys.shape)
print("values.shape:", values.shape)
#? OUTPUT : keys.shape: torch.Size([6, 2])
#?          values.shape: torch.Size([6, 2])

#? Computing the attention scores for the second input token
keys_2 = keys[1]
attn_scores_22 = query_2.dot(keys_2)
print(attn_scores_22)
#? OUTPUT : tensor(1.8524)  --> Attention score of token 2 with respect to itself 

#? Generaliing the above computation for all attention scores using matrix multiplication.
attn_scores_2 = query_2 @ keys.T
print(attn_scores_2)
#? OUTPUT : tensor([1.2705, 1.8524, 1.8111, 1.0795, 0.5577, 1.5440])

#? Computing the attetnion weights by scaling the attention scores and using the softmax function
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim = -1)
print(attn_weights_2)
#? OUTPUT : tensor([0.1500, 0.2264, 0.2199, 0.1311, 0.0906, 0.1820])

#! NOTE : The reason for scaling the attention scores is :
#* When the attention scores get very large (because embeddings have thousands of dimensions), the softmax turns too sharp - one word gets all the focus while others get ignored, and gradients shrink towards zero.
#* Dividing by the square root of the embedding size keeps the scores balanced, so softmax spreads attention more smoothly and training stays stable.
#! In short, Scaling prevents the model from “overreacting” to large numbers, keeps learning steady, and ensures attention works properly even with huge embeddings.

#? Computing the context vector for the second input token
context_vec_2 = attn_weights_2 @ values
print(context_vec_2)
#? OUTPUT : tensor([0.3061, 0.8210]) --> context vector Z^2.

