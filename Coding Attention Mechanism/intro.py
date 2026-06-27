import torch

#! GOAL : To illustrate few key concepts in self-attention before adding trainable weights

#? Self attention is a mechanism that lets every word in an sentence see all other words to determine how important they are when building its representation.

import torch
inputs = torch.tensor(
  [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33],    
   [0.77, 0.25, 0.10],    
   [0.05, 0.80, 0.55]]   
)

#? Attention score determines how important each word is for the input word.

#? We determine intermediate attention scores between the query token and each input token. We determine these scores by computing the dot product of the query with every other input.

#? Calculating the attention score for x^2 
query = inputs[1]
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query)
print(attn_scores_2)
#? OUTPUT : tensor([0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865])

#? After achieving the attention scores, we NORMALIZE each of the attention score.
#? Normalization is used to adjust values measured in a different scale to a common scale. 

#? Normalizing each of the attention scores we computed previously
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())
#? OUTPUT : Attention weights: tensor([0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656])
#?          Sum: tensor(1.0000)

#! NOTE : The main goal behind normalization is to obtain attention weights that sum up to 1.

#? In practice, it is common and advisable to use the softmax function for normalization. This approcah is better at managing extreme values and offers more favourable gradient properties during training.

#? Implementing softmax function for normalizing attention scores.

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())
#? OUTPUT : Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
#?          Sum: tensor(1.)


#! The naive softmax implementation (softmax_naive) may encounter numerical instability problems, such as underflow or overflow, when dealing with small or large inputs.
#* Therefore, in practice it's advisable to use the PyTorch implementation of softmax, which has been extensively optimized for performance.

attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())
#? OUTPUT : Attention weights: tensor([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
#?          Sum: tensor(1.)

#? Calculating the CONTEXT VECOTOR Z^2
query = inputs[1]
context_vec_2 = torch.zeros(query.shape)
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i
print(context_vec_2)
#? OUTPUT : tensor([0.4419, 0.6515, 0.5683]) --> CONTEXT VECTOR Z^2

#^ THE MAIN FLOW OF SELF-ATTENTION MECHANISM
#^ STEP 1: Genereate attention scores for each input token
#^ STEP 2: Calculate the attention weights 
#^ STEP 3: Calculate the context vectors