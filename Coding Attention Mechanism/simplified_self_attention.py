import torch 

#! GOAL : To calculate the attention weights and context vectors for all inputs creating a simplified self-attention mechanism..
#! NOTE : The context vector calculated here are calculated from non-trainable weights. We will implement trainable weights in 'self-attention.py'.

inputs = torch.tensor(
  [[0.43, 0.15, 0.89], 
   [0.55, 0.87, 0.66],
   [0.57, 0.85, 0.64], 
   [0.22, 0.58, 0.33],    
   [0.77, 0.25, 0.10],    
   [0.05, 0.80, 0.55]]   
)

#? The first two steps remains the same as we have discussed in file 'intro.py'

#? STEP 1: Calculating the attention scores for all input token embeddings.
attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i , x_j)
print(attn_scores)

#? OUTPUT : 
#? tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#?         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#?         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#?         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#?         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#?         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])

#! In the above output, each element represents an attention score between each pair of the inputs.

#? In the above code, we used for loops for calculating the attention scores.
#* However, for loops are generally slow and we can achieve the same results using matrix multiplication. 

attn_scores = inputs @ inputs.T
print(attn_scores)

#? OUTPUT : 
#? tensor([[0.9995, 0.9544, 0.9422, 0.4753, 0.4576, 0.6310],
#?         [0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
#?         [0.9422, 1.4754, 1.4570, 0.8296, 0.7154, 1.0605],
#?         [0.4753, 0.8434, 0.8296, 0.4937, 0.3474, 0.6565],
#?         [0.4576, 0.7070, 0.7154, 0.3474, 0.6654, 0.2935],
#?         [0.6310, 1.0865, 1.0605, 0.6565, 0.2935, 0.9450]])

#? STEP 2: Calculating attention weights by normalizing each row so that the values in each row sum to 1

attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

#? OUTPUT : 
#? tensor([[0.2098, 0.2006, 0.1981, 0.1242, 0.1220, 0.1452],
#?         [0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
#?         [0.1390, 0.2369, 0.2326, 0.1242, 0.1108, 0.1565],
#?         [0.1435, 0.2074, 0.2046, 0.1462, 0.1263, 0.1720],
#?         [0.1526, 0.1958, 0.1975, 0.1367, 0.1879, 0.1295],
#?         [0.1385, 0.2184, 0.2128, 0.1420, 0.0988, 0.1896]])

#^ In context of using PyTorch, the 'dim' parameter in functions like torch.softmax specifies the dimension of the input tensor along which the function will be computed.
#^ dim = -1 --> Applies normalization across the column so that the values in each row sum up to 1
#^ dim = 0 --> Applies normalization across the row so that the values in each column sum up to 1
#^ dim = 1 --> Same as dim = -1

#? Verifying that rows indeed all sum to 1.
row_2_sum = sum([0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581])
print("Row 2 sum:", row_2_sum)      
print("All rows sums:", attn_weights.sum(dim=-1))   
#? OUTPUT : row 2 sum: 1.0
#?          All rows sums: tensor([1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000])

#? STEP 3: Computing the Context Vectors using the attention weights and applying matrix multiplication
all_context_vecs = attn_weights @ inputs
print(all_context_vecs)

#? OUTPUT : 
#? tensor([[0.4421, 0.5931, 0.5790],
#?         [0.4419, 0.6515, 0.5683],
#?         [0.4431, 0.6496, 0.5671],
#?         [0.4304, 0.6298, 0.5510],
#?         [0.4671, 0.5910, 0.5266],
#?         [0.4177, 0.6503, 0.5645]])

#! NOTE : The @ operator in 'all_context_vecs = attn_weights @ inputs' is shorthand for matrix multiplication in the PyTorch library.
