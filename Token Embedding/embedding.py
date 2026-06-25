import torch


#! GOAL : Mapping token ID's into vector embeddings for model input 

#? As a preliminary step, we must initialize these embedding weights with random values. This serves as a starting point for the LLM learning process and can be optimized.
#? A continuous vector embedding is necessary since GPT-like LLM's are deep neural networks trained with the back-propogation algorithm.

#? Token Embeding Example
input_ids = torch.tensor([2,3,5,1])

#! NOTE : For the sake of simplicity, we are using a small vocabulary of 6(instead of the 50,257 words in BPR tokenizer vocabulary) and creating emebedding size of onlly 3(which in GPT-3, the embedding size is 12,288 dimensions)
vocab_size = 6
output_dim = 3

torch.manual_seed(123)
#? Instantiating an embedding layer in PyTorch
embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

#? The print statement prints the embedding layer's underlying weight matrix.
print(embedding_layer.weight)

#? OUTPUT :
#?  Parameter containing:
#?  tensor([[ 0.3374, -0.1778, -0.1690],
#?          [ 0.9178,  1.5810,  1.3010],
#?          [ 1.2753, -0.2010, -0.1606],
#?          [-0.4015,  0.9666, -1.1481],
#?          [-1.1589,  0.3255, -0.6315],
#?          [-2.8400, -0.7849, -1.4096]], requires_grad=True)

#^ The weight matrix has 6 rows and 3 columns. One row for each of the possible tokens in the vocabulary and one column for each of the three embedding dimensions.

#? torch.nn.Embedding(vocab_size , output_dimension) --> It is used to create an embedding lookup table given the vocab size and the output dimensions

#? Returns the embedding vector for token 3 (similar to 4 row in previous output as python uses zero based indexing)
print(embedding_layer(torch.tensor([3])))
#? OUTPUT : tensor([[-0.4015,  0.9666, -1.1481]], grad_fn=<EmbeddingBackward0>)

#? Returns the embedding vector for all the token ID's in the vocabulary
print(embedding_layer(input_ids))

#?OUTPUT :
#? tensor([[ 1.2753, -0.2010, -0.1606],
#?         [-0.4015,  0.9666, -1.1481],
#?         [-2.8400, -0.7849, -1.4096],
#?         [ 0.9178,  1.5810,  1.3010]], grad_fn=<EmbeddingBackward0>)