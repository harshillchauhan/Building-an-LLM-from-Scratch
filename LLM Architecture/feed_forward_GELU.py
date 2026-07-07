import torch 
import torch.nn as nn
import matplotlib.pyplot as plt

#! GOAL : To implement a feed forward network with GELU activations

#? GPT-2 Model configuration 
GPT_CONFIG_124M = {
    "vocab_size" : 50257,           #? Vocabulary Size
    "context_length" : 1024,        #? Context Length
    "emb_dim" : 768,                #? Embedding Dimension
    "n_heads" : 12,                 #? Number of attention heads
    "n_layers" : 12,                #? Number of layers
    "drop_rate" : 0.1,              #? Droput rate
    "qkv_bias" : False              #? Query-Key-Value bias
}

#? GELU stands for Gaussian error linear unit and is considered to be more effective than the traditional activation dunction ReLU (Rectified Linear Unit)

#? Implementation of GELU Activation function
class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))))

#? Comapring GELU and ReLU activation function
gelu , relu = GELU(), nn.ReLU()

x = torch.linspace(-3, 3, 100)
y_gelu, y_relu = gelu(x), relu(x)
plt.figure(figsize = (8,3))
for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1, 2, i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)
plt.tight_layout()
plt.show()

#? As we can see in the resulting plot:-
#? ReLU (right) --> Is a piecewise linear function that outputs the input directly if it is positive, otherwise it outputs zero.
#? GELU (left) --> Is a smooth, non linear function that approximates ReLU but with a non-zero gradient for almost all negative values.

#! NOTE : The property of GELU that allows for a small, non-zero outout for negative values means that during the training process, neurons that receive negative input can still contribute to the learning process, although to a lesser extent than possitive values.

#? Using the GELU function to implement a feed forward neural network module 
class FeedForward(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        #? First nn.Linear layer --> Expands the embedding dimension by a factor of 4 (768 -> 3072)
        #? GELU() --> Adds non-linearity 
        #? Second nn.Linear layer --> Projects back down to the original embedding dimesnion (3072 -> 768)
        self.layers = nn.Sequential(nn.Linear(cfg["emb_dim"],4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
    
    def forward(self,x):
        return self.layers(x)
    
#? Initializing FeedForward module 
ffn = FeedForward(GPT_CONFIG_124M)  #? cfg is a Placeholder for GPT_CONFIG_124M
x = torch.rand(2, 3, 768)           #? Token embedding size 768, batch inputs 2, 3 tokens each 
out = ffn(x)            
print(out.shape)
#? OUTPUT : torch.Size([2, 3, 768])

#^ FeedForward module plays a crucial role in enhancing model's ability to learn from and generalize data.
#^ Although the input and output dimension of this module are the same, it internally expands the embedding dimension into a HIGHER DIMENSIOANL SPACE through the first linear layer.
#^ This exapansion is followed by a non-linear GELU activation and then a contraction back to the original dimension witg the second linear transformation.

#! NOTE : Uniformity in input and output dimesnions seimplifies the architecture by enabling the stacking of multiple layers, without the need to adjust dimensions between them, thus making the model more scalable.