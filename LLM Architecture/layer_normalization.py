import torch
import torch.nn as nn

#! GOAL : To implement a layer normalization class in order to stabalize training in deep GPT models.

#? Training deep nuural networks with many layers can sometimes prove challenging due to problems like vanishing or exploding gradients.
#? Vanishing Gradients : During back propagation, gradients shrink toward zero as they pass through deep layers, causing earlier layers to stop learning. 
#? Exploding Gradients : During back propagation, gradient grow uncontrollably large across deep layers, making training unstable and weight updates chaotic.
#! These problems lead to unstable training dynamics and make it difficult for the network to effectively adjust its weights.
#* In order to improve the efficieny and stability of neural network training, we implement layer normalization.

#^ The main idea behind layer normalization is to adjust the activation (outputs) of a neural network layer to have a mean of 0 and a variance of 1.

#? Implementing a neural network layer with five inputs and six outputs 
torch.manual_seed(123)
batch_example = torch.randn(2,5)
layer = nn.Sequential(nn.Linear(5,6), nn.ReLU())
out = layer(batch_example)      #? Transforms Passes the (2,5) input tensor through the sequential layer.
print(out)
#? OUTPUT : 
#? tensor([[0.2260, 0.3470, 0.0000, 0.2216, 0.0000, 0.0000],
#?         [0.2133, 0.2394, 0.0000, 0.5198, 0.3297, 0.0000]],
#?        grad_fn=<ReluBackward0>)

#? The neural network layer we have coded consists of a Linear layer, followed by a non-linear activation function, ReLU.
#^ ReLU() : Stands for Rectified Linear Unit, which is a standard activation function in neural networks. It simply thresholds negative inputs to 0, ensuring that a layer outputs only positive values.

#? Examining the mean and variance of the outputs 
mean = out.mean(dim = -1, keepdim = True)
var = out.var(dim = -1, keepdim = True)
print("Mean:\n", mean)          
print("Variance:\n", var)
#? OUTPUT : 
#? Mean:
#?  tensor([[0.1324],                                   --> Mean of first input row
#?         [0.2170]], grad_fn=<MeanBackward1>)          --> Mean of second input row 
#? Variance:
#?  tensor([[0.0231],                                   --> Variance of first input row
#?         [0.0398]], grad_fn=<VarBackward0>)           --> Varaince of second input row

#! NOTE : Using keepdim= True ensures that the output tensor retains the same number of dimensions as the input tensor, even though the operation reduces the tensor aling the dimension specifies in via 'dim'.

#^ If we specified keepdim = False, the returned mean tensor would be a two-dimensional vector [0.1324, 0.2170] instead of a 2 x 1 dimensional matrix [[0.1324], [0.2170]].

#^ The dim parameter specifies the dimension along which the calculation of the statistic (here, mean or variance) should be performed. dim = 1 or -1 (Column →) and dim = 0 (Row ↓).

#? Applying Normalization to the layer outputs
out_norm = (out - mean) / torch.sqrt(var)
mean = out_norm.mean(dim = -1, keepdim = True)
var = out_norm.var(dim = -1, keepdim = True)
print("Normalized layer outputs:\n", out_norm)
print("Mean:\n", mean)
print("Variance:\n", var)
#? OUTPUT : 
#? Normalized layer outputs:
#?  tensor([[ 0.6159,  1.4126, -0.8719,  0.5872, -0.8719, -0.8719],
#?         [-0.0189,  0.1121, -1.0876,  1.5173,  0.5647, -1.0876]],
#?        grad_fn=<DivBackward0>)
#? Mean:
#?  tensor([[9.9341e-09],
#?         [0.0000e+00]], grad_fn=<MeanBackward1>)
#? Variance:
#?  tensor([[1.0000],
#?         [1.0000]], grad_fn=<VarBackward0>)

#^ The operation of applying layer normalization to the layer outputs consists of subtracting the mean and dividing the square root of the variance (also known as Standard Deviation)

#! NOTE : The value 9.9341e-09 in the output tensor in the scientific notation for 9.9341 x 10^-9, which is       -0.0000000099341 in decimal form. The value is very close to 0, but it is not exactly 0 due ti small numerical errors that can accumalate because of the finite precision with which computers represent numbers.

#* To improve readability, we can also turn off the scientific notation when printing tensor values by setting sci_mode to False.
torch.set_printoptions(sci_mode= False)
print("Mean:\n", mean)
print("Variance:\n", var)
#? OUTPUT : 
#? Mean:
#?  tensor([[0.0000],
#?         [0.0000]], grad_fn=<MeanBackward1>)
#? Variance:
#?  tensor([[1.0000],
#?         [1.0000]], grad_fn=<VarBackward0>)

#? Implementing a layer normalization class
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
    
#? Applying the LayerNorm module to the batched inputs
ln = LayerNorm(emb_dim = 5)
out_ln = ln(batch_example)
mean = out_ln.mean(dim = -1, keepdim = True)
var = out_ln.var(dim = -1, keepdim = True)
print("Mean:\n", mean)
print("Variance:\n", var)
#? OUTPUT : 
#? Mean:
#?  tensor([[-0.0000],
#?         [ 0.0000]], grad_fn=<MeanBackward1>)
#? Variance:
#?  tensor([[1.2499],
#?         [1.2500]], grad_fn=<VarBackward0>)

#* The output shows that the layer normalization code works as expected and normalizes the values of each of the two inputs such that they have a mean 0 and a variance of 1.