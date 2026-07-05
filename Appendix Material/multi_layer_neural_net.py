import torch
import torch.nn as nn

#! PyTorch as a library can be used for implementing deep neural networks.

#^ When implementing a neural network in PyTorch, we can subclass the torch.nn.Module class to define our own custom network architecture. 
#^ --> Withing this subclass, we define the network layers in the __init__ constructor and specify how the layers interact in the forward method.
#^ --> The forward method describes how the input data passes through the network and comes together as a computation graph.
#^ --> The backward method, which we typically do not need to implement ourselves, is used in training to compute gradients of the loss function given the model parameters.

#? Implementing a multi-layer perceptron with two hidden layers
class NeuralNetwork(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super().__init__()

        self.layers = nn.Sequential(
            #? 1st hidden layer
            nn.Linear(num_inputs, 30),
            nn.ReLU(),

            #? 2nd hidden layer
            nn.Linear(30,20),
            nn.ReLU(),

            #? Output Layer
            nn.Linear(20, num_outputs)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
    
#? Instantiating a new neural network object
model = NeuralNetwork(50,3)

#? Printing the summary of our model
print(model)
#? OUTPUT : 
#? NeuralNetwork(
#?   (layers): Sequential(
#?     (0): Linear(in_features=50, out_features=30, bias=True)
#?     (1): ReLU()
#?     (2): Linear(in_features=30, out_features=20, bias=True)
#?     (3): ReLU()
#?     (4): Linear(in_features=20, out_features=3, bias=True)
#?   )
#? )

#! NOTE : We use the sequential class when we implement the NeuralNetwork class.
#* It is not required but can make our life easier if we have a series of layers we want to execute in a specific order. 
#* This way, after instantiating self.layers = Sequential() in the __init__ constructor, we just have to call the self.layers instead of calling each layer individually in the forward method.

#? Checking the total number of trainable parameters of this model
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)
#? OUTPUT : Total number of trainable model parameters: 2213

#^ Each parameter for which requires_grad = True, counts as a trainable parameter and will be updated during training.
#^ In the case of our neural network model, these trainable parameters are contained in the nn.Linear layers.
#^ A linear layer multiplies the inputs with a weight matrix and adds a bias vector. This is sometimes reffered to as a feedforward or fully connected layer.

#? Acessing the weight parameter matirx of first linear layer (index 0)
print(model.layers[0].weight)
print(model.layers[0].weight.shape)
#? OUTPUT : 
#? Parameter containing:
#? tensor([[-0.0414, -0.0936, -0.0941,  ..., -0.0675,  0.0143,  0.1265],
#?         [-0.1171,  0.1401,  0.0597,  ..., -0.1401,  0.0338, -0.0057],
#?         [ 0.1204, -0.1082, -0.1170,  ..., -0.0807, -0.0432,  0.0308],
#?         ...,
#?         [-0.1086, -0.0222, -0.1319,  ..., -0.1077, -0.1272,  0.1230],
#?         [-0.1166, -0.0287,  0.0994,  ..., -0.0509,  0.0946, -0.0018],
#?         [ 0.0614, -0.0163, -0.0447,  ..., -0.0470,  0.1236, -0.0992]],
#?        requires_grad=True)
#? torch.Size([30, 50])

#! NOTE : The weight matrux here is 50 x 30 matrix, and requires_grad = True, which means its entries are trainable -- This is the default setting for weights and biases in nn.Linear.

#? Acessing the bias parameter matrix of first layer (index 0)
print(model.layers[0].bias)
print(model.layers[0].bias.shape)
#? OUTPUT : 
#? Parameter containing:
#? tensor([ 0.1200, -0.1051,  0.0224,  0.1282, -0.0200, -0.0505, -0.1409, -0.0441,
#?          0.0631, -0.0808, -0.0646,  0.0125, -0.1147,  0.0958,  0.0742, -0.0871,
#?         -0.0033, -0.0809, -0.0034, -0.1150, -0.1355,  0.0289, -0.1309, -0.0089,
#?         -0.0948, -0.0530,  0.1406, -0.1232,  0.0328, -0.0882],
#?        requires_grad=True)
#? torch.Size([30])

#^ Upon executing the previous two codes multiple times, the output recieved is different each time. This is because the model weights are initialized with small random numbers, which differ each time we instantiate the network. 
#* It is actually desired in deep learning in order to break symmetry during training. Otherwise the nodes would be performing the same operations and updates during backpropagation, which would not allow the network to learn complex mappings form inputs to outputs.

#? Applying the forward pass method
torch.manual_seed(123)
X = torch.rand((1,50))
out = model(X)
print(out)
#? OUTPUT : tensor([[0.0154, 0.0047, 0.0735]], grad_fn=<AddmmBackward0>)

#^ The forward pass refers to calculating the output tensors from the input tensors. This involves passing the input data through all the neural network layers, starting from the input layer, through hidden layers, and finally to the output layer.
#^ grad_fn = <AddmmBackward0> means that the tensor we are inspecting was created via a matrix multiplication and addition operation. PyTorch will use this informatuon when it computes gradients during back=propagation. 
#^  The <AddmmBackward0> part of grad_fn=<AddmmBackward0> specifies the operation performed. In this case, it is an Addmm operation. Addmm stands for matrix multiplication (mm) followed by an addition (Add).

#! NOTE : When we use a model foe inferenece (for instance, making predictions) rather than training, the best practice is to use the torch.no_grad() context manager. This tells PyTorch that it doesn't need to keep track of the gradients, which can result in significant savings in memory and computation.

with torch.no_grad():
    out = model(X)
print(out)
#? OUTPUT : tensor([[ 0.0098, -0.0913,  0.1854]])
