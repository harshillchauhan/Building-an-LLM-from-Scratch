import torch
import torch.nn as nn

#! GOAL : Implementing a neural network to illustrate shorcut connections

#? Shortcut connections, also calles as skip or residual connections are used to mitigate the challenges of VANISHING GRADIENTS.
#? The vanishing gradient problem refers to the issue where gradients(which guide wieght updates during training) become progressively smaller as they propogate backward through the layers, making it difficult to effectively train earlier layers.
#? A shortcut connection creates an alternative, shorter path for the gradient to flow through the network by skipping one or more layers. This is achieved by adding the output of one layer to the output of second layer.

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))))
    
#? Implementing a neural network to illustrate shortcut connections
class ExampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELU()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELU()),
        ])

    def forward(self, x):
        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else :
                x = layer_output
        return x

#? The above class implements a deep neural network with 5 layers, each consisiting of a Linear layer and a GELU activation function.
#? In the forward pass, we iteratively pass the input through the layers and optionally add the shortcut connections if the self.use_shortcut attribute is set to true

#? Initializing a neural network without shortcut connections.
layer_sizes = [3,3,3,3,3,1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut= False)

#? Implementing a function that computes the gradients in the model's backward pass
def print_gradients(model, x):
    output = model(x)                  #? Forward pass
    target = torch.tensor([[0.]])

    loss = nn.MSELoss()
    loss = loss(output, target)       #? Calculates the loss based on how close the target and the output are

    loss.backward()                  #? Backward pass to calculate the gradients
    for name, param in model.named_parameters():        #? Iterating through the weight parameter
        if 'weight' in name: 
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

#? print_gradients() --> Specifies a loss function that computes how close the model output and a user-specific target are.
#? When calling loss.backward(), PyTorch computes the loss gradient for each layer in the model.
#? We print the mean absolute gradient of these 3 x 3 gradient values to obtain a single gradient value per layer.

print_gradients(model_without_shortcut, sample_input)
#? OUTPUT : 
#? layers.0.0.weight has gradien mean of 0.00020173587836325169
#? layers.1.0.weight has gradien mean of 0.0001201116101583466
#? layers.2.0.weight has gradien mean of 0.0007152041653171182
#? layers.3.0.weight has gradien mean of 0.001398873864673078
#? layers.4.0.weight has gradien mean of 0.005049646366387606

#! NOTE : The output of the print_gradients function shows, the gradients become smaller as we progress from the last layer (layer 4) to the first layer (layer 1), which is a phenomenon called as Vanishing Gradient.

#? Instantiating a model with skip connection 
torch.manual_seed(123)
model_with_shortcut = ExampleDeepNeuralNetwork(layer_sizes, use_shortcut= True)
print_gradients(model_with_shortcut, sample_input)
#? OUTPUT :
#? layers.0.0.weight has gradient mean of 0.22169791162014008
#? layers.1.0.weight has gradient mean of 0.20694105327129364
#? layers.2.0.weight has gradient mean of 0.32896989583969116
#? layers.3.0.weight has gradient mean of 0.2665732204914093
#? layers.4.0.weight has gradient mean of 1.3258541822433472

#* The last layer (layer 4) still has a larger gradient than the other layers. However, the gradient value syabelizes as we progress toward the first layer (layer 0) and doesn't shrink to a vanishingly small value.

#! CONCLUSION : Shortcut connections are important for overcoming the limitations posed by the vanishing gradient problem in neural networks.