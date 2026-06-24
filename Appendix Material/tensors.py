import torch

import torch

#? Creating a scalar (0-dimensional) tensor
tensor0d = torch.tensor(1)

#? Creating a vector (1-dimensional) tensor
tensor1d = torch.tensor([1, 2, 3])

#? Creating a 2-dimensional tensor
tensor2d = torch.tensor([[1, 2],
                         [3, 4]])

#? Creating a 3-dimensional tensor
tensor3d = torch.tensor([[[1, 2], [3, 4]],
                         [[5, 6], [7, 8]]])

#? Printing all the tensors we have created
print(tensor0d)
print(tensor1d)
print(tensor2d)
print(tensor3d)

#^ Tensor DataTypes
#? We can access the datatype of a tensor via the '.dtype' attribute of a tensor

#? PyTorch adopts the default 64-bit integer data type from python
tensor1d = torch.tensor([1, 2, 3])
print(tensor1d.dtype)
#? OUTPUT : torch.int64

#? If we create tensors from Python floats, PyTorch creates tensors with a 32-bit precision by default.
floatvec = torch.tensor([1.0, 2.0, 3.0])
print(floatvec.dtype)
#? OUTPUT : torch.float32

#^ Common PyTorch Tensor Operations

#? 1) torch.tensor : Used to create new tensors
tensorex = torch.tensor([[1, 2, 3],
                         [4, 5, 6]])
print(tensorex)

#? 2) .shape : Allows us to access the shape of a tensor
print(tensorex.shape)       #? OUTPUT : torch.Size([2, 3]) --> Meaning the tensor has 2 rows and 3 columns

#? 3) .reshape : Used to reshape the tensor dimension wise
print(tensorex.reshape(3,2))
#? OUTPUT : tensor([[1, 2],
#?                  [3, 4],
#?                  [5, 6]])

#? 4) .view : More common command for reshaping a tensor 
print(tensorex.view(3,2))
#? OUTPUT : tensor([[1, 2],
#?                  [3, 4],
#?                  [5, 6]])

#? 5) .T : Used to Transpose a tensor, meaning it will flip it across its diagnol
print(tensorex.T)
#? OUTPUT : tensor([[1, 4],
#?                  [2, 5],
#?                  [3, 6]])

#? 6) .matmul : Used to multiply two matrices
print(tensorex.matmul(tensorex.T))
#? OUTPUT : tensor([[14, 32],
#?                  [32, 77]])

#? 7) @ : Used to multiply two matrices but is more compact than .matmul
print(tensorex @ tensorex.T)
#? OUTPUT : tensor([[14, 32],
#?                  [32, 77]])