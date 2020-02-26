from __future__ import print_function
import torch
import numpy as np

# 5x3 matrix, uninitialized
x = torch.empty(5, 3)
print(x)
# randomly initialized matrix
y = torch.rand(5, 3)
print(y)
# Construct a matrix filled with zeros and dtype long
z = torch.zeros(5, 3, dtype=torch.long)
print(z)
# Construct tensor from data
a = torch.tensor([5.5, 3])
print(a)

# create a tensor based on an existing tensor.
# These methods will reuse properties of the input tensor,
# e.g. dtype, unless new values are provided by user
b = x.new_ones(5, 3, dtype=torch.double)  # new_* methods take in sizes
print(b)

c = torch.rand_like(x, dtype=torch.float)  # override dtype!
print(c)  # result has the same size

# get the size
print(x.size())

# Operations

# Addition: syntax type 1
print(x + y)
# Addition: syntax type 2
print(torch.add(x, y))

# Addition: providing an output tensor as argument
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)
# Addition: in-place
y.add_(x)
print(y)
# You can use standard NumPy-like indexing with all bells and whistles!
print(x[:, 1])

# Resizing: If you want to resize/reshape tensor, you can use torch.view

x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())

# if you have a one element tensor, use .item() to get the value as a Python number
x = torch.randn(1)
print(x)
print(x.item())

# NumPy Bridge

# Converting a Torch Tensor to a NumPy Array
a = torch.ones(5)
print(a)

b = a.numpy()
print(b)

# See how the numpy array changes value
a.add_(1)
print(a)
print(b)

# Converting NumPy Array to TOrch Tensor

# Changing the np array changed the Torch Tensor automatically

a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)
