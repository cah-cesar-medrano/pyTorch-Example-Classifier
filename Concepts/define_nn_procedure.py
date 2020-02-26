import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim

"""
A typical training procedure for a neural network is as follows:

* Define the neural network that has some learnable parameters (or weights)
* Iterate over a dataset of inputs
* Process input through the network
* Compute the loss (how far is the output from being correct)
* Propagate gradients back into the network’s parameters
* Update the weights of the network, typically using a simple update rule:
    weight = weight - learning_rate * gradient
"""

# Defining the neural network


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # Kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # ^*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print("Neural Net: {}\n".format(net))

# You just have to define the forward function,
# and the backward function (where gradients are computed) is automatically defined for you using autograd.
# You can use any of the Tensor operations in the forward function.

# The learnable parameters of a model are returned by net.parameters()
params = list(net.parameters())
print("Peramters: {}\n".format(len(params)))
print("Size/ Weight of Perameter 1: {}\n".format(params[0].size()))  # conv1's .weight

# Let’s try a random 32x32 input. Note: expected input size of this net (LeNet) is 32x32.
# To use this net on the MNIST dataset, please resize the images from the dataset to 32x32.
input = torch.randn(1, 1, 32, 32)
out = net(input)
print("Neural Net output: \n{}\n".format(out))

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))

# Note
# torch.nn only supports mini-batches. The entire torch.nn package only supports inputs that are a mini-batch of samples, and not a single sample.
# For example, nn.Conv2d will take in a 4D Tensor of nSamples x nChannels x Height x Width.
# If you have a single sample, just use input.unsqueeze(0) to add a fake batch dimension.

# Loss Function

# A loss function takes the (output, target) pair of inputs,
# and computes a value that estimates how far away the output is from the target.

# There are several different loss functions under the nn package.
# A simple loss is: nn.MSELoss which computes the mean-squared error between the input and the target.

output = net(input)
target = torch.randn(10)
target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)
print("Loss Values: {} \n\n".format(loss))

# Backpropagate

# To backpropagate the error all we have to do is to loss.backward().
# You need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
# Now we shall call loss.backward(), and have a look at conv1’s bias gradients before and after the backward.

net.zero_grad()  # Zeroes the gradient buffers of all parameters

print("conf1.bias.grad before backward: \n\n {} \n\n".format(net.conv1.bias.grad))

loss.backward()

print("conf1.bias.grad after backward: \n\n {} \n\n".format(net.conv1.bias.grad))


# Updating the Weight

# The simplest update rule used in practice is the Stochastic Gradient Descent (SGD):
# weight = weight - learning_rate * gradient

# We can implement this using simple Python code:

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# However, as you use neural networks,
# you want to use various different update rules such as SGD,
# Nesterov-SGD, Adam, RMSProp, etc.
# To enable this, we built a small package:
# torch.optim that implements all these methods. Using it is very simple:

# Create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()  # Zero the gradient buffer
output = net(input)

loss = criterion(output, target)
loss.backward()
optimizer.step()  # Does the Update
