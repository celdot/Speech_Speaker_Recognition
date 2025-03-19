
# This file contains boiler-plate code for defining and training a network in PyTorch.
# Please see PyTorch documentation and tutorials for more information 
# e.g. https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

# define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # define the layers of your network here
        # ...
    def forward(self, x):
        # define the foward computation from input to output
        # ...
        return x

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)

# instantiate the network and print the structure
net = Net()
print(net)
print('number of prameters:'count_parameters(net))

# define your loss criterion (see https://pytorch.org/docs/stable/nn.html#loss-functions)
# criterion = ... 

# define the optimizer 
optimizer = torch.optim.Adam(net.parameters())

# prepare/load the data into tensors 
# train_x = ..., train_y = ..., val_x = ..., val_y = ..., test_x = ..., test_y = ...

batch_size = 256

# create the data loaders for training and validation sets
train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# setup logging so that you can follow training using TensorBoard (see https://pytorch.org/docs/stable/tensorboard.html)
writer = SummaryWriter()

# train the network
num_epochs = 100

for epoch in range(num_epochs):
    net.train()
    train_loss = 0.0
    for inputs, labels in train_loader:
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        # accumulate the training loss
        train_loss += loss.item()

    # calculate the validation loss
    net.eval()
    with torch.no_grad():
        val_loss = 0.0
        for inputs, labels in val_loader:
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    # print the epoch loss
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)

    print(f'Epoch {epoch}: train_loss={train_loss}, val_loss={val_loss}')
    writer.add_scalars('loss',{'train':train_loss,'val':val_loss},epoch)

# finally evaluate model on the test set here
# ...

# save the trained network
torch.save(net.state_dict(), 'trained-net.pt')
