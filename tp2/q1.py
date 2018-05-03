import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
from deeplib.training import train, validate
import math


class SplitDataset(torch.utils.data.Dataset):
    def __init__(self, full_ds, offset, length):
        super(SplitDataset, self).__init__()
        self.full_ds = full_ds
        self.offset = offset
        self.length = length

        assert len(full_ds) >= offset + length, Exception("Dataset length problem")

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.full_ds[i+self.offset]

def split(dataset, percentage):
    val_offset = int(len(dataset)*(1-percentage))
    return SplitDataset(dataset, 0, val_offset), SplitDataset(dataset, val_offset, len(dataset)-val_offset)

def plot(train_data):
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    for batch in train_loader:
        inputs, targets = batch
        break
    plt.figure()
    for i in range(8):
        plt.subplot(2,4,i+1)
        plt.imshow(inputs[i,0,:,:].numpy())
        plt.axis('off')
        plt.title(str(targets[i]))

def mnist_dataset(train_split_percent=0.8, padding=0, rotation=0):


    train_transform = transforms.Compose([
        transforms.Pad((padding, padding)),
        transforms.RandomRotation(rotation),
        transforms.RandomResizedCrop(28, scale=(0.08, 1.0), ratio=(1, 1)),
        transforms.ToTensor(),
    ])
    valid_transform = transforms.Compose([
            transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(
        root='../data', train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.MNIST(
        root='../data', train=True,
        download=True, transform=valid_transform,
    )

    valid_split_percent = 1 - train_split_percent
    # train_size = len(train_dataset)
    # ind_list = list(range(train_size))
    # num_split = int(np.floor(valid_split_percent * train_size))
    # train_ind = ind_list[num_split:]
    # valid_ind = ind_list[:num_split]
    # train_sampler = SubsetRandomSampler(train_ind)
    # valid_sampler = SubsetRandomSampler(valid_ind)
    #
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=1, sampler=train_sampler)
    #
    # valid_loader = torch.utils.data.DataLoader(
    #     valid_dataset, batch_size=1, sampler=valid_sampler,
    # )

    train_ds, _ = split(train_dataset, valid_split_percent)
    _, valid_ds = split(valid_dataset, valid_split_percent)

    test_dataset = datasets.MNIST(
        root='../data', train=False,
        download=True, transform=valid_transform
    )

    return train_ds, valid_ds, test_dataset


class MyDropout(nn.Module):

    def __init__(self, n_layers, hidden_size=100):
        super().__init__()

        self.first_layer = nn.Linear(28*28, hidden_size)
        self.first_layer.weight.data.normal_(0.0, math.sqrt(2 / 28*28))
        self.first_layer.bias.data.fill_(0.005)

        self.layers = []
        for i in range(n_layers - 1):
            layer = nn.Linear(hidden_size, hidden_size)
            layer.weight.data.normal_(0.0, math.sqrt(2 / hidden_size))
            layer.bias.data.fill_(0.005)
            self.layers.append(layer)
            self.add_module('layer-%d' % i, layer)

        self.output_layer = nn.Linear(hidden_size, 10)
        self.output_layer.weight.data.normal_(0.0, math.sqrt(2 / hidden_size))
        self.output_layer.bias.data.fill_(0.005)



    def forward(self, x):
        x = x.view(-1, 28*28)
        out = F.relu(self.first_layer.forward(x))
        for i, l in enumerate(self.layers):
            out = l.forward(out)
            out = F.relu(out)
        return self.output_layer.forward(out)


net = MyDropout(3)
net.cpu()
#sélectionner le padding et la rotation en degrés
train_data, valid_data, test_data = mnist_dataset(padding=2, rotation=45);

plot(train_data)

#optimizer = optim.SGD(net.parameters(), lr=0.01, nesterov=True, momentum=0.9)
optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
history = train(net, optimizer, train_data, valid_data, 5, batch_size=64)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=64)
score, loss = validate(net, test_loader, use_gpu=False)
print(score)
