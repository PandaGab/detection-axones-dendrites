import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from deeplib.trainingq2 import train, validate
import math


class MyDropout(nn.Module):

    def __init__(self, drop_probability, n_layers, hidden_size=100):
        super().__init__()
        self.drop_probability = drop_probability


        self.first_layer = nn.Linear(32 * 32 * 3, hidden_size)
        self.first_layer.weight.data.normal_(0.0, math.sqrt(2 / 32 * 32 * 3))
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

    def dropout(self, x, drop_prob=0.1, training=False):
        if training:
            p = torch.ones(1, x.size(1)) * (1 - drop_prob)
            bern = torch.distributions.Bernoulli(p)
            dropout_mat = bern.sample().repeat(x.size(0), 1)
            y = x.data * dropout_mat
        else:
            y = x.data * (1 - drop_prob)
        x.data = y

        return x

    def forward(self, x):
        x = x.view(-1, 32*32*3)
        out = F.relu(self.first_layer(x))
        for i, l in enumerate(self.layers):
            out = l.forward(out)
            out = F.relu(out)
            
            if i < len(self.layers)-1:
                out = self.dropout(out, self.drop_probability, training=self.training)
        return self.output_layer.forward(out)

if __name__ =="__main__":
    net = MyDropout(0.5, 3)
    net.cpu()
    dataset_train = datasets.CIFAR10('../CIFAR10_data', train=True, download=True)
    #optimizer = optim.SGD(net.parameters(), lr=0.01, nesterov=True, momentum=0.9)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    history = train(net, optimizer, dataset_train, 10, batch_size=32)


    torch.save(net, './trainingModel.pt')
    #the_model = torch.load('./trainingModel.pt')
    dataset_test = datasets.CIFAR10('../CIFAR10_data', train=False, download=True, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=32)
    score, loss = validate(net, test_loader, use_gpu=False)
    print(score)
