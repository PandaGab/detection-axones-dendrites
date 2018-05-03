import torch
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import math
from q2 import MyDropout


model = torch.load('./trainingModel.pt')
n = 25
dataset_test = datasets.CIFAR10('../CIFAR10_data', train=False, download=False, transform=transforms.ToTensor())
num_data = len(dataset_test)
indices = np.arange(num_data)
split = math.floor(0.25 * num_data)
train_idx = indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=train_sampler)

classPrediction = np.zeros(len(test_loader), dtype=np.bool)
distanceEucli = np.zeros(len(test_loader))
predCounter = 0
for j, image in enumerate(test_loader):
    model.train(False)
    input, target = image
    input = torch.autograd.Variable(input)
    out1 = torch.nn.LogSoftmax(model(input)).dim.data.numpy()
    outClass1 = np.argmax(out1)

    model.train(True)
    outAverage = np.zeros((1, 10))
    for i in range(n):
        temp = torch.nn.LogSoftmax(model(input)).dim.data.numpy()
        outAverage += temp
        if np.argmax(temp) != outClass1:
            predCounter += 1
    outAverage = outAverage/n
    outClass2 = np.argmax(outAverage)

    classPrediction[j] = (outClass1 == outClass2)
    distanceEucli[j] = np.linalg.norm((out1-outAverage))

pourcentageClassEqui = np.mean(classPrediction)
moyenneDistEucli = np.mean(distanceEucli)
pourcentagePredCount = predCounter/(len(test_loader)*n)

print("Pourcentage de classes équivalentes: ", pourcentageClassEqui)
print("Moyenne distance Euclidienne: ", moyenneDistEucli)
print("Pourcentage de passes dans réseau en désaccord avec moyenne: ", pourcentagePredCount)
