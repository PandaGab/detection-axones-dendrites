from dataset import datasetDetection
from unet import UNet
import tifTransforms as tifT
from utils import initialize_weights, splitTrainTest, save_checkpoint
from mask import create_mask

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

import os

preprocess = False
use_gpu = True
####### DataPreprocessing #########
root = "/gel/usr/galec39/data/Projet détection axones dendrites"
if preprocess:
    # Création des masques
    create_mask(root)
    
    # Création du dossier train et test
    test_size = 0.2
    splitTrainTest(root, test_size)
    

####### Initialize the dataloaders ########
trainCsvFilePath = os.path.join(root, "train", "transcriptionTable.txt")
mean = [32772.97826603251]
std = [8.193871422305172]
transformations = transforms.Compose([tifT.RandomCrop(256),
                                      tifT.RandomHorizontalFlip(),                                         
#                                      tifT.Pad(100,mode='constant', constant_values=mean[0]),
                                      tifT.RandomVerticalFlip(),
                                      tifT.ToTensor(),
                                      tifT.Normalize(mean=mean,
                                                     std=std)])
train_dataset = datasetDetection(trainCsvFilePath, 
                           transforms=transformations,
                           prediction='axons')
train_dataloader = DataLoader(train_dataset, batch_size=8,shuffle=False)

####### Initialize de unet #######
unet = UNet(1, 1)
if use_gpu:
    unet.cuda()
unet.apply(initialize_weights)

####### Initialize training parameters ######
n_epoch = 10
patience = 5
lr = 0.01
momentum = 0.9

####### Initialize optimizer and loss ######
optimizer = optim.SGD(unet.parameters(), lr=lr, momentum=momentum)

criterion = nn.BCELoss()

####### Training #######
unet.train()
for epoch in range(n_epoch):
    epoch_loss = 0
    for i, (actine, mask) in enumerate(train_dataloader):
        X = actine.type(torch.FloatTensor)
        y = mask.type(torch.ByteTensor)
        
        if use_gpu:
            X = Variable(X).cuda()
            y = Variable(y).cuda()
        else:
            X = Variable(X)
            y = Variable(y)
            
        pred = unet(X)
        prob = F.sigmoid(pred)
        prob_flatten = prob.view(-1)
        
        y_flatten = y.view(-1)
        
        loss = criterion(prob_flatten, y_flatten.float())
        epoch_loss += loss.data[0]
        
        optimizer.zero_grad()
        
        loss.backward()
        
        optimizer.step()
    print('epoch :',epoch,' ---- loss :',epoch_loss)
    state = {
            'epoch' : epoch,
            'loss' : epoch_loss,
            'state_dict' : unet.state_dict(),
            'best_acc' : 0,
            'optimizer' : optimizer.state_dict()
            }
    modelname = 'model_{}.pth'.format(epoch)
    save_checkpoint(state, True, filename=modelname)
    




