from dataset import datasetDetection
from unet import UNet
import tifTransforms as tifT
from utils import initialize_weights, splitTrainTest, save_checkpoint, train_val_dataset
from utils import predict, prediction_accuracy
from mask import create_mask

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch
import torch.nn.functional as F

import os
import numpy as np

preprocess = True
use_gpu = True
####### DataPreprocessing #########
root = "/home/nani/Documents/data/Projet détection axones dendrites"
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
crop_size = 256
transformations = transforms.Compose([tifT.RandomCrop(crop_size),
                                      tifT.RandomHorizontalFlip(),                                         
#                                      tifT.Pad(100,mode='constant', constant_values=mean[0]),
                                      tifT.RandomVerticalFlip(),
                                      tifT.ToTensor(),
                                      tifT.Normalize(mean=mean,
                                                     std=std)])
train_set = datasetDetection(trainCsvFilePath, 
                           transforms=transformations,
                           prediction='axons')
val_size=0.1
val_transforms = transforms.Compose([tifT.ToTensor(),
                                     tifT.Normalize(mean=mean,
                                                    std=std)])
val_set = train_val_dataset(train_set, val_size, val_transforms)

train_dataloader = DataLoader(train_set, batch_size=32,shuffle=True)
val_dataloader = DataLoader(val_set, batch_size=1,shuffle=False)

####### Initialize de unet #######
unet = UNet(1, 1)
if use_gpu:
    unet.cuda()
unet.apply(initialize_weights)

####### Initialize training parameters ######
n_epoch = 300
patience = 5
lr = 0.001
momentum = 0.99

####### Initialize optimizer and loss ######
optimizer = optim.SGD(unet.parameters(), lr=lr, momentum=momentum)

criterion = nn.BCELoss()

####### Training #######
history_loss = []
history_acc = [0]
nb_val = len(val_dataloader)
for epoch in range(n_epoch):
    unet.train()
    epoch_loss = 0
#    if epoch % 10 == 0:
#        lr = lr / 10
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
        
    # Validation
    history_loss.append(epoch_loss)
    
    unet.eval()
    accTot = 0
    for actine, mask in val_dataloader:
        X = np.moveaxis(actine.squeeze(0).cpu().numpy(), 0, 2)
        y = np.moveaxis(mask.squeeze(0).cpu().numpy(), 0, 2)
        
        pred = predict(X, unet, crop_size)
        acc = prediction_accuracy(pred, y, threshold=0.5)
        accTot += acc
    val_acc = accTot / nb_val
        
    print('epoch :',epoch,' ---- loss :',epoch_loss, '---- val acc :', val_acc)
    state = {
            'epoch' : epoch,
            'loss' : epoch_loss,
            'loss_history' : history_loss,
            'state_dict' : unet.state_dict(),
            'acc' : val_acc,
            'acc_history' : history_acc,
            'optimizer' : optimizer.state_dict()
            }
    is_best = False
    if val_acc > history_acc[-1]:
        is_best = True
    history_acc.append(val_acc)

    
    modelname = '/gel/usr/galec39/data/Projet détection axones dendrites/models/model_{}.pth'.format(epoch)
    save_checkpoint(state, is_best, filename=modelname)
    




