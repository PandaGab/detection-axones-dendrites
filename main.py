from dataset import datasetDetection
from unet import UNet
import tifTransforms as tifT
from utils import initialize_weights, splitTrainTest, save_checkpoint, train_val_dataset
from utils import predict, accuracy
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
lr = 0.1
momentum = 0.9
validation_iter = 5
weight_decay = 0.1

####### Initialize optimizer and loss ######
optimizer = optim.SGD(unet.parameters(), lr=lr, momentum=momentum)

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

criterion = nn.BCELoss()

####### Training #######
history_loss = []
history_label = [0]
history_background = [0]
nb_val = len(val_dataloader)
for epoch in range(n_epoch):
    scheduler.step()
    unet.train()
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
            
        optimizer.zero_grad()

        pred = unet(X)
        prob = F.sigmoid(pred)
        prob_flatten = prob.view(-1)
        
        y_flatten = y.view(-1)
        
        loss = criterion(prob_flatten, y_flatten.float())
        epoch_loss += loss.data[0]
        
        
        loss.backward()
        
        optimizer.step()
        
    # Validation
    if (epoch % validation_iter) == 0:
        history_loss.append(epoch_loss)
        
        unet.eval()
        accLabelTot = 0
        accBackTot = 0
        for actine, mask in val_dataloader:
            X = np.moveaxis(actine.squeeze(0).cpu().numpy(), 0, 2)
            y = np.moveaxis(mask.squeeze(0).cpu().numpy(), 0, 2)
            
            pred = predict(X, unet, crop_size)
            
            labelacc, backgroundacc = accuracy(pred, y, threshold=0.5)
            accLabelTot += labelacc
            accBackTot += backgroundacc
        label_acc = accLabelTot / nb_val
        background_acc = accBackTot / nb_val
            
        print('epoch : {} --- loss : {:.3f} --- label acc : {:.3f} --- background acc : {:.3f}'.format(epoch, 
                                                                                                       epoch_loss,
                                                                                                       label_acc,
                                                                                                       background_acc))
        
        state = {
                'epoch' : epoch,
                'loss' : epoch_loss,
                'loss_history' : history_loss,
                'state_dict' : unet.state_dict(),
                'label_acc' : label_acc,
                'label_acc_history' : history_label,
                'background_acc' : background_acc,
                'background_acc_history' : history_background,
                'optimizer' : optimizer.state_dict()
                }
    
        
        is_best = False
        if label_acc > history_label[-1]:
            is_best = True
            
        history_label.append(label_acc)
        history_background.append(background_acc)
        
        modelname = '/gel/usr/galec39/data/Projet détection axones dendrites/models/model_{}.pth'.format(epoch)
        save_checkpoint(state, is_best, filename=modelname)
    




