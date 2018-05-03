#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from utils import create_loaders, separate_train_test, freeze_and_change_fc, validate

import os
import numpy as np

import torch
from torch.autograd import Variable
from torchvision import models



dataset_path = './pixar'
train_path = './q4Train'
test_path = './q4Test'

# inputs
augmentation = True
train_split = 0.8
batch_size = 8
n_epoch = 100
use_gpu = True


# Create train and test folders
if not os.path.exists(train_path):
    separate_train_test(dataset_path, train_path, test_path)
    
# Create loaders
train_loader, valid_loader, test_loader = create_loaders(
        train_path, test_path, batch_size, train_split, augmentation)

model = models.resnet50(pretrained=True)

#freeze_and_change_fc(model)
in_features = model.fc.in_features
model.fc = torch.nn.Linear(in_features, 5)

#train_param = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9,weight_decay=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer,8,0.1)

criterion = torch.nn.CrossEntropyLoss()

if use_gpu:
    model.cuda()

model.train()

nani = 0
patience = 0
model_name = 'modelResnet50.pth'

for i in range(n_epoch):
    print('computing epoch:', i)
    for batch in train_loader:
        b =+ 1
        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()
        
        inputs = Variable(inputs)
        targets = Variable(targets)
        
        optimizer.zero_grad()
        
        output = model(inputs)
        
        loss = criterion(output, targets)
        loss.backward()
        
        optimizer.step()
        
    score, loss = validate(model, valid_loader, use_gpu=use_gpu)
    scheduler.step()
    print('val acc:', score)
    if score > nani:
        nani = score
        patience = 0
        torch.save(model.state_dict(), model_name)
    else:
        patience += 1
    
    if patience > 5:
        break


score, loss = validate(model,test_loader,use_gpu=use_gpu)
print('test acc:',score)
