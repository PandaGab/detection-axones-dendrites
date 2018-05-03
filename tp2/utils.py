#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import os
from shutil import copyfile
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import itertools

def make_dir(file_path):
    if not os.path.exists(file_path):
        os.makedirs(file_path)

def separate_train_test(dataset_path, train_path, test_path):

    class_index = 1
    for classname in sorted(os.listdir(dataset_path)):
        if classname.startswith('.'):
            continue
        make_dir(os.path.join(train_path, classname))
        make_dir(os.path.join(test_path, classname))
        i = 0
        for file in sorted(os.listdir(os.path.join(dataset_path, classname))):
            if file.startswith('.'):
                continue
            file_path = os.path.join(dataset_path, classname, file)
            if i < 15:
                copyfile(file_path, os.path.join(test_path, classname, file))
            else:
                copyfile(file_path, os.path.join(train_path, classname, file))
            i += 1

        class_index += 1

def create_loaders(train_path,test_path,batch_size,train_split,augmentation):
    # Create loaders
    normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    
    valid_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize])
    
    test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            normalize])
    
    if augmentation:
        train_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize])
    else:
        train_transform = transforms.Compose([
                transforms.Resize((224,224)),
                transforms.ToTensor(),
                normalize])   
    
    train_dataset = ImageFolder(train_path, train_transform)
    valid_dataset = ImageFolder(train_path, valid_transform)
    test_dataset = ImageFolder(test_path, test_transform)
    
    num_data = len(train_dataset)
    indices = list(range(num_data))
    np.random.shuffle(indices)
    split = int(np.floor(num_data * train_split))
    
    train_idx, valid_idx = indices[:split], indices[split:]
    
    train_sampler = torch.utils.data.sampler.SubsetRandomSampler(train_idx)
    valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(valid_idx)
    
    train_loader = torch.utils.data.DataLoader(train_dataset,
                               batch_size=batch_size, sampler=train_sampler)
    valid_loader = torch.utils.data.DataLoader(valid_dataset,
                               batch_size=batch_size, sampler=valid_sampler)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, valid_loader, test_loader

def freeze_and_change_fc(model):
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, 5)
    for name, param in model.named_parameters():
        if not name[0:2]=='fc':
            param.requires_grad = False
    
def validate(model, val_loader, use_gpu=True):
    model.train(False)
    true = []
    pred = []
    val_loss = []

    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    for j, batch in enumerate(val_loader):

        inputs, targets = batch
        if use_gpu:
            inputs = inputs.cuda()
            targets = targets.cuda()

        inputs = Variable(inputs, volatile=True)
        targets = Variable(targets, volatile=True)
        output = model(inputs)

        predictions = output.max(dim=1)[1]

        val_loss.append(criterion(output, targets).data[0])
        true.extend(targets.data.cpu().numpy().tolist())
        pred.extend(predictions.data.cpu().numpy().tolist())
    model.train(True)
    return accuracy_score(true, pred) * 100, sum(val_loss) / len(val_loss)

    
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    classes = ['Cars','Toy Story','Toy Story 2','Toy Story 3','Wall-e']
    title = 'Matrice de confusion'
    normalize = True
    cmap = plt.cm.Blues
    cm = cm.astype('float') / cm.sum(axis=1)[:,np.newaxis]
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()