import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from shutil import copyfile
import torch
import pandas

def normalize(img):
    img = img.astype(np.float)
    vmax = np.max(img)
    vmin = np.min(img)
    return (img - vmin) / (vmax - vmin)

def plot(img, show=['actine','axon','dendrite', 'background']):
    nbPlot = len(show)
    plt.figure()
    for n in range(nbPlot):
        if 'actine' in show[n]:
            plt.subplot(1,nbPlot,n+1)
            plt.imshow(img[0])
            plt.title('Actine')
        if 'axon' in show[n]:
            plt.subplot(1,nbPlot,n+1)
            plt.imshow(img[1],cmap='gray')
            plt.title('Axon')
        if 'dendrite' in show[n]:
            plt.subplot(1,nbPlot,n+1)
            plt.imshow(img[2], cmap='gray')
            plt.title('Dendrite')
        if 'background' in show[n]:
            plt.subplot(1,nbPlot,n+1)
            plt.imshow(img[3],cmap='gray')
            plt.title('Background')

def splitTrainTest(root, test_size):
    # create train and test folders, each containing actine, axonsMask and
    # dendritesMask folder
    
    actinePath = os.path.join(root, "actines")
    axonsMaskPath = os.path.join(root, "axonsMask")
    dendritesMaskPath = os.path.join(root, "dendritesMask")
    
    # first gather the number of element
    a = len(os.listdir(actinePath))
    b = len(os.listdir(axonsMaskPath))
    c = len(os.listdir(dendritesMaskPath))
    
    assert a == b
    assert b == c

    train, test = train_test_split(np.arange(a), test_size=test_size)
    
    # Create one csv transcription file for each folder
    csvFilePath = os.path.join(root, "transcriptionTable.txt") # sorted
    with open(csvFilePath) as f:
        csvFile = f.readlines()
    train_csv = []
    test_csv = []
    
    # create the folders
    tt = ["train", "test"]
    la = ["actines", "axonsMask", "dendritesMask"]
    for a in tt:
        os.mkdir(os.path.join(root, a))
        for b in la:
            os.mkdir(os.path.join(root, a, b))
            
    # copy the files
    for trainID in train:
        train_csv.append(csvFile[trainID])
        for b in la:
            src = os.path.join(root, b,str(trainID)+".tif")
            dst = os.path.join(root,"train", b,str(trainID)+".tif")
            copyfile(src, dst)
        
    
    for testID in test:
        test_csv.append(csvFile[testID])
        for b in la:        
            src = os.path.join(root, b,str(testID)+".tif")
            dst = os.path.join(root,"test", b,str(testID)+".tif")
            copyfile(src, dst)
    
    train_temp = [int(t.split(',')[0]) for t in train_csv]
    train_sort_idx = np.argsort(train_temp)
    train_csv = np.array(train_csv)
    train_csv = train_csv[train_sort_idx]
    
    test_temp = [int(t.split(',')[0]) for t in test_csv]
    test_sort_idx = np.argsort(test_temp)
    test_csv = np.array(test_csv)
    test_csv = test_csv[test_sort_idx]
    
    train_csv_path = os.path.join(root, "train", "transcriptionTable.txt")
    with open(train_csv_path, 'w') as f:
        f.writelines(train_csv)
    
    test_csv_path = os.path.join(root, "test", "transcriptionTable.txt")
    with open(test_csv_path, 'w') as f:
        f.writelines(test_csv)
    
        
                
# just pass this function to de the net: net.apply(initialize_weights)
def initialize_weights(m):
    name = m.__class__.__name__
    if name.find('Conv') != -1:
        m.weight.data.normal_(0, 0.02)
#        print('initialize conv weights')
    elif name.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
#        print('initialize BN weights')
    
# À partir de https://discuss.pytorch.org/t/saving-and-loading-a-model-in-pytorch/2610/3
def save_checkpoint(state, is_best, filename='checkpoint.pth'):
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth')
    
    
        