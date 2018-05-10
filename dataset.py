from torch.utils.data import Dataset, DataLoader
import tifTransforms as tifT
import torch
import os

from torchvision import transforms

from skimage.external import tifffile

from utils import plot
#import os
#from fnmatch import fnmatch
#from skimage.external import tifffile

import numpy as np

# statistiques sur les photos dans train/actines
# mean = 32772.82847326139
# std = 8.01126226921115

class datasetDetection(Dataset):
    """
    Dataset used to train the unet to predict one of the following:
       axons_dendrites
       axons
       dendrites
    """  
    def __init__(self, csvFilePath, transforms=None, prediction='axons_dendrites'):
        super(datasetDetection, self).__init__()
        if prediction == 'axons_dendrites':
            self.predDendrites = True
            self.predAxons = True
        elif prediction == 'axons':
            self.predDendrites = False
            self.predAxons = True
        elif prediction == 'dendrites':
            self.predDendrites = True
            self.predAxons = True
        else:
            raise NameError('You must specified an appropriate prediction name')
        
        rootDir = os.path.dirname(csvFilePath)
        with open(csvFilePath) as f:
            csvFile = f.readlines()
        
        baseActinePath = os.path.join(rootDir, "actines")
        self.actines = []
        
        # in order to keep track of where the 
        # image comes from, we add the id of the image (the integer in front
        # of the path in the csv, variable n below)
        self.id = []
        
        if self.predAxons:
            baseAxonsPath = os.path.join(rootDir, "axonsMask")
            self.axonsMask = []
            
        if self.predDendrites:
            baseDendritesPath = os.path.join(rootDir, "dendritesMask")
            self.dendritesMask = []
        
        
        # Extract the name of the files
        fileNumber = [int(f.split(',')[0]) for f in csvFile]
        for n in fileNumber:
            self.id.append(n)
            
            actinePath = os.path.join(baseActinePath, str(n) + ".tif")
            self.actines.append(tifffile.imread(actinePath))
            
            if self.predAxons:
                axonPath = os.path.join(baseAxonsPath, str(n) + ".tif")
                self.axonsMask.append(tifffile.imread(axonPath))
            if self.predDendrites:
                dendritePath = os.path.join(baseDendritesPath, str(n) + ".tif")
                self.dendritesMask.append(tifffile.imread(dendritePath))
                        
        if transforms is None:
            self.transformations = tifT.ToTensor()
        else:
            self.transformations = transforms

    def __getitem__(self, idx):
        data = {}
        data['actine'] = self.actines[idx]
        
        # if we only predict axons or dendrites, we dont need the background mask
        if self.predAxons:
            mask = self.axonsMask[idx]
        if self.predDendrites:
            mask = self.dendritesMask[idx]
        if self.predAxons and self.predDendrites:
            mask = np.stack((self.axonesMask[idx], self.dendritesMask[idx]), axis=2)
         
        data['mask'] = mask
        data_transform = self.transformations(data)
        actine = data_transform['actine']
        masks = data_transform['mask']
        
        if self.predAxons and self.predDendrites:
            background = torch.ones_like(mask[0]) - mask[0] - mask[1]
            background[background < 0] = 0
            masks = torch.cat([mask[0].unsqueeze(0), 
                               mask[1].unsqueeze(0), 
                               background.unsqueeze(0)])
        
        return actine, masks
    
    def __len__(self):
        return len(self.actines)
    
    def keep(self, ID):
        self.actines = [self.actines[i] for i in ID]
        if self.predAxons:
            self.axonsMask = [self.axonsMask[i] for i in ID]
        if self.predDendrites:
            self.dendritesMask = [self.dendritesMask[i] for i in ID] 
        self.id = [self.id[i] for i in ID]

    
    def setTransformations(self, transforms):
        self.transformations = transforms


    
if __name__ == "__main__":
    csvFilePath = "/home/nani/Documents/data/2017-11-14 EXP 201b Drugs/transcriptionTable.txt"
    mean = [32772.82847326139]
    std = [8.01126226921115]
    transformations = transforms.Compose([tifT.RandomCrop(256),
                                          tifT.RandomHorizontalFlip(),                                         
#                                          tifT.Pad(100,mode='constant', constant_values=mean[0]),
                                          tifT.RandomVerticalFlip(),
                                          tifT.ToTensor(),
                                          tifT.Normalize(mean=mean,
                                                         std=std)])
    
    dataset = datasetDetection(csvFilePath, 
                               transforms=transformations)
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False)
    for actine , mask in dataloader:
        actine = actine.type(torch.FloatTensor)
        img = (actine[0,0].numpy(),mask[0,0].numpy(),mask[0,1].numpy(), mask[0,2].numpy())
        plot(img)
        img = (actine[1,0].numpy(),mask[1,0].numpy(),mask[1,1].numpy(),mask[1,2].numpy())
        plot(img)
        break


    
