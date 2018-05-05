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
import matplotlib.pyplot as plt
import pandas
import random
import numpy as np

# statistiques sur les photos dans train/actines
# mean = 32772.82847326139
# std = 8.01126226921115

class datasetDetection(Dataset):
    def __init__(self, csvFilePath, actineTransform=None, maskTransform=None):
        super(datasetDetection, self).__init__()
        
        rootDir = os.path.dirname(csvFilePath)
        csvFile = pandas.read_csv(csvFilePath, header=None)
        
        baseActinePath = os.path.join(rootDir, "actines")
        baseAxonesPath = os.path.join(rootDir, "axonsMask")
        baseDendritesPath = os.path.join(rootDir, "dendritesMask")
        
        self.actines = []
        self.axonesMask = []
        self.dendritesMask = []
        # the name of the images are 0.tif, 1.tif, 2.tif... n.tif in each folders
        for n in range(len(csvFile)):
            actinePath = os.path.join(baseActinePath, str(n) + ".tif")
            axonePath = os.path.join(baseAxonesPath, str(n) + ".tif")
            dendritePath = os.path.join(baseDendritesPath, str(n) + ".tif")
            
            self.actines.append(tifffile.imread(actinePath))
            self.axonesMask.append(tifffile.imread(axonePath))
            self.dendritesMask.append(tifffile.imread(dendritePath))
            
        if actineTransform is None:
            self.actineTransformations = tifT.ToTensor()
        else:
            self.actineTransformations = actineTransform
        
        if maskTransform is None:
            self.maskTransformations = transforms.ToTensor()
        else:
            self.maskTransformations = maskTransform

    def __getitem__(self, idx):
        seed = np.random.randint(2147483647)
        torch.manual_seed(seed)
        actine = self.actineTransformations(self.actines[idx])
        torch.manual_seed(seed)
        axone = self.maskTransformations(self.axonesMask[idx])
        torch.manual_seed(seed)
        dendrite = self.maskTransformations(self.dendritesMask[idx])
        background = torch.ones_like(actine) - axone - dendrite
        background[background < 0] = 0
        masks = torch.cat([axone, dendrite, background])
        return actine, masks
    
    def __len__(self):
        return len(self.actines)
    

    
if __name__ == "__main__":
    csvFilePath = "/home/nani/Documents/data/2017-11-14 EXP 201b Drugs/transcriptionTable.txt"
    mean = [32772.82847326139]
    std = [8.01126226921115]
    actineTransformation = transforms.Compose([tifT.RandomCrop(500),
                                         tifT.ToTensor(),
                                         transforms.Normalize(mean=mean,
                                                              std=std)])
    maskTransform = transforms.Compose([tifT.RandomCrop(500),
                                        tifT.ToTensor(),
                                        tifT.ToBool()])
    dataset = datasetDetection(csvFilePath, 
                               actineTransform=actineTransformation,
                               maskTransform=maskTransform)
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False)
    for actine , mask in dataloader:
        img = np.stack((actine[0,0],mask[0,0],mask[0,1]))
        plot(img)
        img = np.stack((actine[1,0],mask[1,0],mask[1,1]))
        plot(img)
        break


    
