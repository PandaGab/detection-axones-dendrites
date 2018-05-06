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

import pandas
import numpy as np

# statistiques sur les photos dans train/actines
# mean = 32772.82847326139
# std = 8.01126226921115

class datasetDetection(Dataset):
    def __init__(self, csvFilePath, transforms=None):
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
            
        if transforms is None:
            self.transformations = tifT.ToTensor()
        else:
            self.transformations = transforms

    def __getitem__(self, idx):
        data = {}
        data['actine'] = self.actines[idx]
        mask = np.stack((self.axonesMask[idx], self.dendritesMask[idx]), axis=2)
        data['mask'] = mask
        data_transform = self.transformations(data)
        actine = data_transform['actine']
        mask = data_transform['mask']
        background = torch.ones_like(mask[0]) - mask[0] - mask[1]
        background[background < 0] = 0
        masks = torch.cat([mask[0].unsqueeze(0), 
                           mask[1].unsqueeze(0), 
                           background.unsqueeze(0)])
        
        return actine, masks
    
    def __len__(self):
        return len(self.actines)
    

    
if __name__ == "__main__":
    csvFilePath = "/home/nani/Documents/data/2017-11-14 EXP 201b Drugs/transcriptionTable.txt"
    mean = [32772.82847326139]
    std = [8.01126226921115]
    transformations = transforms.Compose([tifT.RandomCrop(500),
                                          tifT.RandomHorizontalFlip(),                                         
                                          tifT.Pad(100,mode='constant', constant_values=mean[0]),
                                          tifT.RandomVerticalFlip(),
                                          tifT.ToTensor(),
                                          tifT.Normalize(mean=mean,
                                                         std=std)])
    
    dataset = datasetDetection(csvFilePath, 
                               transforms=transformations)
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False)
    for actine , mask in dataloader:
        img = (actine[0,0].numpy(),mask[0,0].numpy(),mask[0,1].numpy(), mask[0,2].numpy())
        plot(img)
        img = (actine[1,0].numpy(),mask[1,0].numpy(),mask[1,1].numpy(),mask[1,2].numpy())
        plot(img)
        break


    
