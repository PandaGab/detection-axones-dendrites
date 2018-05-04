from torch.utils.data import Dataset, DataLoader
import torch

from torchvision import transforms

from PIL import Image

import os
from fnmatch import fnmatch
from skimage.external import tifffile
import matplotlib.pyplot as plt
import pandas
import random
import numpy as np


class datasetDetection(Dataset):
    def __init__(self, cvsFilePath, transform=None):
        super(datasetDetection, self).__init__()
        
        csvFile = pandas.read_csv(csvFilePath, header=None)
        
        self.actines = []
        self.axones = []
        self.dendrites = []
        for line in range(len(csvFile)):
            actinePath = csvFile[1][line]
            axonePath = csvFile[2][line]
            dendritePath = csvFile[3][line]
            self.actines.append(Image.open(actinePath))
            self.axones.append(Image.open(axonePath))
            self.dendrites.append(Image.open(dendritePath))
            
        self.transformations = transforms
        if transform is None:
            self.transformations = transforms.ToTensor()
             
    def __getitem__(self, idx):
        seed = np.random.randint(2147483647)
        random.seed(seed)
        actine = self.transformations(self.actines[idx].convert('L'))
        random.seed(seed)
        axone = self.transformations(self.axones[idx])
        random.seed(seed)
        dendrite = self.transformations(self.dendrites[idx])
        background = torch.ones_like(actine) - axone - dendrite
        background[background < 0] = 0
        
        return actine, torch.cat([axone, dendrite, background])
    
    def __len__(self):
        return len(self.actine)
    

    
if __name__ == "__main__":
    csvFilePath = './transcriptionTable.txt'
    dataset = datasetDetection(csvFilePath)
    actine, mask = dataset[1]
    plt.figure()
    plt.imshow(actine)
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False)
  
