from torch.utils.data import Dataset, DataLoader
import torch

from torchvision import transforms

from PIL import Image

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
            
        if transform is None:
            self.transformations = transforms.ToTensor()
        else:
            self.transformations = transform

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
        masks = torch.cat([axone, dendrite, background])
        return actine, masks
    
    def __len__(self):
        return len(self.actines)
    

    
if __name__ == "__main__":
    csvFilePath = './transcriptionTable.txt'
    transformation = transforms.Compose([transforms.RandomCrop(150),
                                         transforms.ToTensor()])
    dataset = datasetDetection(csvFilePath, transform=transformation)
    actine, mask = dataset[1]
    # plt.figure()
    # plt.imshow(actine)
    
    dataloader = DataLoader(dataset, batch_size=2,shuffle=False)
    plt.figure()
    for input , m in dataloader:
        print(input.size())
        plt.subplot(1,2,1)
        plt.imshow(input[0].squeeze(0).numpy())
        plt.subplot(1,2,2)
        plt.imshow(m[0,2].numpy())


    
