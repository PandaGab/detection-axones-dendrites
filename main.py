from dataset import datasetDetection
from unet import UNet
import tifTransforms as tifT
from utils import initialize_weights

from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable


####### Initialize the dataloader ########
csvFilePath = "/home/nani/Documents/data/2017-11-14 EXP 201b Drugs/transcriptionTable.txt"
mean = [32772.82847326139]
std = [8.01126226921115]
transformations = transforms.Compose([tifT.RandomCrop(256),
                                      tifT.RandomHorizontalFlip(),                                         
#                                      tifT.Pad(100,mode='constant', constant_values=mean[0]),
                                      tifT.RandomVerticalFlip(),
                                      tifT.ToTensor(),
                                      tifT.Normalize(mean=mean,
                                                     std=std)])
dataset = datasetDetection(csvFilePath, 
                           transforms=transformations)
dataloader = DataLoader(dataset, batch_size=8,shuffle=False)

####### Initialize de unet #######
unet = UNet(1, 3)
unet.apply(initialize_weights)

####### Initialize the training parameters ######
n_epoch = 10
patience = 5
lr = 0.01

####### Training #######
unet.train()
for epoch in range(n_epoch):
    for i, (actine, mask) in enumerate(dataloader):
        print(i)
        pass




