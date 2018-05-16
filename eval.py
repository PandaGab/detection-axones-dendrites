import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import os
from torchvision import transforms
from tqdm import tqdm
from utils import predict

import tifTransforms as tifT
from dataset import datasetDetection
from unet import UNet

"""This function evaluate images in a folder and create the mask in another folder"""

use_gpu = True

# The model we want to evaluate
axonsModelPath = "/gel/usr/galec39/Documents/dev/detection-axones-dendrites/runs/best3.pth"
dendritesModelPath = "/gel/usr/galec39/Documents/dev/detection-axones-dendrites/runs/best3.pth"

# The folder containing the images to evaluate
root = "/gel/usr/galec39/data/Projet détection axones dendrites/test"
predDir = os.path.join(root, "predictions")
if not os.path.exists(predDir):
    os.mkdir(predDir)

axonsState = torch.load(axonsModelPath)
dendritesState = torch.load(dendritesModelPath)

unet = [UNet(1, 1), UNet(1, 1)]

unet[0].load_state_dict(axonsState['state_dict'])
unet[1].load_state_dict(dendritesState['state_dict'])

if use_gpu:
    [n.cuda() for n in unet]

##### Create the test loaders, one for axons and one for dendrites
testCsvFilePath = os.path.join(root, "transcriptionTable.txt")
mean = [32772.97826603251]
std = [8.193871422305172]
crop_size = 256

test_transforms = transforms.Compose([tifT.ToTensor(),
                                      tifT.Normalize(mean=mean,
                                                    std=std)])

ax_den_dataset = datasetDetection(testCsvFilePath, 
                                  transforms=test_transforms,
                                  prediction='axons_dendrites')

ax_den_Loader = DataLoader(ax_den_dataset, batch_size=1, shuffle=False)
imageID = ax_den_dataset.id # we follow this order if we put shuffle=False
thresholds = np.array([0.5, 0.5])
for i, (actine, masks) in tqdm(enumerate(ax_den_Loader)):
    X = np.moveaxis(actine.squeeze(0).cpu().numpy(), 0, 2)
    y = np.moveaxis(masks.squeeze(0).cpu().numpy(), 0, 2)
    
    axMask = y[:, :, 0]
    denMask = y[:, :, 1]
    
    pred = [predict(X, n, crop_size) for n in unet]
    
    
    






