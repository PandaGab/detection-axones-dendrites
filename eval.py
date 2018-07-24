import torch
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import transforms
from tqdm import tqdm
import skimage.io as io
import matplotlib.pyplot as plt

from utils import predict, plot_confusion_matrix

import tifTransforms as tifT
from dataset import datasetDetection
from unet import UNet
from sklearn.metrics import confusion_matrix

"""This function evaluate images in a folder and create the mask in another folder"""

use_gpu = True

# The model we want to evaluate
axonsModelPath = "/gel/usr/galec39/Documents/dev/detection-axones-dendrites/runs/best4.pth" # best4.pth
dendritesModelPath = "/gel/usr/galec39/Documents/dev/detection-axones-dendrites/runs/dendrite.pth"

# The folder containing the images to evaluate
root = "/gel/usr/galec39/data/Projet détection axones dendrites/test"
predDir = os.path.join(root, "predictions3")
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
[n.eval() for n in unet]
#nb = len(ax_den_Loader)
#cAx = np.zeros((2,2))
#cDen = np.zeros((2,2))
for i, (actine, masks) in tqdm(enumerate(ax_den_Loader)):
    X = np.moveaxis(actine.squeeze(0).cpu().numpy(), 0, 2)
    y = np.moveaxis(masks.squeeze(0).cpu().numpy(), 0, 2)
    
    axMask = y[:, :, 0]
    denMask = y[:, :, 1]
    
    h, w = y.shape[:2]
    im = np.zeros((h, w, 3), np.uint8)
    
    pred = [predict(X, n, crop_size) for n in unet]
    prob = [p > thresholds[i] for i, p in enumerate(pred)]
    
    im[:,:,0] = prob[0][:,:,0] * 255
    im[:,:,1] = prob[1][:,:,0] * 255
    
    io.imsave(os.path.join(predDir, str(imageID[i])+".jpg"), im)
#    y_predAx = prob[0][:,:,0].astype(np.bool).flatten()
#    y_predDen = prob[1][:,:,0].astype(np.bool).flatten()
#    y_trueAx = y[:,:,0].flatten()
#    y_trueDen = y[:,:,1].flatten()
#    
#    cAxtemp = confusion_matrix(y_trueAx, y_predAx)
#    cDentemp = confusion_matrix(y_trueDen, y_predDen)
#    
#    cAx += cAxtemp
#    cDen += cDentemp

#cAx /= nb
#cDen /= nb
#
#
#plt.figure()
#plt.subplot(121)
#plot_confusion_matrix(cAx, classes=['Background','Axones'], normalize=True)
#plt.subplot(122)
#plot_confusion_matrix(cDen, classes=['Background','Dendrites'], normalize=True)  
    
    
    






