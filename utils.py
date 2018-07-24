import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from shutil import copyfile
import torch
from tifTransforms import Crop
import copy
from torch.autograd import Variable
import torch.nn.functional as F
import itertools

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

def train_val_dataset(train_set, val_size, val_transforms):
    """Seperate the dataset in train and val dataset. val_size represent
    the % of the validation set.
    """
    nb = len(train_set)
    train, test = train_test_split(np.arange(nb), test_size=val_size)
    
    val_set = copy.deepcopy(train_set)
    val_set.keep(test)
    val_set.setTransformations(val_transforms)

    train_set.keep(train)
    
    return val_set

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
        
def predict(im, net, crop_size):
    """
    Because the net was train with square crop, the inference will be done the
    same way. We are going to do overlapping prediction. We average the 
    prediction of the overlapped section.
    crop_size will always be an even number.
    im is a numpy array (H, W, C)
    """
    assert isinstance(crop_size, (int, tuple))
    if isinstance(crop_size, int):
        crop_size = (crop_size, crop_size)
    
    h, w = im.shape[:2]

    delta_h = np.round(crop_size[0] / 2)
    delta_w = np.round(crop_size[1] / 2)
    
    prediction = np.zeros_like(im)
    nbPredictions = np.zeros_like(im)
    
    h_step = np.arange(0, h - delta_h, delta_h, dtype=np.int)
    w_step = np.arange(0, w - delta_w, delta_w, dtype=np.int)
    
    for hs in h_step:
        for ws in w_step:
            crop = Crop(crop_size, (hs, ws))
            top, bot, left, right = crop(im)
            
            # we always want crop_size crops
            if (bot - top) < crop_size[0]:
                top = bot - crop_size[0]
            if (right - left) < crop_size[0]:
                left = right - crop_size[0]
            
            cropIm = torch.from_numpy(np.moveaxis(im[top: bot, left: right], 2, 0)).type(torch.FloatTensor).unsqueeze(0)
            cropIm = Variable(cropIm, requires_grad=False).cuda()
            out = F.sigmoid(net(cropIm))
            pred = np.moveaxis(out.data.squeeze(0).cpu().numpy(), 0, 2)
            
            nbPredictions[top: bot, left: right] = nbPredictions[top: bot, left: right] + 1
            prediction[top: bot, left: right] = prediction[top: bot, left: right] + pred                                              
                                              
    return prediction / nbPredictions
    
def prediction_accuracy(predim, GTim, threshold=0.5):
    """This function compute the accuracy of the prediction with the ground
    truth. It is calculted as follow
    acc =     U 
          ---------
          pred+GT-U
    where: 
        U is the union section between pred and GT
        pred is the predicted area
        GT is the ground truth area
    
    Threshold represent the minimum value to be consider 1
    """
    
    
    """ 
    EDIT: Finally, this is not a good way of measuring the accuracy. See next
    function : accuracy
    """
    predim = predim > threshold
    
    # we are going to work with the number of pixels at 1 to define the area
    U = np.sum(np.logical_and(predim, GTim))
    pred = np.sum(predim)
    GT = np.sum(GTim)
    
    return U / (pred + GT - U)
    
def accuracy(predim, GTim, threshold=0.5):
    """This function compute the ratio for each predicted label over the ground
    truth
    """
    
    predim = predim > threshold
    GTim = GTim > 0.5 # just to make boolean
    
    backgroundGT = GTim == 0
    backgroundPred = predim == 0

    labelGT = GTim == 1
    labelPred = predim == 1
    
    backgroundAccuracy = np.sum(backgroundGT * backgroundPred) / np.sum(backgroundGT)
    
    if np.sum(labelGT) == 0:
        labelAccuracy = 0.5
    else:
        labelAccuracy = np.sum(labelGT * labelPred) / np.sum(labelGT)
    
    return labelAccuracy, backgroundAccuracy 
    
# from http://scikit-learn.org/stable/auto_examples/model_selection/
# plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
#    plt.colorbar()
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
    plt.ylabel('Vérité terrain')
    plt.xlabel('Prédiction')


        