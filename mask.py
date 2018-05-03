import numpy as np
import os

from fnmatch import fnmatch
from skimage.external import tifffile
from skimage import filters

import tkinter
from tkinter import filedialog

def gaussian_blur(img, sigma=10, threshold=5e-5, logicalNot=False):
    '''This function performs a gaussian blur on an numpy array then retreives the
    binary mask of the image
    
    :param img : A numpy array
    :param sigma : The standard deviation of the gaussian blur
    :param threshold : The threshold to use for the binary image
    
    :returns : The binary image filtered with gaussian blur
    '''
    gaussianBlur = filters.gaussian(img, sigma=sigma)
    binaryFilter = (gaussianBlur > threshold).astype(int)
    im = gaussianBlur * binaryFilter
    im = im / np.amax(im)
    if logicalNot:
        im[im > 0] = 1
        return np.logical_not(im)
    else:
        im[im > 0] = 1
        return im


if __name__ == "__main__":
    window = tkinter.Tk()
    window.withdraw()
    window.update()
    root = filedialog.askdirectory()
#    root = filedialog.askdirectory()
#    copy_folder = input("\nEnter name of the output directory : ")
#    copy_folder = os.path.join(root, copy_folder)
    #root = "/Users/Flavie/Desktop/2017-10-18 EXP211 Chronic Stim" # where to look
    #os.makedirs(os.path.join(root,"masked"),exist_ok=True) # where to save
    keepFormat = "*.tif" # may have to change to *.tiff in windows
    flist, nlist = [],[]
    for path, subdirs, files in os.walk(root):
        for name in files:
            if fnmatch(name,keepFormat):
                flist.append(os.path.join(path,name))
                nlist.append(name.split('.')[0])

    for i,f in enumerate(flist):
        img = tifffile.imread(f) # reads the file
        for chan in range(img.shape[0]):
            img[chan] = img[chan] - np.amin(img[chan]) # normalize to normal count

        dendrites = gaussian_blur(img[2], sigma=8, threshold=1e-5) # dendrite mask
        axons = gaussian_blur(img[1], sigma=8, threshold=4e-5,logicalNot=True) # axon mask
        
        # For dendrites only
        im = [
            dendrites
        ]
        # For both channels
        # im = [
        #     dendrites,
        #     axons
        # ]
        
        tifffile.imsave(os.path.join(root,"{}_dendrites.tif".format(nlist[i])), np.asarray(im).astype(np.uint16))