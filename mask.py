import numpy as np
import os

from fnmatch import fnmatch
from skimage.external import tifffile
from skimage import filters
import skimage.io as io

from gaussian_mask import get_axon_foreground, get_dendrite_foreground

from tkinter import filedialog

from tqdm import tqdm


#def gaussian_blur(img, sigma=10, threshold=5e-5, logicalNot=False):
#     '''This function performs a gaussian blur on an numpy array then retreives the
#     binary mask of the image
#    
#     :param img : A numpy array
#     :param sigma : The standard deviation of the gaussian blur
#     :param threshold : The threshold to use for the binary image
#    
#     :returns : The binary image filtered with gaussian blur
#     '''
#     gaussianBlur = filters.gaussian(img, sigma=sigma)
##     vmax = np.amax(gaussianBlur)
#     binaryFilter = (gaussianBlur > threshold).astype(int)
##     binaryFilter = (gaussianBlur > threshold * vmax).astype(int)
#     im = gaussianBlur * binaryFilter
#     if np.amax(im) != 0:   
#         im = im / np.amax(im)
#     if logicalNot:
#         im[im > 0] = 1
#         return np.logical_not(im)
#     else:
#         im[im > 0] = 1
#         return im


def create_mask(root):
     keepFormat = "*.tif" # may have to change to *.tiff in windows
     flist, nlist = [],[]
     for path, subdirs, files in os.walk(root):
         for name in files:
             if fnmatch(name,keepFormat):
                 flist.append(os.path.join(path,name))
                 nlist.append(name.split('.')[0])

     flist.sort()
     file = open(os.path.join(root, 'transcriptionTable.txt'), 'w')

     # Create folders
     fullPath = os.path.join(root, "full")
     if not os.path.exists(fullPath):
         os.mkdir(fullPath)
     
     actinePath = os.path.join(root, "actines")
     if not os.path.exists(actinePath):
         os.mkdir(actinePath)

     axonMaskPath = os.path.join(root, "axonsMask")
     if not os.path.exists(axonMaskPath):
         os.mkdir(axonMaskPath)

     dendriteMaskPath = os.path.join(root,"dendritesMask")
     if not os.path.exists(dendriteMaskPath):
         os.mkdir(dendriteMaskPath)
         
     n = 0
     for i,f in tqdm(enumerate(flist)):
         fullSavePath = os.path.join(fullPath, str(n) + ".tif")
         actineSavePath = os.path.join(actinePath, str(n) + ".tif")
         axoneMaskSavePath = os.path.join(axonMaskPath, str(n) + ".tif")
         dendriteMaskSavePath = os.path.join(dendriteMaskPath, str(n) + ".tif")

         file.write(str(n) + ',' + f +'\n')

         img = tifffile.imread(f) # reads the file
         
         tifffile.imsave(fullSavePath, img)
         tifffile.imsave(actineSavePath, img[0])

#         for chan in range(img.shape[0]):
#             img[chan] = img[chan] - np.amin(img[chan]) # normalize to normal count

#         axons = gaussian_blur(img[1], sigma=5, threshold=0.1).astype(np.uint8) * 255 # axon mask
         axons = get_axon_foreground(img).astype(np.uint8) * 255
#         dendrites = gaussian_blur(img[2], sigma=5, threshold=0.2).astype(np.uint8) * 255 # dendrite mask
         dendrites = get_dendrite_foreground(img).astype(np.uint8) * 255

         tifffile.imsave(axoneMaskSavePath, axons)
         tifffile.imsave(dendriteMaskSavePath, dendrites)
         n += 1

     file.close()    

if __name__ == "__main__":
 #    root = filedialog.askdirectory()
 #    copy_folder = input("\nEnter name of the output directory : ")
 #    copy_folder = os.path.join(root, copy_folder)
     root = "/home/nani/Documents/data/Projet détection axones dendrites" # Portable
#     root = "/gel/usr/galec39/data/Projet détection axones dendrites" # ordi avec tout les folders
     #os.makedirs(os.path.join(root,"masked"),exist_ok=True) # where to save
     
     create_mask(root)