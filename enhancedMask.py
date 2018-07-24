import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_softmax, create_pairwise_bilateral, unary_from_labels
from skimage import filters
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv

from scipy.spatial import Delaunay
from shapely.geometry.polygon import Polygon


import os
from fnmatch import fnmatch
from skimage.external import tifffile
from mask import get_axon_foreground

def crf_enhancement(mask, nb=10, test=1):
    H, W = mask.shape
    NLABELS = 2
    
    # Create probability of foreground by filtering with gaussian from the 
    if test == 1:
        prob = filters.gaussian(mask, sigma=1)
        probs = np.zeros((2, H, W), dtype=np.float)
        probs[0] = prob
        probs[1] = 1 - prob
    
    
    # 3ieme test
    if test == 3:
        prob = mask
        probs = np.zeros((2, H, W), dtype=np.float)
        probs[0] = prob
        probs[1] = 1 - prob
    
    
    # Create the unary potential
    if (test == 1) or (test == 3):
        U = unary_from_softmax(probs)
    
    if test == 2:
        labels = mask // 255
        U = unary_from_labels(labels, 2, 0.7, zero_unsure=False)
    
    # Adding pairwise relationship, the bilateral one
    NCHAN = 1
    
    img = np.zeros((H, W, NCHAN), np.uint8)
    img[H//3:2*H//3, W//4:3*W//4,:] = 1
    
    pairwise_energy = create_pairwise_bilateral(sdims=(10, 10), schan=(0.01), 
                                                img=img, chdim=2)
    
    # Create denseCRF and adding the potentials
    d = dcrf.DenseCRF2D(W, H, NLABELS)
    d.setUnaryEnergy(U)
    d.addPairwiseEnergy(pairwise_energy, compat=10)
    
    # inference
    Q, tmp1, tmp2 = d.startInference()
    for _ in range(nb):
        d.stepInference(Q, tmp1, tmp2)
    map_soln = np.argmax(Q, axis=0).reshape((H, W))
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.imshow(mask)
#    plt.subplot(1,2,2)
#    plt.imshow(1 - map_soln)
#    plt.title(str(nb) + 'iterations')
    return 1 - map_soln
    

def morph(mask, kernel_size=5, morphType="opening"):
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (kernel_size, kernel_size))
    if morphType == "opening":
        m = cv.MORPH_OPEN
    elif morphType == "closure":
        m = cv.MORPH_CLOSE
    elif morphType == "dilatation":
        m = cv.MORPH_DILATE
    elif morphType == "erodation":
        m = cv.MORPH_ERODE
        
    opening = cv.morphologyEx(mask, m, kernel)
#    plt.figure()
#    plt.subplot(1,2,1)
#    plt.imshow(mask)
#    plt.subplot(1,2,2)
#    plt.imshow(opening)
    
    return opening

def triangulation(mask):
    mask = (mask // 255).astype(np.bool) # image 0 - 1 
    H, W = mask.shape
    
    mesh = np.meshgrid(np.arange(W), np.arange(H))
    pts = np.stack([mesh[0][mask], mesh[1][mask]]).T.astype(np.float)
    tri = Delaunay(pts)
    
    plt.figure()
    plt.triplot(pts[:,1], pts[:,0], tri.simplices.copy())
    plt.plot(pts[:,1], pts[:,0], 'o')
    plt.axis('equal')
    

    coord_groups = [tri.points[x] for x in tri.simplices]
    polygons = [Polygon(x) for x in coord_groups]
    
    area = np.array([p.area for p in polygons], dtype=np.float)
    perimeter = np.array([p.length for p in polygons], dtype=np.float)
#    sortedArea = np.sort(area)
    
    # visual
#    plt.figure()
#    plt.scatter(np.arange(len(sortedArea)), sortedArea)
#    plt.xlabel('Triangle')
#    plt.ylabel('Area')
    
    meanArea = np.mean(area)
    meanPeri = np.mean(perimeter)
    stdArea = np.std(area)
    stdPeri = np.std(perimeter)
#    ax = plt.gca()
#    xmin, xmax = ax.get_xbound()
#    plt.plot([xmin, xmax], [mean, mean], c='r')
    areaCriterion = meanArea + 3 * stdArea
    periCriterion = meanPeri + 3 * stdPeri
#    plt.plot([xmin, xmax], [criterion, criterion], c='b')
    
    toKeep = np.logical_and(area < areaCriterion, perimeter < periCriterion)
    tri = Delaunay(pts)
    tri.simplices = tri.simplices[toKeep]
    plt.figure()
    plt.triplot(pts[:,1], pts[:,0], tri.simplices.copy())
    plt.plot(pts[:,1], pts[:,0], 'o')
    plt.axis('equal')

    
    
    

    
    
if __name__  == "__main__":
     root = "/home/nani/Documents/data/Projet détection axones dendrites" 
     keepFormat = "*.tif" # may have to change to *.tiff in windows
     flist, nlist = [],[]
     for path, subdirs, files in os.walk(root):
         for name in files:
             if fnmatch(name,keepFormat):
                 flist.append(os.path.join(path,name))
     flist.sort()
     f = flist[200]
     img = tifffile.imread(f)
     axons = get_axon_foreground(img).astype(np.uint8) * 255
     
     
     ax1 = plt.subplot(3,3,1)
     plt.imshow(axons)
     plt.title('Masque original avec otsu (sans filtre)')
     plt.axis('off')
     
     plt.subplot(3,3,2, sharex=ax1, sharey=ax1)
     closure = morph(axons, morphType="closure")
     plt.imshow(closure)
     plt.title('Fermeture')
     plt.axis('off')
          
     plt.subplot(3,3,3, sharex=ax1, sharey=ax1)
     close_open = morph(closure, kernel_size=7, morphType="opening")
     plt.imshow(close_open)
     plt.title('Fermeture - Ouverture')
     plt.axis('off')
     
     plt.subplot(3,3,4, sharex=ax1, sharey=ax1)
     nb = 1
     close_crf = crf_enhancement(closure, nb=nb, test=1).astype(np.uint8)
     plt.imshow(close_crf)
     plt.title('Fermeture - CRF('+str(nb)+' it)')
     plt.axis('off')
     
     plt.subplot(3,3,5, sharex=ax1, sharey=ax1)
     plt.imshow(img[1], cmap='hot')
     plt.title('Image originale')
     plt.axis('off')
     
     plt.subplot(3,3,6, sharex=ax1, sharey=ax1)
     dilatation = morph(axons, kernel_size=5, morphType='dilatation')
     plt.imshow(dilatation)
     plt.title('Dilatation')
     plt.axis('off')
     
     plt.subplot(3,3,7, sharex=ax1, sharey=ax1)
     nb = 25
     dilatation_crf = crf_enhancement(dilatation, nb=nb, test=1).astype(np.uint8)
     plt.imshow(dilatation_crf)
     plt.title('Dilatation - CRF('+str(nb)+' it)')
     plt.axis('off')
     
     plt.subplot(3,3,8, sharex=ax1, sharey=ax1)
     dilat_crf_erode = morph(dilatation_crf, kernel_size=5, morphType='erodation')
     plt.imshow(dilat_crf_erode)
     plt.title('Dilatation - CRF - Erodation')
     plt.axis('off')

     plt.subplot(3,3,9, sharex=ax1, sharey=ax1)
     nb = 1
     close_open_crf = crf_enhancement(close_open, nb=nb, test=1).astype(np.uint8)
     plt.imshow(close_open_crf)
     plt.title('Fermeture - Ouverture - CRF('+str(nb)+' it)')
     plt.axis('off')

    
     
     plt.subplots_adjust(wspace=0.001, hspace=0.1)
     
     
     


    
    