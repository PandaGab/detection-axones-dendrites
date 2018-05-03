import numpy as np
import matplotlib.pyplot as plt

def normalize(img):
    img = img.astype(np.float)
    vmax = np.max(img)
    vmin = np.min(img)
    return (img - vmin) / (vmax - vmin)

def plot(img, show=['actine','dendrite','axon']):
    nbPlot = len(show)
    plt.figure()
    for n in range(nbPlot):
        if 'actine' in show[n]:
            plt.subplot(1,nbPlot,n+1)
            plt.imshow(img[0])
            plt.title('Actine')
        if 'axon' in show[n]:
            plt.subplot(1,nbPlot,n+1)
            plt.imshow(img[1])
            plt.title('Axon')
        if 'dendrite' in show[n]:
            plt.subplot(1,nbPlot,n+1)
            plt.imshow(img[2])
            plt.title('Dendrite')

        

    