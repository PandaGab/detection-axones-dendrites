
import numpy as np
from skimage import filters


# actin - 0, axons - 1, dendrites - 2
ACTIN, AXON, DENDRITE = 0, 1, 2


def get_axon_foreground(img):
    """Gets the foreground of the axon channel using a gaussian blur of sigma = 1
    and the otsu threshold.

    :param img: A 3D numpy array

    :returns : A binary 2D numpy array of the foreground
    """
#    blurred = filters.gaussian(img[AXON], sigma=1)
    blurred = img[AXON].astype(np.float)
    blurred /= blurred.max()
    val = filters.threshold_otsu(blurred)
    return (blurred > val).astype(np.uint8)


def get_dendrite_foreground(img):
    """Gets the foreground of the dendrite channel using a gaussian blur of
    sigma = 20 and the otsu threshold.

    :param img: A 3D numpy array

    :returns : A binary 2D numpy array of the foreground
    """
    blurred = filters.gaussian(img[DENDRITE], sigma=20)
    blurred /= blurred.max()
    val = filters.threshold_otsu(blurred)
    return (blurred > val).astype(np.uint8)
