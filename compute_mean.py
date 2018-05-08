from glob import glob
import numpy as np
from skimage.external import tifffile
from tqdm import tqdm

images = glob("/home-local/galec39.nobkp/data/Projet détection axones dendrites/train/actines/*.tif", recursive=True)

mean = 0.
stddev = 0.
nb_images = 0
for imp in tqdm(images):
    im = (tifffile.imread(imp)[0].astype('float64'))
    mean += im.mean()
    stddev += im.std()
    nb_images += 1
#    print(mean/nb_images, stddev/nb_images)

print(mean/nb_images, stddev/nb_images)
