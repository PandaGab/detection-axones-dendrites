import torch
import numpy as np
import random

import torchvision.transforms.functional as F

"""
Ce module permet d'appliquer les mêmes transformations aléatoires sur une 
image en entrée (image d'actine) et sur les masques. Il est grandement inspiré 
du module transforms de Pytorch.
"""

class Crop(object):
    """Crop the image.
    
    Args:
        ouput_size (tuple or int): Desired output size. If int, square crop
            is made.
        corner (tuple): top-left corner of the desired crop (h ,w)
    """
    def __init__(self, output_size, corner):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size
        assert isinstance(corner, tuple)
        assert len(corner) == 2
        self.corner = corner
    
    def __call__(self, im):
#        actine = data['actine']
#        mask = data['mask']
        h, w = im.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.round(self.corner[0])
        left = np.round(self.corner[1])
        
        bot = np.clip(top + new_h, 0, h)
        right = np.clip(left + new_w, 0, w)
        
        return top, bot, left, right
        
class FiveCrop(object):
    """Take five crop (corners and center) from the actine and masks."""
    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2, "Please provide only two dimensions (h, w) for size."
            self.output_size = output_size
        
    def __call__(self, data):
        size = self.output_size
        actine = data['actine']
        mask = data['mask']
        h, w = actine.shape[:2]    
        crop_h, crop_w = size
        if crop_h > h and crop_w > w:
            raise ValueError("Crop size error")
        
        tl = Crop(size, (0, 0))(actine)
        tr = Crop(size, (0, w - crop_w))(actine)
        bl = Crop(size, (h - crop_h, 0))(actine)
        br = Crop(size, (h - crop_h, w - crop_w))(actine)
        
        i = int(round((h - crop_h) / 2.))
        j = int(round((w - crop_w) / 2.))

        cc = Crop(size, (i, j))(actine)
    
        outCrops = [{'actine' : actine[tl[0]: tl[1], tl[2]: tl[3]], 'mask' : mask[tl[0]: tl[1], tl[2]: tl[3]]},
                    {'actine' : actine[tr[0]: tr[1], tr[2]: tr[3]], 'mask' : mask[tr[0]: tr[1], tr[2]: tr[3]]},
                    {'actine' : actine[bl[0]: bl[1], bl[2]: bl[3]], 'mask' : mask[bl[0]: bl[1], bl[2]: bl[3]]},
                    {'actine' : actine[br[0]: br[1], br[2]: br[3]], 'mask' : mask[br[0]: br[1], br[2]: br[3]]},
                    {'actine' : actine[cc[0]: cc[1], cc[2]: cc[3]], 'mask' : mask[cc[0]: cc[1], cc[2]: cc[3]]}]
 
        return outCrops
        
class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        actine = data['actine']
        mask = data['mask']
        h, w = actine.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        
        emptyMask = np.sum(mask[top: top + new_h, left: left + new_w]) == 0
        if np.sum(mask) == 0: # because the whole image is empty
            emptyMask = False
        
        while emptyMask:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

            emptyMask = np.sum(mask[top: top + new_h, left: left + new_w]) == 0
        data['actine'] = actine[top: top + new_h,
                       left: left + new_w]
        data['mask'] = mask[top: top + new_h,
                    left: left + new_w]
        
        return data
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, data):
        actine = data['actine']
        mask = data['mask']
        
        isActine3Channels = len(actine.shape) == 3
        if not isActine3Channels:
            actine = actine[:, :, np.newaxis]
        
        isMask3Channels = len(mask.shape) == 3
        if not isMask3Channels:
            mask = mask[:, :, np.newaxis]
        
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data['actine'] = torch.from_numpy(np.moveaxis(actine, 2, 0).astype('float'))
        data['mask'] = torch.from_numpy(np.moveaxis(ToBool()(mask), 2, 0).astype('float'))
        return data

class ToBool(object):
    """Convert to 0-1 Tensor."""
    
    def __call__(self, image):
        image[image > 0] = 1
        
        return image
        
class Normalize(object):
    """Normalize only the actine image."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
    
    def __call__(self, data):
        actine = data['actine']
        data['actine'] = F.normalize(actine, self.mean, self.std)
        
        return data
    
class RandomHorizontalFlip(object):
    """Random horizontal flip on actine and masks."""
    def __call__(self, data):
        
        if random.random() < 0.5:
            data['actine'] = np.fliplr(data['actine']).copy()
            data['mask'] = np.fliplr(data['mask']).copy()
        return data
    
class RandomVerticalFlip(object):
    """Random vertical flip on actine and masks."""
    def __call__(self, data):
        if random.random() < 0.5:
            data['actine'] = np.flipud(data['actine']).copy()
            data['mask'] = np.flipud(data['mask']).copy()
        return data
        
class Pad(object):
    """Pad only the actine image. Can be use with the same arguments as 
    https://docs.scipy.org/doc/numpy-1.14.0/reference/generated/numpy.pad.html"""
    def __init__(self, pad_width, mode, **kwargs):
        self.pad_width = pad_width
        self.mode = mode
        self.kwargs = kwargs
        
    def __call__(self, data):
        data['actine'] = np.pad(data['actine'], self.pad_width, self.mode, **self.kwargs)
        return data


        

        