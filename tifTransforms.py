import torch
import numpy as np
import random

import torchvision.transforms.functional as F

"""
Ce module permet d'appliquer les mêmes transformations aléatoires sur une 
image en entrée (image d'actine) et sur les masques
"""

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
        
        