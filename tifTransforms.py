import torch
import numpy as np

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

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        image = image[top: top + new_h,
                      left: left + new_w]
        
        return image
    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, image):
        threeChannels = len(image.shape) == 3
        
        if not threeChannels:
            image = image[:, :, np.newaxis]
        
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = np.moveaxis(image, 2, 0).astype('float')
        return torch.from_numpy(image)

class ToBool(object):
    """Convert to 0-1 Tensor."""
    
    def __call__(self, image):
        image[image > 0] = 1
        
        return image
        