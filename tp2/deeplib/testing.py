from torchvision.transforms import ToTensor
from data import train_valid_loaders
from training import validate


def test(model, dataset, batch_size, use_gpu=False):
    dataset.transform = ToTensor()
    loader, _ = train_valid_loaders(dataset, batch_size)
    return validate(model, loader, use_gpu)
