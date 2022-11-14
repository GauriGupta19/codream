import numpy as np
import torch
import torchvision.transforms as T
from torchvision.datasets.cifar import CIFAR10


class CIFAR10_DSET():
    def __init__(self, dpath) -> None:
        self.IMAGE_SIZE = 32
        self.NUM_CLS = 10
        self.mean = np.array((0.4914, 0.4822, 0.4465))
        self.std = np.array((0.2023, 0.1994, 0.2010))
        train_transform = T.Compose(
            [
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        test_transform = T.Compose(
            [
                T.ToTensor(),
                T.Normalize(
                    self.mean, 
                    self.std
                ),
            ]
        )
        self.train_dset = CIFAR10(
            root=dpath, train=True, download=True, transform=train_transform
        )
        self.test_dset = CIFAR10(
            root=dpath, train=False, download=True, transform=test_transform
        )
        self.IMAGE_BOUND_L = torch.tensor((-self.mean / self.std).reshape(1, -1, 1, 1)).float()
        self.IMAGE_BOUND_U = torch.tensor(((1 - self.mean) / self.std).reshape(1, -1, 1, 1)).float()


def get_dataset(dname, dpath):
    dset_mapping = {"cifar10": CIFAR10_DSET}
    return dset_mapping[dname](dpath)
