import torch
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import (
    Resize,
    Normalize,
    Compose,
    RandomHorizontalFlip,
    ToTensor
)
from torch.utils.data import DataLoader


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.train_cifar_transforms = Compose(
            [
                # Resize((config.data.image_size, config.data.image_size)),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=config.data.norm_mean, std=config.data.norm_std),
                # to [-1; 1]
            ]
        )
        self.valid_cifar_transforms = Compose(
            [
                # Resize((config.data.image_size, config.data.image_size)),
                ToTensor(),
                Normalize(mean=config.data.norm_mean, std=config.data.norm_std),
                # to [-1; 1]
            ]
        )
        self.train_loader = DataLoader(
            CIFAR10(root='../data', download=True, train=True, transform=self.train_cifar_transforms),
            batch_size=config.training.batch_size,
            shuffle=True,
            drop_last=True
        )
        self.valid_loader = DataLoader(
            CIFAR10(root='../data', download=True, train=False, transform=self.valid_cifar_transforms),
            batch_size= config.training.batch_size,
            shuffle=False,
            drop_last=False
        )

    def sample_train(self):
        while True:
            for batch in self.train_loader:
                yield batch

    def get_images(self, gen, batch_size: int = 100):
        tmp = []
        already = 0
        while already < batch_size:
            (X, y) = next(gen)
            tmp += [X]
            already += len(X)
        return torch.cat(tmp)[:batch_size]
