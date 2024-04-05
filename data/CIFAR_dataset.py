import os
import torch
from torchvision.datasets import MNIST, CIFAR10, VisionDataset, ImageFolder
from torchvision.transforms import (
    Resize,
    Normalize,
    Compose,
    RandomHorizontalFlip,
    ToTensor
)
from torch.utils.data import DataLoader
import torch.distributed as dist


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.root_path = config.data.dataset_path # os.path.join(config.data.dataset_path, "tmp")
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

        self.make_train_loader()
        self.make_valid_loader()

    def make_train_loader(self):
        self.train_dataset = CIFAR10(
            root=self.root_path,
            download=True,
            train=True,
            transform=self.train_cifar_transforms
        )

        if dist.is_initialized():
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()

            self.sampler_train = torch.utils.data.DistributedSampler(
                self.train_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=True,
            )
        else:
            self.sampler_train = None

        self.train_loader = DataLoader(
            self.train_dataset,
            sampler=self.sampler_train,
            batch_size=self.config.training.batch_size_per_gpu,
            drop_last=True,
            num_workers=10,
        )

    def make_valid_loader(self):
        self.valid_dataset = CIFAR10(
            root=self.root_path,
            download=True,
            train=False,
            transform=self.valid_cifar_transforms
        )
        if dist.is_initialized():
            num_tasks = dist.get_world_size()
            global_rank = dist.get_rank()

            self.sampler_valid = torch.utils.data.DistributedSampler(
                self.valid_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
            )
        else:
            self.sampler_valid = None
        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=self.sampler_valid,
            batch_size=self.config.training.batch_size_per_gpu,
            drop_last=False,
            num_workers=10,
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
