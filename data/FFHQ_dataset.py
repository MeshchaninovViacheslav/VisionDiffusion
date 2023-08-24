import os
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk

from torchvision.transforms import (
    Resize,
    Normalize,
    Compose,
    RandomHorizontalFlip,
    ToTensor
)
from torch.utils.data import DataLoader
import torch.distributed as dist


def load_FFHQ():
    from datasets import load_dataset
    dt = load_dataset("Dmini/FFHQ-64x64")
    dt["train"].save_to_disk("/home/vmeshchaninov/VisionDiffusion/datasets/FFHQ-64x64/tmp")


class FFHQDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.dataset = load_from_disk(root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        if self.transform is not None:
            image = self.transform(item["image"])
        else:
            image = item["image"]
        return (image, item["label"])


class DataGenerator:
    def __init__(self, config):
        self.config = config
        self.root_path = os.path.join(config.data.dataset_path, "tmp")
        self.train_transforms = Compose(
            [
                # Resize((config.data.image_size, config.data.image_size)),
                RandomHorizontalFlip(p=0.5),
                ToTensor(),
                Normalize(mean=config.data.norm_mean, std=config.data.norm_std),
                # to [-1; 1]
            ]
        )

        self.make_train_loader()

    def make_train_loader(self):
        self.train_dataset = FFHQDataset(
            root=self.root_path,
            transform=self.train_transforms
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
