import pandas as pd
import numpy as np
import os
from tqdm.auto import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import Resize, Compose
from skimage.io import imsave

from time import gmtime, strftime

from data.FFHQ_dataset import FFHQDataset


def log_fid(fid_value, log_path, proj_name,
            total_images, time='unk'):
    """
    Logs fid values
    INPUT:
        -- fid_value, float
        -- log_path, str
        -- proj_name, str
        -- total_images, int
        -- time, datetime.timedelta
    """
    try:
        df = pd.read_csv(log_path)
    except FileNotFoundError:
        df = pd.DataFrame(columns=['FID', 'Project', 'Date', 'Time', 'total_images'])

    date = strftime("%Y-%d-%m", gmtime())
    df.loc[-1] = [fid_value, proj_name, date, time, total_images]
    df.index = df.index + 1
    df = df.sort_index()
    df.to_csv(log_path, index=False)


def cifar10_to_png(config, total_images=50000):
    """
    Downloads total_images of CIFAR10 if it isn't downloaded yet
    INPUT:
        -- total_images, int
    """
    path = os.path.join(config.data.dataset_path, "samples")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        real_dataset = CIFAR10(
            root=os.path.join(config.data.dataset_path, "tmp"),
            download=True, train=True,
            transform=Compose([Resize((32, 32))])
        )
        for idx, (image_cifar, label) in enumerate(tqdm(real_dataset, total=total_images)):
            image = np.array(image_cifar)
            imsave(f"{path}/{idx:05d}.png", image)


def ffhq_to_png(config, total_images=50000):
    path = os.path.join(config.data.dataset_path, "samples")
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        real_dataset = FFHQDataset(
            root=os.path.join(config.data.dataset_path, "tmp"),
        )
        for idx, (image, label) in enumerate(tqdm(real_dataset, total=total_images)):
            image = np.array(image)
            imsave(f"{path}/{idx:05d}.png", image)
            if idx >= total_images:
                return


def dataset_to_png(config, total_images):
    if "cifar" in config.inference.checkpoints_prefix:
        cifar10_to_png(config, total_images)
    elif "ffhq" in config.inference.checkpoints_prefix:
        ffhq_to_png(config, total_images)
