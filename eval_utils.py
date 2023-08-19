import numpy as np
import pandas as pd
import torch
import argparse
import os
import pathlib
import random
from skimage.io import imsave
from tqdm.auto import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import Resize, Compose
from time import gmtime, strftime
from pytorch_fid import fid_score
from pytorch_fid.inception import InceptionV3
from torch.backends import cudnn


IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}


def compute_statistics_of_path(path, model, batch_size, dims, device, total_images,
                               num_workers=1):
    """
    Calculates the expectation and variance of two paths
    The difference between pytorch_fid version is the ability to choose
    quantity of images to evaluate, i.e. total_images parameter
    """
    if path.endswith('.npz'):
        with np.load(path) as f:
            m, s = f['mu'][:], f['sigma'][:]
    else:
        path = pathlib.Path(path)
        files = sorted([file for ext in IMAGE_EXTENSIONS
                       for file in path.glob('*.{}'.format(ext))])
        m, s = fid_score.calculate_activation_statistics(files[:total_images], model, batch_size,
                                               dims, device, num_workers)

    return m, s


def calculate_fid_given_paths(paths, batch_size, device, dims, total_images, num_workers=1):
    """
    Calculates the FID of two paths
    The difference between pytorch_fid version is the ability to choose
    quantity of images to evaluate, i.e. total_images parameter
    """
    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError('Invalid path: %s' % p)

    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]

    model = InceptionV3([block_idx]).to(device)

    m1, s1 = compute_statistics_of_path(paths[0], model, batch_size,
                                        dims, device, total_images, num_workers)
    m2, s2 = compute_statistics_of_path(paths[1], model, batch_size,
                                        dims, device, total_images, num_workers)
    fid_value = fid_score.calculate_frechet_distance(m1, s1, m2, s2)

    return fid_value


def parse_eval_args():
    """
    Parses argument for model evaluation, for details: README.md
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--total_images', type=int, default=8192)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--dataset1', type=str, default='CIFAR10_pngs')
    parser.add_argument('--log_path', type=str, default='fid_logs.csv')
    parser.add_argument('--dataset2', type=str, required=True)
    parser.add_argument('--proj_name', type=str, default="cifar")
    return parser.parse_args()


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


def set_seed(seed: int = 0):
    """
    Sets all random seeds to one number for results reproducing purposes
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.benchmark = True

import numpy as np
import os
import cv2
from tqdm.auto import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import Resize, Compose


def cifar10_to_png(total_images=50000):
    """
    Downloads total_images of CIFAR10 if it isn't downloaded yet
    INPUT:
        -- total_images, int
    """
    path = "/home/vmeshchaninov/VisionDiffusion/data/CIFAR_real_samples_imsave"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        real_dataset = CIFAR10(
            root='/home/vmeshchaninov/VisionDiffusion/data/CIFAR',
            download=True, train=True,
            transform=Compose([Resize((32, 32))])
        )
        for idx, (image_cifar, label) in enumerate(tqdm(real_dataset, total=total_images)):
            image = np.array(image_cifar)
            imsave(f"{path}/{idx:05d}.png", image)
            #cv2.imwrite(f"{path}/{idx:05d}.png", image)

import sys

sys.path.append("/home/vmeshchaninov/VisionDiffusion/")

from pytorch_fid import fid_score

def compute_fid_folders():
    total_images = 50000
    #cifar10_to_png(total_images)

    dataset1 = "/home/vmeshchaninov/VisionDiffusion/data/CIFAR_real_samples"
    dataset2 = "/home/vmeshchaninov/VisionDiffusion/generated_images/cifar10_ddpm/350000"


    fid_value = fid_score.calculate_fid_given_paths(
        paths=[dataset1, dataset2],
        batch_size=1000,
        device='cuda:0',
        dims=2048,
        num_workers=30,
    )

    print(fid_value)