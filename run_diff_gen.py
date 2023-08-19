import torch
import torchvision
import math
import ml_collections
import numpy as np

from skimage.io import imsave
from default_cifar_config import create_default_cifar_config
from diffusion import DiffusionRunner
from typing import Optional, Sequence
from tqdm.auto import trange

from sys import argv
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=250)
    parser.add_argument('--idx_start', type=int, default=0)
    parser.add_argument('--total_images', type=int, default=50000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--parametrization', type=str, default='eps')
    parser.add_argument('--checkpoints_folder', type=str, default='./ddpm_checkpoints')
    parser.add_argument('--path', type=str, required=True)
    return parser.parse_args()


def create_inference_config():
    config = ml_collections.ConfigDict()

    config.batch_size = 2000
    config.total_images = 50000
    config.checkpoints_folder = './d_checkpoints'
    config.checkpoints_name = "last_ema"
    config.image_path = "./generated_images/cifar10_ddpm_egor/350000/"

    return config


config = create_default_cifar_config()
config_inference = create_inference_config()
config.inference = config_inference

config.checkpoints_prefix = ''
config.training.checkpoints_folder = config_inference.checkpoints_folder

gen = DiffusionRunner(config, eval=True)

total_images = config_inference.total_images
batch_size = config_inference.batch_size
folder_path = config_inference.image_path

os.makedirs(folder_path, exist_ok=True)

num_iters = total_images // batch_size

global_idx = 0

for idx in trange(num_iters):
    images: torch.Tensor = gen.sample_images(batch_size=batch_size).cpu()
    images = images.permute(0, 2, 3, 1).data.numpy().astype(np.uint8)

    for i in range(len(images)):
        imsave(os.path.join(folder_path, f'{global_idx}:05d.png'), images[i])
        global_idx += 1
