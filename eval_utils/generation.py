import os
import torch
import numpy as np
from skimage.io import imsave
from tqdm.auto import trange
import torch.distributed as dist
import sys

sys.path.append("/home/vmeshchaninov/VisionDiffusion/")

from ddpm_config import create_default_cifar_config
from diffusion import DiffusionRunner


def evaluate():
    config = create_default_cifar_config()

    gen = DiffusionRunner(config, eval=True)

    total_images = config.inference.total_images
    batch_size = config.inference.batch_size
    image_path = config.inference.image_path
    generate_images(gen, total_images, batch_size, image_path)


def generate_images(diffusion, total_images, batch_size, image_path):
    os.makedirs(image_path, exist_ok=True)

    num_iters = total_images // batch_size + 1

    global_idx = 0

    for idx in trange(num_iters):
        tmp_batch_size = min(batch_size, total_images - global_idx)
        images: torch.Tensor = diffusion.sample_images(batch_size=tmp_batch_size).cpu()
        images = images.permute(0, 2, 3, 1).data.numpy().astype(np.uint8)

        for i in range(len(images)):
            if dist.get_rank() == 0:
                imsave(os.path.join(image_path, f'{global_idx:05d}.png'), images[i])
            global_idx += 1

        if global_idx >= total_images:
            break
