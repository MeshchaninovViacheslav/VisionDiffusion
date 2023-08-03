import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

import os
import cv2
from config import create_default_mnist_config
from data_generator import DataGenerator


@torch.no_grad()
def accuracy(output, target):
    """Computes the accuracy"""
    assert output.shape == target.shape
    return torch.sum(output == target) / target.size(0)


def read_gen_images(gen_path):
    gen_list = os.listdir(gen_path)
    gen_images = []

    for i, image in tqdm(enumerate(gen_list)):
        if '.png' not in image:
            continue

        img = cv2.imread(f'{gen_path}/{image}')
        img = img.transpose((2, 0, 1))
        img_tensor = torch.tensor(img)[None, ...] / 255.

        gen_images.append(img_tensor)

    gen_images = torch.cat(gen_images)
    return gen_images


def read_real_images(num_images):
    dataloader = DataGenerator(create_default_mnist_config())
    real_images = []

    for (x, y) in tqdm(dataloader.valid_loader):
        x = torch.cat((x, x, x), dim=1) / 255.
        real_images.append(x)

        if sum([t.shape[0] for t in real_images]) >= num_images:
            break

    real_images = torch.cat(real_images)[:num_images]
    return real_images


def compute_fid_mnist(gen_path: str, num_images=10000):
    metric = FrechetInceptionDistance(feature=768, normalize=True).to('cuda')
    batch_images_fid = 100

    gen_images = read_gen_images(gen_path)
    real_images = read_real_images(num_images)
    print(gen_images.shape, real_images.shape)

    assert gen_images.shape == real_images.shape

    for i in range(0, gen_images.shape[0], batch_images_fid):
        metric.update(gen_images[i:i + batch_images_fid].cuda(), real=False)
        metric.update(real_images[i:i + batch_images_fid].cuda(), real=True)

    result = metric.compute()
    print(f'FID result: {result}')
    return result


# from calculate_fid import compute_fid_mnist
# gen_path = "mnist_generated_samples/images-10000-1000"
# compute_fid_mnist(gen_path)