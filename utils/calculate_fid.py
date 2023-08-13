import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from tqdm import tqdm

import os
import cv2
from config import create_default_mnist_config
from data.MNIST_dataset import MNISTDataGenerator
from data.CIFAR_dataset import CIFARDataGenerator, InferenceGenerator


@torch.no_grad()
def accuracy(output, target):
    """Computes the accuracy"""
    assert output.shape == target.shape
    return torch.sum(output == target) / target.size(0)


def read_gen_images(gen_path, num_images):
    gen_list = os.listdir(gen_path)
    gen_images = []

    for i, image in tqdm(enumerate(gen_list), desc="Loading gen images"):
        if '.png' not in image:
            continue

        img = cv2.imread(f'{gen_path}/{image}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))
        img_tensor = torch.tensor(img, dtype=torch.uint8)[None, ...]

        gen_images.append(img_tensor)

        if sum([t.shape[0] for t in gen_images]) >= num_images:
            break

    gen_images = torch.cat(gen_images)
    return gen_images


def read_real_images(num_images):
    dataloader = InferenceGenerator(create_default_mnist_config(), is_real=True).loader
    real_images = []

    for (x, y) in tqdm(dataloader, desc="Loading real images"):
        real_images.append(x)

        if sum([t.shape[0] for t in real_images]) >= num_images:
            break

    real_images = torch.cat(real_images)[:num_images]
    return real_images


def compute_fid(gen_path: str, num_images=50000):
    metric = FrechetInceptionDistance(feature=2048, normalize=False).to('cuda')
    batch_images_fid = 100

    real_images = read_real_images(num_images).type(torch.uint8)
    gen_images = read_gen_images(gen_path, num_images)
    print(gen_images.shape, real_images.shape)
    print(gen_images.dtype, real_images.dtype)

    # assert gen_images.shape == real_images.shape

    for i in tqdm(range(0, gen_images.shape[0], batch_images_fid), desc="FID updating"):
        metric.update(gen_images[i:i + batch_images_fid].cuda(), real=False)
        metric.update(real_images[i:i + batch_images_fid].cuda(), real=True)

    result = metric.compute()
    print(f'FID result: {result}')
    return result





#
# from utils.calculate_fid import compute_fid
# gen_path = "generated_samples/cifar10/sde-50000-2000"
# compute_fid(gen_path)
