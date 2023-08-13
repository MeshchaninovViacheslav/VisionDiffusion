import numpy as np
import os
import cv2
from tqdm.auto import tqdm
from torchvision.datasets import CIFAR10
from torchvision.transforms import Resize, Compose


def cifar10_to_png(total_images):
    """
    Downloads total_images of CIFAR10 if it isn't downloaded yet
    INPUT:
        -- total_images, int
    """
    path = "/home/vmeshchaninov/VisionDiffusion/data/CIFAR_real_samples"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        real_dataset = CIFAR10(
            root='/home/vmeshchaninov/VisionDiffusion/data/CIFAR',
            download=True, train=True,
            transform=Compose([Resize((32, 32))])
        )
        for idx, (image_cifar, label) in enumerate(tqdm(real_dataset, total=total_images)):
            image = np.array(image_cifar)
            cv2.imwrite(f"{path}/{idx:05d}.png", image)