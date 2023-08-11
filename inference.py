import tqdm
import cv2
import os
import numpy as np

from config import create_default_mnist_config
from diffusion import DiffusionRunner
from utils.calculate_fid import compute_fid_mnist
from utils.utils import set_seed


def inference_run(diffusion, num_images=10_000, batch_size=500):
    save_path = f'mnist_generated_samples/images-ddm-{num_images}-{diffusion.config.sde.N}'
    os.makedirs(save_path, exist_ok=True)

    image_ind = 0
    for _ in tqdm.tqdm(range(0, num_images, batch_size)):
        x_pred_batch = diffusion.inference(batch_size)

        for j, x_pred in enumerate(x_pred_batch):
            cv2.imwrite(f'{save_path}/{image_ind:05d}.png', x_pred.permute(1, 2, 0).numpy().astype(np.uint8))
            image_ind += 1

    return save_path


def main(N=10):
    set_seed(0)
    config = create_default_mnist_config()
    config.sde.N = N
    diffusion = DiffusionRunner(config, eval=True)
    save_path = inference_run(diffusion)
    compute_fid_mnist(save_path)

# from inference import main; main()
