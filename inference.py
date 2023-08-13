import tqdm
import cv2
import os
import numpy as np
import torch
import torch.distributed as dist

from config import create_default_mnist_config
from diffusion import DiffusionRunner
from utils.calculate_fid import compute_fid
from utils.utils import set_seed


def inference_run(diffusion, config, num_images=10_000, batch_size=500):
    save_path = f'/home/vmeshchaninov/VisionDiffusion/generated_samples/{config.dataset}/sde-{num_images}-{config.sde.N}'
    os.makedirs(save_path, exist_ok=True)

    image_ind = 0
    for _ in tqdm.tqdm(range(0, num_images, batch_size)):
        x_pred_batch = diffusion.inference_ddp(batch_size // dist.get_world_size())

        if dist.get_rank() == 0:
            print(f"Saving batch with shape of {x_pred_batch.shape}")
            for j, x_pred in enumerate(x_pred_batch):
                cv2.imwrite(f'{save_path}/{image_ind:05d}.png', x_pred.permute(1, 2, 0).numpy().astype(np.uint8))
                image_ind += 1

    return save_path


def ddp_init(config):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1

    config.local_rank = rank
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()


def main(N=2000):
    config = create_default_mnist_config()
    ddp_init(config)
    seed = config.seed + dist.get_rank()
    set_seed(seed)

    config.sde.N = N
    diffusion = DiffusionRunner(config, eval=True)
    save_path = inference_run(diffusion, config, num_images=50000, batch_size=2000)
    compute_fid_mnist(save_path)


main()

# from inference import main; main()
