import os
import torch
import torch.distributed as dist

from ddpm_config import create_default_cifar_config
from diffusion import DiffusionRunner
from utils.utils import set_seed
from eval_utils.generation import generate_images
from eval_utils.compute_fid import compute_fid
from eval_utils.utils import dataset_to_png

config = create_default_cifar_config()

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

config.training.batch_size_per_gpu = config.training.batch_size // dist.get_world_size()

diffusion = DiffusionRunner(config)

seed = config.seed + dist.get_rank()
set_seed(seed)

diffusion.train()

diffusion.model.eval()
generate_images(
    diffusion,
    total_images=config.inference.total_images,
    batch_size=config.inference.batch_size,
    image_path=config.inference.image_path,
)

if dist.get_rank() == 0:
    dataset_to_png(config, total_images=config.inference.total_images)

    compute_fid(
        dataset_path_real=f"{config.data.dataset_path}/samples",
        dataset_path_gen=config.inference.image_path
    )
