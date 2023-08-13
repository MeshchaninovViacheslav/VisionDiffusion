import os
import torch
import torch.distributed as dist

from config import create_default_mnist_config
from diffusion import DiffusionRunner
from utils.utils import set_seed

config = create_default_mnist_config()

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
seed = config.seed
set_seed(seed)

diffusion = DiffusionRunner(config)

seed = config.seed + dist.get_rank()
set_seed(seed)

diffusion.train()
