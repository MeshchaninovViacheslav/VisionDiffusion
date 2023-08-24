import os
import torch
import random
import numpy as np
import torch.distributed as dist
import itertools

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def gather_images(images):
    output = [None for _ in range(dist.get_world_size())]
    gather_objects = images
    dist.all_gather_object(output, gather_objects)
    gathered_images = torch.cat(output)
    return gathered_images