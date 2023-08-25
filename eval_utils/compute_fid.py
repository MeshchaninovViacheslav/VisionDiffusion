import os
import sys

sys.path.append("/home/vmeshchaninov/VisionDiffusion/")

from pytorch_fid import fid_score

def compute_fid(dataset_path_real: str, dataset_path_gen: str) -> float:
    print(f"Number of real images = {len(os.listdir(dataset_path_real))}")
    print(f"Number of gen images = {len(os.listdir(dataset_path_gen))}")

    fid_value = fid_score.calculate_fid_given_paths(
        paths=[dataset_path_real, dataset_path_gen],
        batch_size=1000,
        device='cuda:0',
        dims=2048,
        num_workers=30,
    )

    print("FID: ", fid_value)
    return fid_value