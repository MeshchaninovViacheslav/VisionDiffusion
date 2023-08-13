import sys

sys.path.append("/home/vmeshchaninov/VisionDiffusion/")

from data.utils import cifar10_to_png
from pytorch_fid import fid_score

def compute_fid_folders():
    total_images = 50000
    cifar10_to_png(total_images)

    dataset1 = "/home/vmeshchaninov/VisionDiffusion/data/CIFAR_real_samples"
    dataset2 = "/home/vmeshchaninov/VisionDiffusion/generated_samples/cifar10/sde-50000-2000"


    fid_value = fid_score.calculate_fid_given_paths(
        paths=[dataset1, dataset2],
        batch_size=1000,
        device='cuda:0',
        dims=2048
    )

    print(fid_value)

compute_fid_folders()
