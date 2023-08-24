import ml_collections


def cifar_config():
    data = ml_collections.ConfigDict()

    # data
    data.image_size = 32
    data.num_channels = 3
    data.centered = True
    data.norm_mean = (0.5)
    data.norm_std = (0.5)
    data.dataset_path = '/home/vmeshchaninov/VisionDiffusion/datasets/CIFAR'
    return data
