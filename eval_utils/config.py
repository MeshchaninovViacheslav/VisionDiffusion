import ml_collections

def create_inference_config():
    inference = ml_collections.ConfigDict()

    inference.batch_size = 10000
    inference.total_images = 50000
    inference.checkpoints_folder = '/home/vmeshchaninov/VisionDiffusion/checkpoints'
    inference.checkpoints_prefix = 'ddpm_cifar'
    inference.checkpoints_name = "400000"
    inference.image_path = f"/home/vmeshchaninov/VisionDiffusion/generated_images/" \
                           f"{inference.checkpoints_prefix}_ddim_100/{inference.checkpoints_name}/"

    return inference