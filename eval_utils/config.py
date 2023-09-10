import ml_collections

def create_inference_config():
    inference = ml_collections.ConfigDict()

    inference.batch_size = 20
    inference.total_images = 50000
    inference.checkpoints_folder = '/home/vmeshchaninov/VisionDiffusion/checkpoints'
    inference.checkpoints_prefix = "ddpm-ffhq-eps-v3"
    inference.checkpoints_name = "last"
    inference.image_path = f"/home/vmeshchaninov/VisionDiffusion/generated_images/" \
                           f"{inference.checkpoints_prefix}_heun/{inference.checkpoints_name}/"

    return inference