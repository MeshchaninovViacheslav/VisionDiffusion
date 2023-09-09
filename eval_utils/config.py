import ml_collections

def create_inference_config():
    inference = ml_collections.ConfigDict()

    inference.batch_size = 2000
    inference.total_images = 50000
    inference.checkpoints_folder = '/home/vmeshchaninov/VisionDiffusion/checkpoints'
    inference.checkpoints_prefix = "boots-ffhq-ddim-sd-x_0+eps"
    inference.checkpoints_name = "last"
    inference.image_path = f"/home/vmeshchaninov/VisionDiffusion/generated_images/" \
                           f"{inference.checkpoints_prefix}/{inference.checkpoints_name}/"

    return inference