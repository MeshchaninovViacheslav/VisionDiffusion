from config import create_default_mnist_config
from diffusion import DiffusionRunner
from inference import inference_run

config = create_default_mnist_config()
diffusion = DiffusionRunner(config)

diffusion.train()

inference_run(diffusion)

