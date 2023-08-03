from config import create_default_mnist_config
from diffusion import DiffusionRunner

config = create_default_mnist_config()
diffusion = DiffusionRunner(config)

diffusion.train()