from default_cifar_config import create_default_cifar_config
from diffusion import DiffusionRunner
import argparse

def parse_train_args():
    """
    Parses arguments for model training, for details: README.md
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--iters', type=int, default=350_000)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--checkpoints_prefix', type=str, default='')
    parser.add_argument('--parametrization', type=str, default='eps')
    parser.add_argument('--exp_name', type=str, default='vp-sde')
    parser.add_argument('--proj_name', type=str, default='vp-diffusion')
    return parser.parse_args()


# args = parse_train_args()
# config = create_default_cifar_config()
# config.training.batch_size = args.batch_size
# config.training.training_iters = args.iters
# config.device = args.device
# config.checkpoints_prefix = args.checkpoints_prefix
# config.parametrization = args.parametrization
# config.experiment_name = args.exp_name
# config.project_name = args.proj_name
config = create_default_cifar_config()
diffusion = DiffusionRunner(config)
diffusion.train()
