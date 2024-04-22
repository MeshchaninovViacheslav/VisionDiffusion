import torch
import ml_collections

from models.model_config import create_big_model_config
from models.model_config import create_small_model_config

from data.CIFAR_config import cifar_config
from data.LSUN_config import lsun_config
from data.FFHQ_config import ffhq_config

from eval_utils.config import create_inference_config


def create_default_cifar_config():
    config = ml_collections.ConfigDict()

    # data
    config.data = cifar_config()

    # model
    config.model = create_small_model_config()

    # optim
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = None
    optim.linear_warmup = 5000
    optim.lr = 1e-4
    optim.min_lr = 1e-4
    optim.warmup_lr = 0
    optim.weight_decay = 0
    optim.betas = (0.9, 0.999)
    optim.eps = 1e-8

    # training
    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 500_000
    training.checkpoint_freq = 50_000
    training.eval_freq = 50_000
    training.snapshot_freq = 1_000
    training.snapshot_batch_size = 25
    training.batch_size = 128
    training.batch_size_per_gpu = training.batch_size
    training.ode_sampling = True
    training.log_freq = 1
    training.num_type = torch.bfloat16

    # inference
    config.inference = create_inference_config()

    # sde
    dynamic = config.dynamic = ml_collections.ConfigDict()
    dynamic.typename = 'vp-sde'
    dynamic.scheduler = "cosine"
    dynamic.beta_min = 0.1
    dynamic.beta_max = 20
    dynamic.step_size = 0.04
    dynamic.N = 250
    dynamic.solver = "ddim"
    dynamic.T = 1.
    dynamic.eps = 0.001

    config.project_name = 'test'
    config.experiment_name = "cont_time_cd_correct_time_sampling-" + config.inference.checkpoints_prefix
    config.parametrization = 'x_0'
    config.seed = 0
    config.validate = False
    config.timesteps = "linear"
    config.teacher_checkpoint_name = "/home/echimbulatov/shared_folder/vision_diffusion/ddpm-cifar-cosine_sde-x_0/500000.pth"
    config.init_checkpoint_name = "/home/echimbulatov/shared_folder/vision_diffusion/ddpm-cifar-cosine_sde-x_0/500000.pth"
    config.loss_bc_freq = 4
    config.loss_bc_beta = 1
    config.clip_target = True
    config.solver_type = "ddim"

    return config
