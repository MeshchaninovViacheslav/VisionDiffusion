import ml_collections

from models.model_config import create_big_model_config
from data.CIFAR_config import cifar_config
from eval_utils.config import create_inference_config


def create_default_cifar_config():
    config = ml_collections.ConfigDict()

    # data
    config.data = cifar_config()

    # model
    config.model = create_big_model_config()

    # optim
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.0
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 0
    optim.weight_decay = 0
    optim.betas = (0.9, 0.999)
    optim.eps = 1e-8

    # training
    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 500_000
    training.checkpoint_freq = 50_000
    training.eval_freq = 50_000
    training.snapshot_freq = 50_000
    training.snapshot_batch_size = 100
    training.batch_size = 128
    training.batch_size_per_gpu = training.batch_size
    training.ode_sampling = False
    training.log_freq = 10

    # inference
    config.inference = create_inference_config()

    # sde
    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 1000
    sde.beta_min = 0.1
    sde.beta_max = 20

    config.project_name = 'integrators'
    config.experiment_name = config.inference.checkpoints_prefix
    config.parametrization = 'eps'
    config.seed = 0
    return config
