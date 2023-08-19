import ml_collections

from models.model_config import create_big_model_config


def create_default_cifar_config():
    config = ml_collections.ConfigDict()

    # data
    data = config.data = ml_collections.ConfigDict()
    data.image_size = 32
    data.num_channels = 3
    data.centered = True
    data.norm_mean = (0.5)
    data.norm_std = (0.5)

    # model
    config.model = create_big_model_config()

    # optim
    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.0
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.weight_decay = 0

    # training
    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 350_000
    training.checkpoint_freq = 50_000
    training.eval_freq = 50_000
    training.snapshot_freq = 50_000
    training.snapshot_batch_size = 100
    training.batch_size = 128
    training.ode_sampling = False

    training.checkpoints_folder = './ddpm_checkpoints'
    config.checkpoints_prefix = ''

    # sde
    sde = config.sde = ml_collections.ConfigDict()
    sde.typename = 'vp-sde'
    sde.solver = 'euler'
    sde.N = 1000
    sde.beta_min = 0.1
    sde.beta_max = 20

    config.project_name = 'integrators'
    config.experiment_name = 'cifar-training'
    config.parametrization = 'eps'
    config.device = "cuda:0"
    return config
