import ml_collections


def create_big_model_config():
    model = ml_collections.ConfigDict()
    model.dropout = 0.1
    model.embedding_type = 'fourier'
    model.name = 'ncsnpp'
    model.scale_by_sigma = False
    model.ema_rate = 0.9999
    model.normalization = 'GroupNorm'
    model.nonlinearity = 'swish'
    model.nf = 128
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 4
    model.attn_resolutions = (16,)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = False
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = 'biggan'
    model.progressive = 'none'
    model.progressive_input = 'none'
    model.progressive_combine = 'sum'
    model.attention_type = 'ddpm'
    model.init_scale = 0.
    model.embedding_type = 'positional'
    model.fourier_scale = 16
    model.conv_size = 3
    return model


def create_default_mnist_config():
    config = ml_collections.ConfigDict()

    # data
    data = config.data = ml_collections.ConfigDict()
    data.image_size = 32
    data.num_channels = 3
    data.centered = True
    data.norm_mean = (0.5)
    data.norm_std = (0.5)

    optim = config.optim = ml_collections.ConfigDict()
    optim.grad_clip_norm = 1.0
    optim.linear_warmup = 5000
    optim.lr = 2e-4
    optim.min_lr = 2e-4
    optim.warmup_lr = 0
    optim.weight_decay = 0
    optim.betas = (0.9, 0.999)
    optim.eps = 1e-8

    config.model = create_big_model_config()

    # training
    training = config.training = ml_collections.ConfigDict()
    training.training_iters = 500_000
    training.checkpoint_freq = 25_000
    training.eval_freq = 25_000
    training.snapshot_freq = 25_000
    training.snapshot_batch_size = 128
    training.batch_size = 128
    training.batch_size_per_gpu = training.batch_size
    training.ode_sampling = False
    training.logging_freq = 10
    training.exp_name = 'cosine_sde'

    training.checkpoints_folder = './checkpoints/'

    # sde
    sde = config.sde = ml_collections.ConfigDict()
    sde.N = 1000
    sde.beta_min = 0.1
    sde.beta_max = 20
    sde.scheduler = "cosine_sde"

    config.project = "vision_diffusion"
    config.checkpoints_prefix = 'cosine_sde'
    config.predict = 'noise'
    config.device = 'cuda'
    config.chkp_name = "cosine_sde-475000.pth"
    config.ddp = True
    config.seed = 0
    config.dataset = "cifar10"
    return config
