# Continious DDPM
## Training
To start training on CIFAR-10 use:

`$ python train_model.py`

Checkpoints will be stored in `./ddpm_checkpoints`

### Parameters:
- `--batch_size` - size of the batch, `default=128`
- `--iters` - number of training iterations,`default=350_000`
- `--device` - computational device, `default='cuda:0'`
- `--parametrization` - chooses parametrization, `default='eps'`. Can be either `'eps'` or `'x_0'`
- `--checkpoints_prefix` - prefix for needed checkpoints, `'default=''`
- `--exp_name`, `'default='vp-sde'`
- `--proj_name`, `'default='vp-diffusion'`

## Generation 

To generate samples use `$ python run_diff_gen.py --path='path_name'`, where `'path_name'` <br /> 
is the name of the folder to contain them

### Parameters:
- `--batch_size` - size of the batch, `default=250`
- `--idx_start` - global counter for generated samples (for distributed computations),`default=0` 
- `--total_images` - total number of samples to generate, `default=50000`
- `--device` - computational device, `default='cuda:0'`
- `--parametrization` - chooses parametrization, `default='eps'`. Can be either `'eps'` or `'x_0'`
- `--checkpoints_folder` - path to required checkpoints folder, `'default='./ddpm_checkpoints'`

## Evaluation 

To evaluate FID scores use `$ python evaluate.py --dataset2='ds_path' --proj_name='name'`,
where `'ds_path'` is the path to generated images and `'name'` is the
name of the project with which the images are associated. Default dataset for evluation is CIFAR10 with 50 000 images

### Parameters

- `--batch_size` - size of the batch, `default=128`
- `--total_images` - total number of samples to evaluate, `default=8192`
- `--device` - computational device, `default='cuda:0'`
- `--log_path` - path to FID loggings .csv file directory, `default='fid_logs.csv'`.
If such file doesn't exist, makes one
- `--dataset1` - path to dataset1, `default='CIFAR10_png'`

