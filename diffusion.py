import torch
import torchvision
import wandb
import os
import math

import numpy as np

from models.utils import create_model
from models.ema import ExponentialMovingAverage
from ddpm_sde import create_sde, create_solver
from data_generator import DataGenerator

from ml_collections import ConfigDict
from typing import Optional, Union, Dict, Tuple
from tqdm import trange, tqdm
from torch.nn import functional as F


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False
    ):
        self.config = config
        self.eval = eval
        self.parametrization = config.parametrization

        self.model = create_model(config=config)
        self.sde = create_sde(config=config)
        self.diff_eq_solver = create_solver(
            config, self.sde,
            ode_sampling=config.training.ode_sampling
        )
        self.inverse_scaler = lambda x: torch.clip(127.5 * (x + 1), 0, 255)

        self.checkpoints_folder = config.training.checkpoints_folder
        if eval:
            self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
            self.restore_parameters()
            self.model.eval()
            self.switch_to_ema()
        self.model = torch.nn.DataParallel(self.model)

        device = torch.device(self.config.device)
        self.device = device
        self.model.to(device)
        self.project_name = config.project_name
        self.experiment_name = config.experiment_name

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        if device is None:
            device = torch.device('cpu')
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix + '_'
        ema_ckpt = torch.load(
            checkpoints_folder + '/' + prefix + f'{self.config.inference.checkpoints_name}.pth',
            map_location='cpu'
        )
        self.ema.load_state_dict(ema_ckpt)

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.model
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.model
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.rand(batch_size) * (self.sde.T - eps) + eps

    def calc_loss(self, clean_x: torch.Tensor, eps: float = 1e-5) -> Dict[str, torch.Tensor]:
        batch_size = clean_x.size(0)
        t = self.sample_time(batch_size, eps=eps).to(clean_x.device)

        marg_forward = self.sde.marginal_forward(clean_x, t)
        x_t, noise = marg_forward['x_t'], marg_forward['noise']

        scores = self.sde.calc_score(self.model, x_t, t)

        if self.parametrization == 'eps':
            eps_theta = scores.pop('eps_theta')
            losses = torch.square((eps_theta - noise) / self.sde.div_std)
        elif self.parametrization == 'x_0':
            x_0 = scores.pop('x_0')
            losses = torch.square((x_0 - clean_x) / self.sde.div_std)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=1)
        loss = torch.mean(losses)
        loss_dict = {
            'loss': loss,
            'total_loss': loss
        }
        return loss_dict

    def set_data_generator(self) -> None:
        self.datagen = DataGenerator(self.config)

    def manage_optimizer(self) -> None:
        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm
            )
        self.lrs = []
        if self.warmup > 0 and self.step < self.warmup:
            for g in self.optimizer.param_groups:
                self.lrs += [g['lr']]
                g['lr'] = g['lr'] * float(self.step + 1) / self.warmup

    def restore_optimizer_state(self) -> None:
        if self.lrs:
            self.lrs = self.lrs[::-1]
            for g in self.optimizer.param_groups:
                g['lr'] = self.lrs.pop()

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()

        self.manage_optimizer()
        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.optimizer.step()
        self.ema.update(self.model.parameters())
        self.restore_optimizer_state()

    def validate(self) -> None:
        prev_mode = self.model.training

        self.model.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = 0
        with torch.no_grad():
            for (X, y) in self.datagen.valid_loader:
                X = X.to(self.device)

                loss_dict = self.calc_loss(clean_x=X)
                for k, v in loss_dict.items():
                    if k in valid_loss:
                        valid_loss[k] += v.item() * X.size(0)
                    else:
                        valid_loss[k] = v.item() * X.size(0)
                valid_count += X.size(0)

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        self.valid_loss = valid_loss['total_loss']
        for k, v in valid_loss.items():
            self.log_metric(k, 'valid_loader', v)

        self.switch_back_from_ema()
        self.model.train(prev_mode)

    def train(self) -> None:
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)
        self.set_optimizer()
        self.set_data_generator()
        train_generator = self.datagen.sample_train()
        self.train_gen = train_generator
        self.step = 0

        wandb.init(project=self.project_name, name=self.experiment_name)

        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        self.model.train()

        for iter_idx in trange(1, 1 + self.config.training.training_iters):
            self.step = iter_idx

            (X, y) = next(train_generator)
            X = X.to(self.device)

            loss_dict = self.calc_loss(clean_x=X)
            for k, v in loss_dict.items():
                self.log_metric(k, 'train', v.item())
            self.optimizer_step(loss_dict['total_loss'])

            if iter_idx % self.config.training.snapshot_freq == 0:
                self.snapshot()

            if iter_idx % self.config.training.eval_freq == 0:
                self.validate()

            if iter_idx % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

        self.model.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()

    def save_checkpoint(self, last: bool = False) -> None:
        if not os.path.exists(self.checkpoints_folder):
            os.makedirs(self.checkpoints_folder)
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix + '_'
        if last:
            prefix = prefix + 'last_'
        else:
            prefix = prefix + str(self.step) + '_'
        if True:
            torch.save(self.model.state_dict(), os.path.join(self.checkpoints_folder,
                                                             prefix + f'model.pth'))
            torch.save(self.ema.state_dict(), os.path.join(self.checkpoints_folder,
                                                           prefix + f'ema.pth'))
            torch.save(self.optimizer.state_dict(), os.path.join(self.checkpoints_folder,
                                                                 prefix + f'opt.pth'))

    def sample_tensor(
            self, batch_size: int,
            eps: float = 1e-3,
            verbose: bool = True
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.data.num_channels,
            self.config.data.image_size,
            self.config.data.image_size
        )
        device = torch.device(self.config.device)
        with torch.no_grad():
            x = x_mean = self.sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(self.sde.T, eps, self.sde.N, device=device)
            rang = trange if verbose else range
            for idx in tqdm(range(self.sde.N)):
                t = timesteps[idx]
                input_t = t * torch.ones(shape[0], device=device)
                new_state = self.diff_eq_solver.step(self.model, x, input_t)
                x, x_mean = new_state['x'], new_state['x_mean']
        return x_mean

    def sample_images(
            self, batch_size: int,
            eps: float = 1e-3,
            verbose: bool = False
    ) -> torch.Tensor:
        x_mean = self.sample_tensor(batch_size, eps, verbose)
        return self.inverse_scaler(x_mean)

    def snapshot(self) -> None:
        prev_mode = self.model.training
        self.model.eval()
        self.switch_to_ema()

        images = self.sample_images(self.config.training.snapshot_batch_size).cpu()
        nrow = int(math.sqrt(self.config.training.snapshot_batch_size))
        grid = torchvision.utils.make_grid(images, nrow=nrow).permute(1, 2, 0)
        grid = grid.data.numpy().astype(np.uint8)
        self.log_metric('images', 'from_noise', wandb.Image(grid))

        self.switch_back_from_ema()
        self.model.train(prev_mode)
