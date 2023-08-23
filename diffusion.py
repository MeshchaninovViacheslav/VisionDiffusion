import torch
import torchvision
import wandb
import os
import math
import numpy as np
from torch_ema import ExponentialMovingAverage
from torch.cuda.amp import GradScaler
from timm.scheduler.cosine_lr import CosineLRScheduler
import torch.distributed as dist

from models.utils import create_model
from diffusion_utils.ddpm_sde import create_sde, create_solver
from data.FFHQ_dataset import DataGenerator
from utils.utils import gather_images

from ml_collections import ConfigDict
from typing import Optional, Union, Dict
from tqdm import trange


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
        self.config.total_number_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.device = f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0"
        self.model.to(self.device)
        self.model_without_ddp = self.model
        self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
        self.sde = create_sde(config=config)
        self.diff_eq_solver = create_solver(
            config, self.sde,
            ode_sampling=config.training.ode_sampling
        )
        self.inverse_scaler = lambda x: torch.clip(127.5 * (x + 1), 0, 255)
        self.project_name = config.project_name
        self.experiment_name = config.experiment_name

        if eval:
            self.restore_parameters()
            self.model.eval()
        else:
            self.set_optimizer()
            self.set_data_generator()
            train_generator = self.datagen.sample_train()
            self.train_gen = train_generator
            self.step = 0
            if dist.get_rank() == 0:
                wandb.init(
                    project=self.project_name,
                    name=self.experiment_name,
                    config=dict(self.config),
                )

            if self.load_checkpoint():
                self.snapshot()
                self.validate()


        if dist.is_initialized():
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        load_path = os.path.join(
            self.config.inference.checkpoints_folder,
            self.config.inference.checkpoints_prefix,
            self.config.inference.checkpoints_name + ".pth"
        )
        ema_ckpt = torch.load(
            load_path,
            map_location='cpu'
        )
        self.ema.load_state_dict(ema_ckpt["ema"])

    def load_checkpoint(self) -> int:
        prefix_folder = os.path.join(self.config.inference.checkpoints_folder, self.config.inference.checkpoints_prefix)

        if not os.path.exists(prefix_folder):
            return False

        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return False

        name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"

        load = torch.load(checkpoint_name, map_location="cpu")

        self.ema.load_state_dict(load["ema"])
        self.model_without_ddp.load_state_dict(load["model"])
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.grad_scaler.load_state_dict(load["scaler"])
        self.step = load["step"]
        print(f"Checkpoint loaded {checkpoint_name}")
        return True

    def save_checkpoint(self, last: bool = False) -> None:
        if not dist.get_rank() == 0:
            return

        if not os.path.exists(self.config.inference.checkpoints_folder):
            os.makedirs(self.config.inference.checkpoints_folder)

        prefix_folder = os.path.join(self.config.inference.checkpoints_folder, self.config.inference.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        if last:
            prefix = 'last'
        else:
            prefix = str(self.step)

        save_path = os.path.join(prefix_folder, prefix + ".pth")
        torch.save(
            {
                "model": self.model_without_ddp.state_dict(),
                "ema": self.ema.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "scaler": self.grad_scaler.state_dict(),
                "step": self.step,
            },
            save_path
        )
        print(f"Save model to: {save_path}")

    def set_optimizer(self) -> None:
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.optim.lr,
            betas=self.config.optim.betas,
            eps=self.config.optim.eps,
            weight_decay=self.config.optim.weight_decay
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.grad_scaler = GradScaler()
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.sqrt(sum([torch.sum(t.grad ** 2) for t in self.model.parameters()]))

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm
            )

        self.log_metric('grad_norm', 'train', grad_norm.item())
        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        self.ema.update(self.model.parameters())
        self.scheduler.step_update(self.step)
        return grad_norm

    def set_data_generator(self) -> None:
        self.datagen = DataGenerator(self.config)

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

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if dist.get_rank() == 0:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    @torch.no_grad()
    def validate(self) -> None:
        if not self.config.validate:
            return

        prev_mode = self.model.training
        self.model.eval()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = 0
        with self.ema.average_parameters():
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

        self.model.train(prev_mode)

    def train(self) -> None:
        self.model.train()

        for iter_idx in trange(self.step + 1, self.config.training.training_iters + 1):
            self.step = iter_idx

            (X, y) = next(self.train_gen)
            X = X.to(self.device)

            with torch.cuda.amp.autocast():
                loss_dict = self.calc_loss(clean_x=X)
            if iter_idx % self.config.training.log_freq == 0:
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
        if dist.get_rank() == 0:
            wandb.finish()

    @torch.no_grad()
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
        device = f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0"
        with torch.no_grad():
            x = x_mean = self.sde.prior_sampling(shape).to(device)
            timesteps = torch.linspace(self.sde.T, eps, self.sde.N, device=device)
            rang = trange if verbose else range
            for idx in rang(self.sde.N):
                t = timesteps[idx]
                input_t = t * torch.ones(shape[0], device=device)
                new_state = self.diff_eq_solver.step(self.model, x, input_t)
                x, x_mean = new_state['x'], new_state['x_mean']
        return x_mean

    @torch.no_grad()
    def sample_images(
            self, batch_size: int,
            eps: float = 1e-3,
            verbose: bool = True
    ) -> torch.Tensor:
        if dist.is_initialized():
            n_devices = dist.get_world_size()
            rest = batch_size % n_devices
            batch_size /= n_devices
            if dist.get_rank() == 0:
                batch_size += rest
            batch_size = int(batch_size)

        with self.ema.average_parameters():
            x_mean = self.sample_tensor(batch_size, eps, verbose)

        if dist.is_initialized():
            x_mean = gather_images(x_mean.cpu())

        return self.inverse_scaler(x_mean)

    def snapshot(self) -> None:
        prev_mode = self.model.training
        self.model.eval()

        images = self.sample_images(self.config.training.snapshot_batch_size).cpu()
        nrow = int(math.sqrt(self.config.training.snapshot_batch_size))
        grid = torchvision.utils.make_grid(images, nrow=nrow).permute(1, 2, 0)
        grid = grid.data.numpy().astype(np.uint8)
        self.log_metric('images', 'from_noise', wandb.Image(grid))

        self.model.train(prev_mode)
