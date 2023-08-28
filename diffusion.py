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
from functools import partial

from models.utils import create_model
from data.FFHQ_dataset import DataGenerator
from utils.utils import gather_images
from diffusion_utils.dynamic import DynamicBoot
from diffusion_utils.solvers import create_solver

from ml_collections import ConfigDict
from typing import Optional, Union, Dict, Tuple
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
        self.load_model_initialization()

        self.config.total_number_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.device = f"cuda:{dist.get_rank()}" if dist.is_initialized() else "cuda:0"
        self.model.to(self.device)
        self.model_without_ddp = self.model
        self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
        self.inverse_scaler = lambda x: torch.clip((x + 1.) / 2., 0., 1.) * 255
        self.project_name = config.project_name
        self.experiment_name = config.experiment_name

        self.teacher_model = create_model(config=config)
        self.load_teacher_model()
        self.teacher_model.cuda()

        self.dynamic = DynamicBoot(config=config)

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
        if dist.get_rank() == 0:
            print(f"Checkpoint loaded {checkpoint_name}")
        return True

    def load_teacher_model(self) -> None:
        teacher_ema = ExponentialMovingAverage(self.teacher_model.parameters(), self.config.model.ema_rate)
        load = torch.load(self.config.teacher_checkpoint_name, map_location="cpu")
        teacher_ema.load_state_dict(load["ema"])
        teacher_ema.copy_to()
        if dist.get_rank() == 0:
            print(f"Teacher checkpoint loaded {self.config.teacher_checkpoint_name}")

    def load_model_initialization(self) -> None:
        init_ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        load = torch.load(self.config.init_checkpoint_name, map_location="cpu")
        init_ema.load_state_dict(load["ema"])
        init_ema.copy_to()
        if dist.get_rank() == 0:
            print(f"Initialization checkpoint loaded {self.config.init_checkpoint_name}")

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
        return torch.rand(batch_size) * (self.dynamic.T - self.dynamic.eps) + self.dynamic.eps

    def calc_score(self, model, x_t, t) -> Dict[str, torch.Tensor]:
        input_t = t * 999  # just technic for training, SDE looks the same
        params = self.dynamic.marginal_params(t)
        mu, std = params["mu"], params["std"]

        x_0 = model(x_t, input_t)
        eps_theta = (x_t - mu * x_0) / std
        score = -eps_theta / std

        return {
            "score": score,
            "eps_theta": eps_theta,
            "x_0": x_0,
        }

    def get_stat(self, x: torch.Tensor) -> Dict[str, float]:
        stat_dict = {}
        stat_dict["mean"] = torch.mean(x).item()
        stat_dict["std"] = torch.std(x).item()
        dim = x.shape[1] * x.shape[2] * x.shape[3]
        stat_dict["norm"] = torch.mean(
            torch.sqrt(torch.sum(x ** 2, dim=(1, 2, 3)) / dim)
        ).item()
        return stat_dict

    def calc_loss(self, clean_x: torch.Tensor = None, eps: float = 1e-5) -> Tuple[
        Dict[str, torch.Tensor], Dict[str, Dict[str, float]]]:
        batch_size = self.config.training.batch_size_per_gpu
        shape = (
            batch_size,
            self.config.data.num_channels,
            self.config.data.image_size,
            self.config.data.image_size
        )
        noise = torch.randn(shape).cuda()
        time_t = self.sample_time(batch_size, eps=eps).cuda()
        time_s = torch.clip(time_t - self.dynamic.step_size, min=self.dynamic.eps)
        time_t_max = torch.ones_like(time_t) * self.dynamic.T
        time_t_min = torch.ones_like(time_t) * self.dynamic.eps

        marg_params_t = self.dynamic.marginal_params(time_t)
        marg_params_s = self.dynamic.marginal_params(time_s)

        lambda_der_t = 1 - (marg_params_t["mu"] * marg_params_s["std"]) / (marg_params_s["mu"] * marg_params_t["std"])

        # Target prediction
        with torch.no_grad(), torch.cuda.amp.autocast(dtype=self.config.training.num_type):
            y_t = self.model(noise, time_t).detach()

            if self.config.solver_type == "ddim":
                x_t = marg_params_t["mu"] * y_t + marg_params_t["std"] * noise
                f_t = self.teacher_model(x_t, time_t)
                y_target = y_t + lambda_der_t * (f_t - y_t)
            elif self.config.solver_type == "heun":
                x_t = marg_params_t["mu"] * y_t + marg_params_t["std"] * noise
                f_t = self.teacher_model(x_t, time_t)

                y_s = y_t + lambda_der_t * (f_t - y_t)
                x_s = marg_params_s["mu"] * y_s + marg_params_s["std"] * noise
                f_s = self.teacher_model(x_s, time_s)

                y_target = y_t + lambda_der_t / 2 * (f_t - y_t + f_s - y_s)

            if self.config.clip_target:
                y_target = torch.clip(y_target, min=-1, max=1)

        # Model prediction
        with torch.cuda.amp.autocast(dtype=self.config.training.num_type):
            y_pred = self.model(noise, time_s)

        loss_recon = torch.mean(torch.square(y_pred - y_target))  # / (self.dynamic.step_size ** 2)
        loss_bc = torch.tensor([0.]).cuda()

        stat_dict = {
            "y_target": self.get_stat(y_target),
            "y_pred": self.get_stat(y_pred),
            "y_t": self.get_stat(y_t),
        }

        if self.step % self.config.loss_bc_freq == 0:
            with torch.cuda.amp.autocast(dtype=self.config.training.num_type):
                y_t_max = self.model(noise, time_t_max)
                with torch.no_grad():
                    f_t_max = self.teacher_model(noise, time_t_max).detach()
                    if self.config.clip_target:
                        f_t_max = torch.clip(f_t_max, min=-1, max=1)

            loss_bc = torch.mean(torch.square(y_t_max - f_t_max)) * self.config.loss_bc_beta

            stat_dict["f_t_max"] = self.get_stat(f_t_max)
            stat_dict["y_t_max"] = self.get_stat(y_t_max)

        loss = loss_recon + loss_bc

        loss_dict = {
            'loss': loss,
            'total_loss': loss,
            "reconstruction loss": loss_recon,
            "boundary condition": loss_bc,
        }

        return loss_dict, stat_dict

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

                loss_dict, _ = self.calc_loss(clean_x=X)
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

            # (X, y) = next(self.train_gen)
            # X = X.to(self.device)

            loss_dict, stat_dict = self.calc_loss()
            if iter_idx % self.config.training.log_freq == 0:
                for k, v in loss_dict.items():
                    self.log_metric(k, 'train', v.item())
                for key1 in stat_dict:
                    for key2, v in stat_dict[key1].items():
                        self.log_metric(key1, key2, v)

            self.optimizer_step(loss_dict['total_loss'])

            if iter_idx % self.config.training.snapshot_freq == 0:
                self.snapshot()
                self.model.train()

            if iter_idx % self.config.training.eval_freq == 0:
                self.validate()
                self.model.train()

            if iter_idx % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()
                self.model.train()

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
        self.model.eval()

        with torch.no_grad():
            noise = self.dynamic.prior_sampling(shape=shape).to(device)
            input_t = self.dynamic.eps * torch.ones(shape[0], device=device)
            x = self.model(noise, input_t)

        return x

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
