import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Dict
from abc import ABCMeta, abstractmethod


def get_scheduler(config):
    if config.dynamic.scheduler == "cosine":
        from diffusion_utils.schedulers import CosineSDE
        return CosineSDE(config)
    elif config.dynamic.scheduler == "cosine_iddpm":
        from diffusion_utils.schedulers import CosineIDDPM
        return CosineIDDPM(config)
    elif config.dynamic.scheduler == "sd":
        from diffusion_utils.schedulers import SD_sched
        return SD_sched(config)


class DynamicBase(metaclass=ABCMeta):
    def __init__(self, eps=0.001, T=1):
        self.eps_ = eps
        self.T_ = T
    @abstractmethod
    def marginal_params(self, t: Tensor) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def marginal(self, x_0: Tensor, t: Tensor) -> Dict[str, Tensor]:
        pass

    def reverse(self, alpha):
        pass

    @property
    def T(self):
        return self.T_

    @property
    def eps(self):
        return self.eps_

    @staticmethod
    def prior_sampling(shape) -> Tensor:
        return torch.randn(*shape)


class DynamicSDE(DynamicBase):
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(config.dynamic.eps, config.dynamic.T)
        self.N = config.dynamic.N
        self.scheduler = get_scheduler(config)

    def marginal_params(self, t: Tensor) -> Dict[str, Tensor]:
        mu, std = self.scheduler.params(t)
        return {
            "mu": mu,
            "std": std
        }

    def marginal(self, x_0: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        params = self.marginal_params(t)
        mu, std = params["mu"], params["std"]
        noise = torch.randn_like(x_0)
        x_t = x_0 * mu + noise * std
        return {
            "x_t": x_t,
            "noise": noise,
            "mu": mu,
            "std": std,
        }

    def reverse_params(self, model, x_t, t, score_fn, ode_sampling=False):
        beta_t = self.scheduler.beta_t(t)
        drift_sde = (-1) / 2 * beta_t[:, None, None, None] * x_t
        diffuson_sde = torch.sqrt(beta_t)

        if ode_sampling:
            drift = drift_sde - (1 / 2) * beta_t[:, None, None, None] * score_fn(model=model, x_t=x_t, t=t)['score']
            diffusion = 0
        else:
            drift = drift_sde - beta_t[:, None, None, None] * score_fn(model, x_t, t)['score']
            diffusion = diffuson_sde
        return drift, diffusion


class DynamicBoot(DynamicBase):
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        super().__init__(config.dynamic.eps, config.dynamic.T)
        self.N = config.dynamic.N
        self.step_size = config.dynamic.step_size
        self.scheduler = get_scheduler(config)

    def marginal_params(self, t: Tensor) -> Dict[str, Tensor]:
        mu, std = self.scheduler.params(t)
        return {
            "mu": mu,
            "std": std
        }

    def marginal(self, x_0: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        params = self.marginal_params(t)
        mu, std = params["mu"], params["std"]
        noise = torch.randn_like(x_0)
        x_t = x_0 * mu + noise * std
        return {
            "x_t": x_t,
            "noise": noise,
            "mu": mu,
            "std": std,
        }

    def reverse_params(self, x_t, t, score_fn, ode_sampling=False):
        beta_t = self.scheduler.beta_t(t)
        drift_sde = (-1) / 2 * beta_t[:, None, None, None] * x_t
        diffuson_sde = torch.sqrt(beta_t)

        if ode_sampling:
            drift = drift_sde - (1 / 2) * beta_t[:, None, None, None] * score_fn(x_t=x_t, t=t)['score']
            diffusion = 0
        else:
            drift = drift_sde - beta_t[:, None, None, None] * score_fn(x_t=x_t, t=t)['score']
            diffusion = diffuson_sde
        return drift, diffusion
