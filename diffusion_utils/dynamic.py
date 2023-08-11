import torch
import numpy as np
from torch import Tensor
from typing import Tuple, Dict
from abc import ABCMeta, abstractmethod


def get_scheduler(config):
    if config.sde.scheduler == "cosine_sde":
        from diffusion_utils.schedulers import CosineSDE
        return CosineSDE(config)
    if config.sde.scheduler == "ddm":
        from diffusion_utils.schedulers import DDM_Scheduler
        return DDM_Scheduler(config)


class DynamicBase(metaclass=ABCMeta):
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
        return 1

    @staticmethod
    def prior_sampling(self, shape):
        return torch.randn(*shape)


class DynamicSDE(DynamicBase):
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """

        self.N = config.sde.N
        self.scheduler = get_scheduler(config)
        self.predict = config.predict

    def marginal_params(self, t: Tensor):
        mu, std = self.scheduler.params(t)
        return mu[:, None, None, None], std[:, None, None, None]

    def marginal(self, x_0: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        mu, std = self.marginal_params(t)
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
            drift = drift_sde - (1 / 2) * beta_t[:, None, None, None] * score_fn(x_t, t)['score']
            diffusion = 0
        else:
            drift = drift_sde - beta_t[:, None, None, None] * score_fn(x_t, t)['score']
            diffusion = diffuson_sde
        return drift, diffusion

class DynamicDDM(DynamicBase):
    def __init__(self, config):
        """Construct a Variance Preserving SDE.

        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """

        self.N = config.sde.N
        self.scheduler = get_scheduler(config)
        self.predict = config.predict

    def marginal_params(self, t: Tensor):
        mu, std = self.scheduler.params(t)
        return mu[:, None, None, None], std[:, None, None, None]

    def marginal(self, x_0: Tensor, t: Tensor) -> Dict[str, Tensor]:
        """
        Calculate marginal q(x_t|x_0)'s mean and std
        """
        mu, std = self.marginal_params(t)
        noise = torch.randn_like(x_0)
        x_t = x_0 * mu + noise * std
        return {
            "x_t": x_t,
            "noise": noise,
            "mu": mu,
            "std": std,
            "x_0": x_0,
        }