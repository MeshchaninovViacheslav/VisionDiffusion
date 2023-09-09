import torch
from abc import ABCMeta, abstractmethod
import numpy as np


class Scheduler(metaclass=ABCMeta):
    @abstractmethod
    def params(self, t):
        pass


class CosineSDE(Scheduler):
    def __init__(self, config):
        self.beta_0 = config.dynamic.beta_min
        self.beta_1 = config.dynamic.beta_max

    def beta_t(self, t):
        return self.beta_0 + (self.beta_1 - self.beta_0) * t

    def params(self, t):
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_gamma_coeff = log_mean_coeff * 2
        alpha = torch.exp(log_mean_coeff)[:, None, None, None].to(t.device)
        std = torch.sqrt(1. - torch.exp(log_gamma_coeff))[:, None, None, None].to(t.device)
        return alpha, std


class CosineIDDPM(Scheduler):
    def __init__(self, config):
        self.s = 0.008

    def f(self, t):
        return torch.cos((t + self.s) / (1 + self.s) * np.pi / 2)

    @property
    def f_0(self):
        return np.cos(self.s / (1 + self.s) * np.pi / 2)

    def params(self, t):
        mu = torch.clip(self.f(t)[:, None, None, None].to(t.device) / self.f_0, 0, 1)
        std = torch.sqrt(1. - mu ** 2)
        return mu, std
