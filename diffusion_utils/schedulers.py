import torch
from abc import ABCMeta, abstractmethod


class Scheduler(metaclass=ABCMeta):
    @abstractmethod
    def beta_t(self, t):
        pass

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
