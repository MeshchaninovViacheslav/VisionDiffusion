import torch
import numpy as np
from abc import ABCMeta, abstractmethod


class Scheduler(metaclass=ABCMeta):
    def beta_t(self, t):
        pass

    @abstractmethod
    def params(self, t):
        pass


class CosineSDE(Scheduler):
    def __init__(self, config):
        self.beta_0 = config.sde.beta_min
        self.beta_1 = config.sde.beta_max

    def beta_t(self, t):
        return self.beta_0 + (self.beta_1 - self.beta_0) * t

    def params(self, t):
        """

        :param t:
        :return: mu, std
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_gamma_coeff = log_mean_coeff * 2
        mu = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(log_gamma_coeff))
        return mu, std
