import numpy as np
import torch


def create_solver(config):
    if config.dynamic.solver == "euler":
        return EulerDiffEqSolver
    elif config.dynamic.solver == "ddim":
        return DDIMSolver


class EulerDiffEqSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def step(self, x_t, t, labels=None):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        dt = -1 / self.dynamic.N
        noise = torch.randn_like(x_t)
        drift, diffusion = self.dynamic.reverse_params(x_t, t, self.score_fn, self.ode_sampling)
        x_mean = x_t + drift * dt
        x = x_mean + diffusion.view(-1, 1, 1, 1) * np.sqrt(-dt) * noise
        return {
            "x": x,
            "x_mean": x_mean
        }


class DDIMSolver:
    def __init__(self, dynamic, score_fn, ode_sampling=False):
        self.dynamic = dynamic
        self.score_fn = score_fn
        self.ode_sampling = ode_sampling

    def q_x_t_reverse(self, x_t, x_0, t):
        dt = 1 / self.dynamic.N

        alpha_t = self.dynamic.marginal_params(t)["mu"] ** 2
        alpha_t_1 = self.dynamic.marginal_params(t - dt)["mu"] ** 2

        sigma_t = torch.zeros_like(alpha_t)

        noise_t = (x_t - torch.sqrt(alpha_t) * x_0) / torch.sqrt(1 - alpha_t)
        mu = torch.sqrt(alpha_t_1) * x_0 + \
             torch.sqrt(1 - alpha_t_1 - sigma_t ** 2) * noise_t
        std = sigma_t
        return mu, std

    def step(self, x_t, t, labels=None):
        """
        Implement reverse SDE/ODE Euler solver
        """

        """
        x_mean = deterministic part
        x = x_mean + noise (yet another noise sampling)
        """
        noise = torch.randn_like(x_t)
        x_0 = self.score_fn(x_t, t)["x_0"]
        mu, std = self.q_x_t_reverse(x_t, x_0, t)
        x = mu + std * noise
        return {
            "x": x,
            "x_mean": mu
        }
