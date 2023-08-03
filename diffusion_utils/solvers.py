import numpy as np
import torch


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
        return x, x_mean
