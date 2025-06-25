"""
schedule.py

This module provides noise schedulers which are used to control the
noise flow in the forward process.

The main distinction between schedulers is how the noise variance
is adjusted over time. Some schedulers use a linear schedule, while
others use a cosine schedule, or even some trainable parameters.
"""


from __future__ import annotations
from typing import Union, Optional

import torch


# Type alias for tensor or integer
TensorOrInt = Union[torch.Tensor, int]


class NoiseScheduler:
    """
    A non-trainable noise scheduler that adjusts the noise
    variance over a specified number of steps.
    """

    def __init__(self, steps: int=1000, betas: tuple[float, float]=(1e-4, 0.02), 
                 schedule: str='linear', seed: Optional[int]=None):
        """
        __init__

        Initializes the noise scheduler with the specified parameters.
        Note that the betas parameter is ignored for the cosine schedule.

        Args:
            steps: The number of steps for the scheduler.
            betas: A tuple containing the minimum and maximum beta values.
            schedule: The type of noise schedule to use. Available options are
                      'linear', 'cosine', 'quadratic', 'sigmoid' and 'geometric'.
            seed: Optional seed for reproducibility.
        """
        super(NoiseScheduler, self).__init__()
        assert steps > 0, "steps must be a positive integer"
        assert schedule in ['linear', 'cosine', 'quadratic', 'sigmoid', 'geometric'], \
            "schedule must be one of 'linear', 'cosine', 'quadratic', 'sigmoid', 'geometric'"
        
        if schedule == 'cosine' and betas != (1e-4, 0.02):
            print("[Warning] For cosine schedule, the betas parameter is ignored.")

        self.steps = steps
        self.schedule = schedule

        # Variance schedule parameters
        if schedule != 'cosine':
            self.betas = getattr(self, f"_{schedule}_betas")(betas[0], betas[1])

            # Compute the cumulative product of alphas for cumulative noise addition
            alphas = 1 - self.betas
            self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        else:
            self.betas, self.alphas_cumprod = self._cosine_betas()

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def add_noise_step(self, x: torch.Tensor, t: TensorOrInt) -> torch.Tensor:
        """
        Add a single step of noise to the input tensor.

        Args:
            x: The input tensor to which noise will be added.
               (batch_size, channels, height, width) or (channels, height, width)
            t: The step index or tensor of indices. 
               (0 <= t < steps), int or tensor of shape (batch_size,)

        Returns:
            The input tensor with added noise.
        """
        self._validate_t(x, t)

        beta = self.betas[t] # (batch_size,) if t is a tensor, else scalar
        noise = torch.randn(x.shape, generator=self.generator) # (batch_size, channels, height, width) or (channels, height, width)
        return torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
    
    def add_noise_cumulative(self, x: torch.Tensor, t: TensorOrInt) -> torch.Tensor:
        """
        Add noise to the input tensor for multiple steps at once.

        Args:
            x: The input tensor to which noise will be added.
               (batch_size, channels, height, width) or (channels, height, width)
            t: The step index or tensor of indices. 
               (0 <= t < steps), int or tensor of shape (batch_size,)

        Returns:
            The input tensor with added noise.
        """
        self._validate_t(x, t)

        alphas_cumprod = self.alphas_cumprod[t]  # (batch_size,) if t is a tensor, else scalar
        noise = torch.randn(x.shape, generator=self.generator) # (batch_size, channels, height, width) or (channels, height, width)
        return torch.sqrt(alphas_cumprod) * x + torch.sqrt(1 - alphas_cumprod) * noise
    
    def _linear_betas(self, beta_start, beta_end) -> torch.Tensor:
        """
        Generate linear betas for the noise schedule.
        """
        return torch.linspace(beta_start, beta_end, self.steps)
    
    def _cosine_betas(self, s: float=0.008) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate cosine betas for the noise schedule.
        """
        t = torch.linspace(0, self.steps, self.steps + 1) / self.steps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * (torch.pi / 2)) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, min=0, max=0.999), alphas_cumprod

    def _quadratic_betas(self, beta_start, beta_end) -> torch.Tensor:
        """
        Generate quadratic betas for the noise schedule.
        """
        t = torch.linspace(0, 1, self.steps)
        return beta_start + (beta_end - beta_start) * t ** 2

    def _sigmoid_betas(self, beta_start, beta_end) -> torch.Tensor:
        """
        Generate sigmoid betas for the noise schedule.
        """
        sigmoid = torch.sigmoid(torch.linspace(-6, 6, self.steps))
        return beta_start + (beta_end - beta_start) * sigmoid

    def _geometric_betas(self, beta_start, beta_end) -> torch.Tensor:
        """
        Generate geometric betas for the noise schedule.
        """
        return beta_start * (beta_end / beta_start) ** (torch.arange(self.steps) / (self.steps - 1))
    
    def _validate_t(self, x: torch.Tensor, t: TensorOrInt):
        """
        Validate the step index or tensor of indices.
        """
        if isinstance(t, torch.Tensor):
            assert t.min() >= 0 and t.max() < self.steps, "t must be in the range [0, steps["
            if t.dim() != 0:
                assert t.shape[0] == x.shape[0], "Batch size of t must match batch size of x"
        elif isinstance(t, int):
            assert 0 <= t < self.steps, "t must be in the range [0, steps["
        else:
            raise TypeError("t must be an int or a torch.Tensor")
