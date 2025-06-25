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


class LinearNoiseScheduler:
    """
    A simple linear noise scheduler that adjusts the noise variance linearly
    over a specified number of steps.
    """

    def __init__(self, steps: int=1000, betas: tuple[float, float]=(1e-4, 0.02), 
                 seed: Optional[int]=None):
        """
        __init__

        Args:
            steps: The number of steps for the scheduler.
            beta: A tuple containing the minimum and maximum beta values.
            seed: Optional seed for reproducibility.
        """
        super(LinearNoiseScheduler, self).__init__()
        assert steps > 0, "steps must be a positive integer"
        
        self.steps = steps

        # Variance schedule parameters
        self.beta_start, self.beta_end = betas
        self.betas = torch.linspace(self.beta_start, self.beta_end, steps)

        # Alphas and cumulative product of alphas for multi-step noise addition
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def single_step(self, x: torch.Tensor, t: Union[int, torch.Tensor]) -> torch.Tensor:
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
        if isinstance(t, torch.Tensor):
            assert t.min() >= 0 and t.max() < self.steps, "t must be in the range [0, steps["
            if t.dim() != 0:
                assert t.shape[0] == x.shape[0], "Batch size of t must match batch size of x"
        elif isinstance(t, int):
            assert 0 <= t < self.steps, "t must be in the range [0, steps["
        else:
            raise TypeError("t must be an int or a torch.Tensor")

        beta = self.betas[t] # (batch_size,) if t is a tensor, else scalar
        noise = torch.randn(x.shape, generator=self.generator) # (batch_size, channels, height, width) or (channels, height, width)
        return torch.sqrt(1 - beta) * x + torch.sqrt(beta) * noise
    
    def multi_steps(self, x: torch.Tensor, t: Union[int, torch.Tensor]) -> torch.Tensor:
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
        if isinstance(t, torch.Tensor):
            assert t.min() >= 0 and t.max() < self.steps, "t must be in the range [0, steps["
            if t.dim() != 0:
                assert t.shape[0] == x.shape[0], "Batch size of t must match batch size of x"
        elif isinstance(t, int):
            assert 0 <= t < self.steps, "t must be in the range [0, steps["
        else:
            raise TypeError("t must be an int or a torch.Tensor")

        alphas_cumprod = self.alphas_cumprod[t]  # (batch_size,) if t is a tensor, else scalar
        noise = torch.randn(x.shape, generator=self.generator) # (batch_size, channels, height, width) or (channels, height, width)
        return torch.sqrt(alphas_cumprod) * x + torch.sqrt(1 - alphas_cumprod) * noise
