"""
samplers.py

This module provides samplers that are used to sample from the forward
process of a diffusion model. The samplers are designed to control the
noise flow and can be used to generate samples based on the noise schedule
defined in the NoiseScheduler class and the diffusion model.

There are two main types of samplers:
    1. DDPMSampler: A sampler that uses the DDPM (Denoising Diffusion Probabilistic Models)
    2. DDIMSampler: A sampler that uses the DDIM (Denoising Diffusion Implicit Models)

DDIM sampler requires fewer steps than DDPM to generate samples, making it more efficient.
More specifically, DDPM samples with the same number of steps as the noising process,
while DDIM can sample with fewer steps, allowing for faster generation of samples.
"""

from __future__ import annotations
from typing import Union, Callable, Optional
from contextlib import nullcontext

from tqdm import tqdm

import torch

from TheNoiseMustFlow.core.schedulers import NoiseScheduler


class DDPMSampler(torch.nn.Module):
    """
    DDPM Sampler

    This sampler implements the DDPM (Denoising Diffusion Probabilistic Models) sampling
    process. It uses the noise schedule defined in the NoiseScheduler class to sample
    from the forward process.

    Note that step indexes are defined in a monotonic increasing (zero-indexed) in the
    forward process of adding noise, by convention. Thus, the sampling process handles
    indexes in a monotonically decreasing manner.
    """

    def __init__(self, noise_scheduler: NoiseScheduler, use_tqdm: bool = True):
        """
        __init__

        Initializes the DDPM sampler with the specified noise scheduler.

        Args:
            noise_scheduler: An instance of NoiseScheduler that defines the noise schedule.
            use_tqdm: If True, uses tqdm for progress bars during sampling.
                Default is True.
        """
        super(DDPMSampler, self).__init__()
        assert isinstance(noise_scheduler, NoiseScheduler), (
            "noise_scheduler must be an instance of NoiseScheduler"
        )
        self.noise_scheduler = noise_scheduler
        self.use_tqdm = use_tqdm

        self.steps = noise_scheduler.steps
        self.alphas = noise_scheduler.alphas
        self.alphas_cumprod = noise_scheduler.alphas_cumprod
        self.betas = noise_scheduler.betas

        self.generator = noise_scheduler.generator

    def sample_prev_step(
        self, x: torch.Tensor, t: int, pred_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        sample_prev_step

        Samples the previous step in the DDPM process based on
        the current timestep and predicted noise.

        x_{t+1} -> x_{t}

        Args:
            x: The current sample tensor.
                (batch_size, channels, height, width) or (channels, height, width)
            t: The current timestep.
            pred_noise: The predicted noise tensor.
                (batch_size, channels, height, width) or (channels, height, width)

        Returns:
            The sample tensor for the previous step.
                (batch_size, channels, height, width) or (channels, height, width)
        """
        assert x.shape == pred_noise.shape, "x and pred_noise must have the same shape"
        self._validate_xt(x, t)

        if t != 0:
            z = torch.randn(x.shape, generator=self.generator, device=x.device)

            mean = (1.0 / torch.sqrt(self.alphas[t])) * (
                x
                - (self.betas[t] / torch.sqrt(1.0 - self.alphas_cumprod[t]))
                * pred_noise
            )
            var = (
                (1.0 - self.alphas_cumprod[t - 1]) / (1.0 - self.alphas_cumprod[t])
            ) * self.betas[t]
            sigma = torch.sqrt(var)

            return mean + sigma * z
        else:
            return (
                x - torch.sqrt(1.0 - self.alphas_cumprod[t]) * pred_noise
            ) / torch.sqrt(self.alphas_cumprod[t])

    def sample(
        self,
        x: torch.Tensor,
        pred_noise_func: Callable,
        func_inputs: dict = {},
        return_intermediates: bool = False,
        return_step: int = 50,
        training: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        sample

        Samples from x_{t} to x_{0} by executing the reverse process
        with the provided pred_noise_func.

        The current timestep is assumed starting for the beginning (t=steps-1).

        Args:
            x: The current sample tensor.
                (batch_size, channels, height, width) or (channels, height, width)
            pred_noise_func: A callable that takes x and t, and returns the predicted noise.
            func_inputs: Optional additional inputs to the pred_noise_func (other than x and t).
            return_intermediates: If True, returns intermediate samples at each step.
            return_step: The step at which to return the intermediate sample if return_intermediates is True.
                Otherwise, it is ignored.
            training: If True, the sampler is in training mode.

        Returns:
            The final sample tensor after sampling from x_{t} to x_{0}.
        """
        assert callable(pred_noise_func), (
            "pre_noise_func must be a callable that takes x and t"
        )
        self._validate_xt(x, self.steps - 1)

        # Sample from x_{t} to x_{0}
        intermediates = []
        pbar = tqdm(range(self.steps), desc="Sampling", disable=not self.use_tqdm)
        for step in reversed(pbar):  # steps-1, ..., 1, 0
            with torch.no_grad() if not training else nullcontext():
                pred_noise = pred_noise_func(
                    x, torch.tensor([step], device=x.device), **func_inputs
                )
            x = self.sample_prev_step(x, step, pred_noise)
            if return_intermediates and step % return_step == 0:
                intermediates.append(x.clone())

        if return_intermediates:
            return intermediates
        else:
            return x

    def _validate_xt(self, x: torch.Tensor, t: int):
        """
        Validate the step index or tensor of indices.
        """
        assert x.dim() in [3, 4], (
            "x must be a 3D or 4D tensor (channels, height, width) or (batch_size, channels, height, width)"
        )

        if isinstance(t, int):
            assert 0 <= t < self.steps, "t must be in the range [0, steps["
        else:
            raise TypeError("t must be an int")


class DDIMSampler(torch.nn.Module):
    """
    DDIM Sampler

    This sampler implements the DDIM (Denoising Diffusion Implicit Models) sampling
    process. It uses the noise schedule defined in the NoiseScheduler class to sample
    from the forward process, but using fewer steps.

    Note that step indexes are defined in a monotonic increasing (zero-indexed) in the
    forward process of adding noise, by convention. Thus, the sampling process handles
    indexes in a monotonically decreasing manner.
    """

    def __init__(
        self,
        noise_scheduler: NoiseScheduler,
        steps: int = 50,
        eta: float = 0.0,
        use_tqdm: bool = True,
    ):
        """
        __init__

        Initializes the DDIM sampler with the specified noise scheduler.

        Args:
            noise_scheduler: An instance of NoiseScheduler that defines the noise schedule.
            steps: The number of steps to sample. Default is 50.
            eta: Controls stochasticity. Default is 0.0 (deterministic sampling).
                eta=1 approaches DDPM sampling.
            use_tqdm: If True, uses tqdm for progress bars during sampling.
                Default is True.
        """
        super(DDIMSampler, self).__init__()
        assert isinstance(noise_scheduler, NoiseScheduler), (
            "noise_scheduler must be an instance of NoiseScheduler"
        )
        assert noise_scheduler.steps % steps == 0, (
            "steps must be a divisor of noise_scheduler.steps"
        )

        self.noise_scheduler = noise_scheduler
        self.use_tqdm = use_tqdm

        self.steps = steps
        self.steps_list = list(
            range(0, noise_scheduler.steps, noise_scheduler.steps // steps)
        )

        self.alphas_cumprod = noise_scheduler.alphas_cumprod
        self.eta = eta

        self.generator = noise_scheduler.generator

    def sample_prev_step(
        self, x: torch.Tensor, t: int, pred_noise: torch.Tensor
    ) -> torch.Tensor:
        """
        sample_prev_step

        Samples the previous step in the DDIM process based on
        the current timestep and predicted noise.

        x_{t+1} -> x_{t}

        Args:
            x: The current sample tensor.
                (batch_size, channels, height, width) or (channels, height, width)
            t: The current timestep, 0 < t < steps, e.g., 0, 1, ..., 49 (zero-indexed).
            pred_noise: The predicted noise tensor.
                (batch_size, channels, height, width) or (channels, height, width)

        Returns:
            The sample tensor for the previous step.
                (batch_size, channels, height, width) or (channels, height, width)
        """
        assert x.shape == pred_noise.shape, "x and pred_noise must have the same shape"
        self._validate_xt(x, t)

        real_t = self.steps_list[t]
        alphas_cumprod_t = self.alphas_cumprod[real_t]

        # Estimate x_0 from x_t
        x0_pred = (x - torch.sqrt(1.0 - alphas_cumprod_t) * pred_noise) / torch.sqrt(
            alphas_cumprod_t
        )
        x0_pred = x0_pred.clamp(-1.0, 1.0)

        if t != 0:
            real_t_prev = self.steps_list[t - 1]
            alphas_cumprod_t_prev = self.alphas_cumprod[real_t_prev]

            # Compute the direction pointing to x_{t-1}
            sigma_t = self.eta * torch.sqrt(
                (1.0 - alphas_cumprod_t_prev)
                / (1.0 - alphas_cumprod_t)
                * (1.0 - alphas_cumprod_t / alphas_cumprod_t_prev)
            )
            noise = (
                torch.randn(x.shape, generator=self.generator, device=x.device)
                if self.eta > 0
                else 0.0
            )

            return (
                torch.sqrt(alphas_cumprod_t_prev) * x0_pred
                + torch.sqrt(1.0 - alphas_cumprod_t_prev - sigma_t**2) * pred_noise
                + sigma_t * noise
            )
        else:
            return x0_pred

    def sample(
        self,
        x: torch.Tensor,
        pred_noise_func: Callable,
        func_inputs: dict = {},
        return_intermediates: bool = False,
        return_step: int = 1,
        training: bool = False,
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """
        sample

        Samples from x_{t} to x_{0} by executing the reverse process
        with the provided pre_noise_func.

        Args:
            x: The current sample tensor.
                (batch_size, channels, height, width) or (channels, height, width)
            pred_noise_func: A callable that takes x and t, and returns the predicted noise.
            func_inputs: Optional additional inputs to the pred_noise_func (other than x and t).
            return_intermediates: If True, returns intermediate samples at each step.
            return_step: The step at which to return the intermediate sample if return_intermediates is True.
                Otherwise, it is ignored.
            training: If True, the sampler is in training mode.

        Returns:
            The final sample tensor after sampling from x_{t} to x_{0}.
        """
        assert callable(pred_noise_func), (
            "pre_noise_func must be a callable that takes x and t"
        )
        self._validate_xt(x, self.steps - 1)

        # Sample from x_{t} to x_{0}
        intermediates = []
        pbar = tqdm(range(self.steps), desc="Sampling", disable=not self.use_tqdm)
        for step in reversed(pbar):  # steps-1, ..., 1, 0
            with torch.no_grad() if not training else nullcontext():
                pred_noise = pred_noise_func(
                    x,
                    torch.tensor([self.steps_list[step]], device=x.device),
                    **func_inputs,
                )
            x = self.sample_prev_step(x, step, pred_noise)
            if return_intermediates and step % return_step == 0:
                intermediates.append(x.clone())

        if return_intermediates:
            return intermediates
        else:
            return x

    def _validate_xt(self, x: torch.Tensor, t: int):
        """
        Validate the step index or tensor of indices.
        """
        assert x.dim() in [3, 4], (
            "x must be a 3D or 4D tensor (channels, height, width) or (batch_size, channels, height, width)"
        )

        if isinstance(t, int):
            assert 0 <= t < self.steps, "t must be in the range [0, steps["
        else:
            raise TypeError("t must be an int")
