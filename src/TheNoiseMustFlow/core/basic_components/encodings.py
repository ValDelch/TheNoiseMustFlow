"""
encodings.py

This module implements several positional encoding methods.
"""


from __future__ import annotations
from typing import Union

import torch
from torch import nn


# Type alias for tensor or integer
TensorOrInt = Union[torch.Tensor, int]


class TimeEncoding(nn.Module):
    """
    A positional encoding module that encodes time steps into a
    higher-dimensional space using sine and cosine functions.
    The encoding is afterwards expanded using two linear layers
    to increase its dimensionality and build a richer representation.

    This encoding is useful to incorporate information about the
    current time step into the diffusion model.
    """

    def __init__(self, dim: int = 1280):
        """
        __init__

        Initializes the TimeEncoding module.

        Args:
            dim: The dimension of the output encoding. Must be divisible by 8.
        """
        super(TimeEncoding, self).__init__()
        assert dim % 8 == 0, "dim must be divisible by 8"

        self.dim = dim

        self.linear1 = nn.Linear(dim//4, dim, bias=True)
        self.linear2 = nn.Linear(dim, dim, bias=True)
        self.silu = nn.SiLU()

        self.register_buffer(
            'freqs',
            torch.pow(
                float(1e4),
                -torch.arange(
                    start = 0, end = dim // 8, dtype = torch.float32
                ) / (dim // 8)
            )
        )

    def get_time_encoding(self, t: TensorOrInt) -> torch.Tensor:
        """
        get_time_encoding

        Computes the time encoding for a given time step.

        Args:
            t: time step. Can be a tensor of shape (batch_size,) or a single integer.

        Returns:
            A tensor of shape (batch_size, 4*dim) containing the time encoding.
                Note: The returned tensor is always 2D, even for a single integer input (shape will be (1, 4*dim)).
        """
        device = self.freqs.device

        if isinstance(t, int):
            assert t >= 0, "t must be a non-negative integer"
            t = torch.tensor(t, dtype=torch.float32, device=device).unsqueeze(0)
        elif isinstance(t, torch.Tensor):
            assert t.dim() == 1, "t must be a 1D tensor"
            assert t.min() >= 0, "t must contain non-negative integers"
            t = t[:, None].float()

        x = t * self.freqs[None, :]

        return torch.cat([torch.cos(x), torch.sin(x)], dim=-1)        
        
    def forward(self, t: TensorOrInt) -> torch.Tensor:
        """
        forward

        Compute the encoding of timestep(s) and expand it to a 
        higher-dimensional space using linear layers.

        Args:
            t: time step. Can be a tensor of shape (batch_size,) or a single integer.
        """
        x = self.linear1(self.get_time_encoding(t))
        x = self.silu(x)
        x = self.linear2(x)

        return x