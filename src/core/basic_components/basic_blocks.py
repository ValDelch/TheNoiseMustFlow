"""
basic_blocks.py

The module implements several blocks that can be used in different
DL models, like residual blocks, attention blocks, etc.

# TODO: investigate the use of https://docs.pytorch.org/torchtune/0.4/generated/torchtune.modules.RotaryPositionalEmbeddings.html
"""


from __future__ import annotations
from typing import Optional, Union

import torch
from torch import nn


class BasicResidualBlock(nn.Module):
    """
    A basic residual block that consists of two convolutional layers
    with a skip connection and group normalization.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 groups: int = 32, activation: Optional[nn.Module] = None, 
                 padding: Union[int, str] = 'same', use_bias: bool = True, 
                 padding_mode: str = 'zeros', dropout: float = 0.0):
        """
        __init__

        Initializes the BasicResidualBlock with the specified parameters.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Size of the convolutional kernel.
            groups: Number of groups for group normalization.
            activation: Activation function to use. If None, no activation is applied.
            padding: Padding type. Can be 'same', 'valid' or an integer.
            use_bias: Whether to use bias in the convolutional layers.
            padding_mode: Padding mode for the convolutional layers.
            dropout: Dropout rate to apply after the second convolution.
        """
        super(BasicResidualBlock, self).__init__()

        self.group_norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=use_bias, padding_mode=padding_mode)
        
        self.group_norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=use_bias, padding_mode=padding_mode)
        
        self.activation = activation if activation is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout)

        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                             padding=0, bias=use_bias)
        else:
            self.skip_connection = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward

        Applies the residual block to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
        
        residual = x
        
        x = self.group_norm1(x)
        x = self.activation(x)
        x = self.conv1(x)
        
        x = self.group_norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        return self.dropout(x + self.skip_connection(residual))

class BasicAttentionBlock(nn.Module):
    """
    A basic attention block that applies self-attention to the input tensor
    with a skip connection and group normalization.
    """

    def __init__(self, in_channels: int, num_heads: int = 8, groups: int = 32,
                 dropout: float = 0.0):
        """
        __init__

        Initializes the BasicAttentionBlock with the specified parameters.

        Args:
            in_channels: Number of input channels.
            num_heads: Number of attention heads.
            groups: Number of groups for group normalization.
            dropout: Dropout rate to apply after attention.
        """
        super(BasicAttentionBlock, self).__init__()

        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.attention = nn.MultiheadAttention(embed_dim=in_channels, num_heads=num_heads, 
                                               dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward

        Applies the attention block to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
               
        Returns:
            Output tensor of shape (batch_size, in_channels, height, width).
        """
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"

        residual = x

        x = self.group_norm(x)
        
        batch_size, channels, height, width = x.shape
        x = x.view(batch_size, channels, -1).transpose(1, 2)

        x = self.attention(x, x, x, need_weights=False)[0]

        x = x.transpose(1, 2).view(batch_size, channels, height, width)

        return x + residual