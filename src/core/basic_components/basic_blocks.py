"""
basic_blocks.py

The module implements several blocks that can be used in different
DL models, like residual blocks, attention blocks, etc.

# TODO: investigate the use of https://docs.pytorch.org/torchtune/0.4/generated/torchtune.modules.RotaryPositionalEmbeddings.html
"""


from __future__ import annotations
from typing import Optional, Union, Type
import warnings

import torch
from torch import nn

from core.basic_components.functional_blocks import LayerNorm, SelfAttention, CrossAttention, GEGLU


class BasicResidualBlock(nn.Module):
    """
    A basic residual block that consists of two convolutional layers
    with a skip connection and group normalization.

    This block can also apply context conditioning if a conditioning dimension is provided.
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 groups: int = 32, activation: Optional[Type[nn.Module]] = None, 
                 padding: Union[int, str] = 'same', use_bias: bool = True, 
                 padding_mode: str = 'zeros', dropout: float = 0.0, 
                 d_context: Optional[int] = None):
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
            d_context: Optional conditioning dimension. If provided, the block can use it for conditioning.
        """
        super(BasicResidualBlock, self).__init__()

        self.group_norm1 = nn.GroupNorm(num_groups=groups, num_channels=in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=use_bias, padding_mode=padding_mode)
        
        if d_context is not None:
            self.context_proj = nn.Linear(d_context, out_channels)
        
        self.group_norm2 = nn.GroupNorm(num_groups=groups, num_channels=out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, 
                               padding=padding, bias=use_bias, padding_mode=padding_mode)
        
        self.activation = activation if activation is not None else nn.Identity()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Skip connection
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                                             padding=0, bias=use_bias)
        else:
            self.skip_connection = nn.Identity()

    def apply_context(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        apply_context

        Applies the context conditioning to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, out_channels, height, width).
            context: Conditioning tensor of shape (batch_size, d_context).

        Returns:
            Tensor with context applied.
        """
        assert context.dim() == 2, "Conditioning tensor must be 2D (batch_size, d_context)"
        assert hasattr(self, 'context_proj'), "You must define a d_context to use conditioning projection"

        context = self.context_proj(context)
        return x + context.unsqueeze(-1).unsqueeze(-1)

    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward

        Applies the residual block to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            context: Optional context tensor of shape (batch_size, d_context).

        Returns:
            Output tensor of shape (batch_size, out_channels, height, width).
        """
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
        if context is None and hasattr(self, 'context_proj'):
            warnings.warn(
                f"[{self.__class__.__name__}] Context tensor is None, but a d_context was previously defined."
                "Skipping context attention.", 
                UserWarning, stacklevel=2
            )
        
        residual = x
        
        x = self.group_norm1(x)
        x = self.activation(x)
        x = self.conv1(x)

        if context is not None:
            x = self.apply_context(x, context)
        
        x = self.group_norm2(x)
        x = self.activation(x)
        x = self.conv2(x)
        
        return self.dropout(x + self.skip_connection(residual))

class BasicAttentionBlock(nn.Module):
    """
    A basic attention block that applies self-attention to the input tensor
    with a skip connection and group normalization.

    This block can also apply cross-attention if a conditioning dimension is provided.
    It uses GEGLU activation by default, but can accept any activation function.
    """

    def __init__(self, channels: int, num_heads: int = 8, groups: int = 32, 
                 activation: Optional[nn.Module] = None, use_bias: bool = True,
                 dropout: float = 0.0, d_context: Optional[int] = None):
        """
        __init__

        Initializes the BasicAttentionBlock with the specified parameters.

        Args:
            channels: Number of input channels.
            num_heads: Number of attention heads.
            groups: Number of groups for group normalization.
            activation: Activation function to use. If None, geglu is used.
            use_bias: Whether to use bias in the convolutional layers.
            dropout: Dropout rate to apply after attention.
            d_context: Optional conditioning dimension. If provided, the block can use it for conditioning.
        """
        super(BasicAttentionBlock, self).__init__()
        assert channels % groups == 0, "Channels must be divisible by groups for GroupNorm"
        assert channels % num_heads == 0, "Channels must be divisible by num_heads for SelfAttention"

        # Input block
        self.group_norm = nn.GroupNorm(num_groups=groups, num_channels=channels)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=use_bias)

        # Self-attention block
        self.layer_norm_1 = LayerNorm(channels)
        self.attention_1 = SelfAttention(channels, num_heads=num_heads, dropout=dropout)

        # Optional context cross-attention block
        if d_context is not None:
            self.layer_norm_context = LayerNorm(channels)
            self.attention_context = CrossAttention(
                channels, d_context, num_heads=num_heads, dropout=dropout
            )

        # Activation block
        self.layer_norm_2 = LayerNorm(channels)
        if activation is None:
            self.activation = GEGLU(in_dim=channels, inter_dim=4 * channels, bias=use_bias)
        else:
            self.activation = activation
        
        # Output block
        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0, bias=use_bias)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def apply_context(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        apply_context

        Applies the context conditioning to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            context: Context tensor of shape (batch_size, seq_len, d_context).

        Returns:
            Tensor with context applied.
        """
        assert context.dim() == 3, "Context tensor must be 2D (batch_size, seq_len, d_context)"
        assert hasattr(self, 'attention_context'), "Context attention is not defined in this block"

        # Applying the optional cross-attention
        short_term_residual = x

        x = self.layer_norm_context(x)
        x = self.attention_context(x, context, causal_mask=False, return_attn=False, key_padding_mask=None)

        return x + short_term_residual
    
    def forward(self, x: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        forward

        Applies the attention block to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).
            context: Optional context tensor of shape (batch_size, seq_len, d_context).

        Returns:
            Output tensor of shape (batch_size, channels, height, width).
        """
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, channels, height, width)"
        if context is None and hasattr(self, 'attention_context'):
            warnings.warn(
                f"[{self.__class__.__name__}] Context tensor is None, but a d_context was previously defined."
                "Skipping context attention.", 
                UserWarning, stacklevel=2
            )

        # Input block
        long_term_residual = x

        x = self.group_norm(x)
        x = self.conv_input(x)

        n, c, h, w = x.shape
        x = x.view(n, c, h * w).permute(0, 2, 1)  # (batch_size, height * width, channels)

        # Self-attention block
        short_term_residual = x

        x = self.layer_norm_1(x)
        x = self.attention_1(x)
        x += short_term_residual

        # Optional context cross-attention block
        if context is not None:
            x = self.apply_context(x, context)

        # Activation block
        short_term_residual = x

        x = self.layer_norm_2(x)
        x = self.activation(x)

        x += short_term_residual

        # Reshape back to (batch_size, channels, height, width)
        x = x.transpose(-1, -2).view(n, c, h, w)

        return self.dropout(self.conv_output(x) + long_term_residual)
    
class BasicFeedForwardBlock(nn.Module):
    """
    A basic feed-forward block that applies two linear transformations.
    """

    def __init__(self, dim: int, d_ff: int, out_features: Optional[int] = None,
                 activation: nn.Module = nn.GELU(), layer_norm: bool = True, 
                 dropout: float = 0.0):
        """
        __init__

        Initializes the BasicFeedForwardBlock with the specified parameters.

        Args:
            dim: Dimensionality of the input tensor.
            d_ff: Dimensionality of the feed-forward layer.
            out_features: Dimensionality of the output tensor. If None, defaults to dim.
            activation: Activation function to apply after the first linear layer.
            layer_norm: Whether to apply group normalization after the activation.
            dropout: Optional dropout rate
        """
        super(BasicFeedForwardBlock, self).__init__()

        self.linear1 = nn.Linear(dim, d_ff, bias=True)
        self.linear2 = nn.Linear(d_ff, out_features if out_features is not None else dim, bias=True)

        self.activation = activation
        self.layer_norm = LayerNorm(d_ff) if layer_norm else nn.Identity()

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward

        Applies the feed-forward block to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, ..., dim).

        Returns:
            Output tensor of shape (batch_size, ..., out_features).
        """
        assert x.dim() >= 2, "Input tensor must be at least 2D (batch_size, ..., dim)"
        
        h = self.linear1(x)
        h = self.activation(h)
        h = self.layer_norm(h)
        out = self.linear2(h)
        
        return self.dropout(out)
