"""
functional_blocks.py

The module implements several functional blocks that can be used in different
DL models, like attention mechanisms, layer normalization, custom activations, etc.

# TODO: implement a fast attention mechanism variant, e.g. using Flash Attention or similar
"""


from __future__ import annotations
from typing import Union, Optional
import warnings

import torch
from torch import nn
from torch.nn import functional as F


class SelfAttention(nn.Module):
    """
    A basic implementation of self-attention mechanism.
    """

    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0):
        """
        __init__

        Initializes the SelfAttention with the specified parameters.

        Args:
            dim: Dimensionality of the input tensor.
            num_heads: Number of attention heads.
            dropout: Dropout rate to apply after attention.
        """
        super(SelfAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # 3 * dim because we need to project input to query, key, value
        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, causal_mask: bool = False, return_attn: bool = False, 
                key_padding_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        forward

        Applies self-attention to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            causal_mask: If True, applies a causal mask to the attention scores.
            return_attn: If True, returns the attention weights along with the output.
            key_padding_mask: Optional mask to ignore certain positions in the input.
                (batch_size, seq_len) shape with True for positions to ignore.

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        assert x.dim() == 3, "Input tensor must be 3D (batch_size, seq_len, dim)"
        assert x.shape[2] == self.dim, f"Input tensor must have last dimension of size {self.dim}, got {x.shape[2]}"

        batch_size, seq_len, _ = x.shape
        interm_shape = (batch_size, seq_len, self.num_heads, self.head_dim)

        # Project input to query, key, value
        q, k, v = self.qkv_proj(x).chunk(3, dim=-1) # (batch_size, seq_len, 3 * dim)

        q = q.reshape(*interm_shape).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.reshape(*interm_shape).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.reshape(*interm_shape).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (batch_size, num_heads, seq_len, seq_len)

        # Apply masks if provided
        if causal_mask or key_padding_mask is not None:
            full_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
            # full_mask: True where masking is needed (either causal or padding)

            if causal_mask:
                causal = torch.triu(
                    torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
                )
                full_mask |= causal.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, seq_len) for broadcasting

            if key_padding_mask is not None:
                assert key_padding_mask.dim() == 2 and key_padding_mask.shape == (batch_size, seq_len), \
                    "key_padding_mask must be 2D with shape (batch_size, seq_len)"
                padding = key_padding_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, seq_len) for broadcasting
                full_mask |= padding

            mask_value = -1e9 if attn_scores.dtype == torch.float16 else float('-inf') # Better for mixed precision support
            attn_scores = attn_scores.masked_fill(full_mask, mask_value)

        # Apply softmax to get attention weights
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values # For numerical stability
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1)) # (batch_size, num_heads, seq_len, seq_len)

        output = torch.matmul(attn_weights, v) # (batch_size, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)

        if return_attn:
            return self.out_proj(output), attn_weights.to(dtype=torch.float32)
        else:
            return self.out_proj(output) # (batch_size, seq_len, dim)
        
class CrossAttention(nn.Module):
    """
    A basic implementation of cross-attention mechanism.
    """

    def __init__(self, dim: int, cross_dim: int, num_heads: int = 8, dropout: float = 0.0):
        """
        __init__

        Initializes the CrossAttention with the specified parameters.

        Args:
            dim: Dimensionality of the input tensor.
            cross_dim: Dimensionality of the cross-attention input.
            num_heads: Number of attention heads.
            dropout: Dropout rate to apply after attention.
        """
        super(CrossAttention, self).__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        assert cross_dim % num_heads == 0, "cross_dim must be divisible by num_heads"

        self.dim = dim
        self.cross_dim = cross_dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.k_proj = nn.Linear(cross_dim, dim, bias=True)
        self.v_proj = nn.Linear(cross_dim, dim, bias=True)
        self.out_proj = nn.Linear(dim, dim, bias=True)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, cross_x: torch.Tensor, 
                causal_mask: bool = False, return_attn: bool = False, 
                key_padding_mask: Optional[torch.Tensor] = None) -> Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        forward

        Applies cross-attention to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, seq_len, dim).
            cross_x: Cross-attention input tensor of shape (batch_size, cross_seq_len, cross_dim).
            causal_mask: If True, applies a causal mask to the attention scores.
                Note: causal_mask if generally not used in cross-attention, because it assumes
                a self-attention is previously applied. However, causal_mask is supported
                for consistency with the SelfAttention interface. If used, it requires
                seq_len and cross_seq_len to be equal.
            return_attn: If True, returns the attention weights along with the output.
            key_padding_mask: Optional mask to ignore certain positions in the input.
                (batch_size, cross_seq_len) shape with True for positions to ignore.

        Returns:
            Output tensor of shape (batch_size, seq_len, dim).
        """
        assert x.dim() == 3 and cross_x.dim() == 3, "Input tensors must be 3D (batch_size, seq_len, dim)"
        assert x.shape[2] == self.dim and cross_x.shape[2] == self.cross_dim, \
            f"Input tensors must have last dimensions of size {self.dim} and {self.cross_dim}, got {x.shape[2]} and {cross_x.shape[2]}"

        batch_size, seq_len, _ = x.shape
        _, cross_seq_len, _ = cross_x.shape

        interm_shape = (batch_size, -1, self.num_heads, self.head_dim)

        # Project inputs to query, key, value
        q = self.q_proj(x)
        k = self.k_proj(cross_x)
        v = self.v_proj(cross_x)

        q = q.reshape(*interm_shape).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.reshape(*interm_shape).transpose(1, 2)  # (batch_size, num_heads, cross_seq_len, head_dim)
        v = v.reshape(*interm_shape).transpose(1, 2)  # (batch_size, num_heads, cross_seq_len, head_dim)

        # Calculate attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5) # (batch_size, num_heads, seq_len, cross_seq_len)

        # Apply masks if provided
        if causal_mask or key_padding_mask is not None:
            full_mask = torch.zeros_like(attn_scores, dtype=torch.bool)
            # full_mask: True where masking is needed (either causal or padding)

            if causal_mask:
                assert seq_len == cross_seq_len, "Causal mask requires seq_len and cross_seq_len to be equal"
                warnings.warn(
                    "Using causal_mask=True in CrossAttention is unusual. "
                    "Causal masking is typically only used in self-attention, not cross-attention.",
                    category=UserWarning,
                    stacklevel=2,
                )

                causal = torch.triu(
                    torch.ones(seq_len, cross_seq_len, device=x.device, dtype=torch.bool), diagonal=1
                )
                full_mask |= causal.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len, cross_seq_len) for broadcasting

            if key_padding_mask is not None:
                assert key_padding_mask.dim() == 2 and key_padding_mask.shape == (batch_size, cross_seq_len), \
                    "key_padding_mask must be 2D with shape (batch_size, cross_seq_len)"
                padding = key_padding_mask.unsqueeze(1).unsqueeze(2) # (batch_size, 1, 1, cross_seq_len) for broadcasting
                full_mask |= padding

            mask_value = -1e9 if attn_scores.dtype == torch.float16 else float('-inf') # Better for mixed precision support
            attn_scores = attn_scores.masked_fill(full_mask, mask_value)

        # Apply softmax to get attention weights
        attn_scores = attn_scores - attn_scores.max(dim=-1, keepdim=True).values # For numerical stability
        attn_weights = self.dropout(F.softmax(attn_scores, dim=-1)) # (batch_size, num_heads, seq_len, cross_seq_len)

        output = torch.matmul(attn_weights, v) # (batch_size, num_heads, seq_len, head_dim)
        output = output.transpose(1, 2).reshape(batch_size, seq_len, self.dim)

        if return_attn:
            return self.out_proj(output), attn_weights.to(dtype=torch.float32)
        else:
            return self.out_proj(output)

class LayerNorm(nn.Module):
    """
    A basic implementation of Layer Normalization.
    """

    def __init__(self, features: int, eps: float = 1e-5):
        """
        __init__

        Initializes the BasicLayerNorm with the specified parameters.

        Args:
            features: Number of features to normalize.
            eps: A small value added to the denominator for numerical stability.
        """
        super(LayerNorm, self).__init__()
        self.eps = eps

        # Learnable parameters for layer normalization
        self.alphas = nn.Parameter(torch.ones(features))
        self.betas = nn.Parameter(torch.zeros(features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward

        Applies layer normalization to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, ..., features).

        Returns:
            Normalized tensor of the same shape as input.
        """
        assert x.dim() >= 2, "Input tensor must have at least 2 dimensions (batch_size, features)"
        
        # Calculate mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalize the input
        normalized_x = (x - mean) / torch.sqrt(var + self.eps)

        # Scale and shift
        return self.alphas * normalized_x + self.betas
    
class Upsample(nn.Module):
    """
    A basic implementation of an upsampling block that uses bilinear interpolation.
    """

    def __init__(self, channels: int, scale_factor: Union[int, float] = 2, 
                 mode: str = 'bilinear', align_corners: bool = True):
        """
        __init__

        Initializes the Upsample block with the specified parameters.

        Args:
            channels: Number of input channels.
            scale_factor: The factor by which to upsample the input tensor.
            mode: The interpolation mode to use ('nearest', 'bilinear', etc.).
            align_corners: If True, aligns the corners of the input and output tensors.
        """
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
        self.out_conv = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward

        Applies upsampling to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, channels, height, width).

        Returns:
            Upsampled tensor of shape (batch_size, channels, height * scale_factor, width * scale_factor).
        """
        if self.mode in ['linear', 'bilinear', 'bicubic', 'trilinear']:
            align_corners = self.align_corners if self.align_corners else None
        else:
            align_corners = None # align_corners is not used for nearest neighbor or area
            
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=align_corners)
        return self.out_conv(x)
    
class GEGLU(nn.Module):
    """
    A custom activation function that applies the GEGLU (Gated Linear Unit) activation.
    This is a variant of the GELU activation that uses a gating mechanism.
    """

    def __init__(self, in_dim: int, inter_dim: int, bias: bool = True):
        """
        __init__

        Initializes the GEGLU activation with the specified parameters.

        Args:
            in_dim: Input dimensionality.
            inter_dim: Dimensionality of the intermediate representation.
            bias: If True, adds a bias term to the linear layers.
        """
        super(GEGLU, self).__init__()
        self.linear_geglu_1 = nn.Linear(in_dim, inter_dim * 2, bias=bias)
        self.linear_geglu_2 = nn.Linear(inter_dim, in_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward

        Applies the GEGLU activation to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, ..., in_dim).

        Returns:
            Output tensor of shape (batch_size, ..., in_dim).
        """
        assert x.dim() >= 2, "Input tensor must be at least 2D (batch_size, ..., in_dim)"

        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1)
        return self.linear_geglu_2(x * F.gelu(gate))
    
class MultiWaveletAct(nn.Module):
    """
    A custom activation function that applies a mixture of Gabor's like wavelets
    as an activation function.

    This use of wavelets is expected to improve the model's ability to model
    multi-frequency details in the data.
    """

    def __init__(self, dim: int = 256, num_components: int = 4, freq_scale: float = 1.0, 
                 scale: float = 1.0, activation: nn.Module = nn.ReLU(), 
                 wavelet_only: bool = True, normalize: bool = True):
        """
        __init__

        Initializes the WaveletAct with the specified parameters.

        Args:
            dim: Dimensionality of the input tensor.
            num_components: Number of wavelet components to use.
            freq_scale: Scaling factor for the frequency of the wavelets.
            scale: Scaling factor for the amplitude of the wavelets.
            activation: Activation function to apply after the wavelet transformation.
            wavelet_only: If True, only applies the wavelet transformation without the skip connection.
            normalize: If True, uses layer normalization on the wavelet output.
        """
        super(MultiWaveletAct, self).__init__()
        self.dim = dim
        self.num_components = num_components
        self.activation = activation
        self.wavelet_only = wavelet_only

        # Initialize wavelet parameters
        init_freqs = torch.linspace(0.5, 3., num_components) * freq_scale
        self.freq_scales = nn.Parameter(init_freqs + 0.1 * torch.randn(num_components))
        self.log_scales = nn.Parameter(torch.randn(num_components) * scale)

        # Leanrable weights for the wavelet components
        self.weights = nn.Parameter(torch.ones(num_components) / num_components)

        if wavelet_only:
            self.project = nn.Linear(dim, dim, bias=True)
        else:
            self.project = nn.Linear(dim * 2, dim, bias=True)

        if normalize:
            self.layer_norm = LayerNorm(dim)
        else:
            self.layer_norm = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        forward

        Applies the wavelet activation function to the input tensor.

        Args:
            x: Input tensor of shape (batch_size, ..., dim).

        Returns:
            Output tensor of shape (batch_size, ..., dim).
        """
        assert x.dim() >= 2, "Input tensor must be 2D (batch_size, ..., dim)"
        assert x.shape[-1] == self.dim, \
            f"Input tensor must have last dimension of size {self.dim}, got {x.shape[-1]}"
        
        x_exp = x.unsqueeze(1) # Expand to (batch_size, 1, ..., dim)

        freqs = self.freq_scales.view(self.num_components, *([1] * (x.dim() - 1))) # Expand to (num_components, ..., 1)
        sigmas = torch.clamp(torch.exp(self.log_scales), min=1e-3).view(self.num_components, *([1] * (x.dim() - 1)))

        print(f"Input shape: {x.shape}, Expanded shape: {x_exp.shape}, Frequencies shape: {freqs.shape}, Sigmas shape: {sigmas.shape}")

        # Apply wavelet transformation
        wavelets = torch.cos(freqs * x_exp) * torch.exp(-0.5 * (x_exp / sigmas) ** 2) # (batch_size, num_components, ..., dim)

        weights = F.softmax(self.weights, dim=0).view(self.num_components, *([1] * (x.dim() - 1))) # Normalize weights across components
        result = torch.sum(wavelets * weights, dim=1, keepdim=False)

        if self.wavelet_only:
            result = self.project(result)
            return self.activation(self.layer_norm(result)) # Apply activation and normalization
        else:
            result = self.project(torch.cat([x, result], dim=-1))
            return self.activation(self.layer_norm(result))