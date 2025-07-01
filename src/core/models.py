"""
models.py

The module implements the different deep learning models that
are useful in the diffusion pipeline. The main architectures are
the U-Net used to sample noise, the VAE to build latent
representations of images.
"""


from __future__ import annotations
from typing import Type, Union

import torch
from torch import nn

from core.basic_components.basic_blocks import BasicResidualBlock, BasicAttentionBlock
from core.basic_components.encoder_blocks import VAEncoderBlock
from core.basic_components.decoder_blocks import VADecoderBlock

# Type alias
TensorOrMore = Union[
    torch.Tensor,
    tuple[torch.Tensor, tuple],
    tuple[torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, tuple, torch.Tensor]
]


class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.

    This model consists of an encoder and a decoder block.
    The encoder compresses the input into a latent representation,
    while the decoder reconstructs the input from the latent space.
    """

    def __init__(self, in_channels: int, config_file: str,
                 ResidualBlock: Type[nn.Module] = BasicResidualBlock,
                 AttentionBlock: Type[nn.Module] = BasicAttentionBlock):
        """
        Initializes the VAE model.

        Args:
            in_channels: Number of input channels, defaults to 1.
            config_file: Path to a YAML configuration file for the encoder and decoder blocks.
            ResidualBlock: Class for the residual block, defaults to BasicResidualBlock.
            AttentionBlock: Class for the attention block, defaults to BasicAttentionBlock.
        """
        super(VAE, self).__init__()

        self.encoder = VAEncoderBlock(
            in_channels=in_channels,
            config_file=config_file,
            ResidualBlock=ResidualBlock,
            AttentionBlock=AttentionBlock
        )
        self.decoder = VADecoderBlock(
            in_channels=in_channels,
            config_file=config_file,
            ResidualBlock=ResidualBlock,
            AttentionBlock=AttentionBlock
        )
        
    def forward(self, x: torch.Tensor, noise: torch.Tensor, return_stats: bool = False, 
                rescale: bool = False, return_rec: bool = False) -> TensorOrMore:
        """
        forward

        Forward pass of the VAE model.
        This method processes the input tensor through the encoder. It can optionally return
        the mean and log variance from the encoder, and the reconstructed output from the decoder.

        Args:
            x: Input tensor.
            noise: Optional noise tensor for the encoder.
            return_stats: If True, returns mean and log variance from the encoder.
            rescale: If True, rescales the output of the decoder.
            return_rec: If True, returns the reconstructed output from the decoder.

        Returns:
            Always returns the encoder output.
            If return_stats is True, also returns the mean and log variance.
            If return_rec is True, also returns the reconstructed output from the decoder.
        """
        # Pass through the encoder
        out, (mean, logvar) = self.encoder(x, noise, return_stats=True, rescale=rescale)
        if return_rec:
            # Pass through the decoder
            rec = self.decoder(out, rescale=rescale)
            if return_stats:
                return out, (mean, logvar), rec
            return out, rec
        if return_stats:
            return out, (mean, logvar)
        return out