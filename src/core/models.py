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
from core.basic_components.encoder_blocks import VAEncoderBlock, UNetEncoderBlock
from core.basic_components.decoder_blocks import VADecoderBlock, UNetDecoderBlock
from core.basic_components.encodings import TimeEncoding


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
        __init__

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
    
class UNet(nn.Module):
    """
    U-Net model.

    This model consists of an encoder and a decoder U-Net block.
    The encoder compresses the input into a latent representation, and
    the decoder reconstructs the input from the latent space, with skip connections.
    """

    def __init__(self, latent_dim: int, config_file: str,
                 ResidualBlock: Type[nn.Module] = BasicResidualBlock,
                 AttentionBlock: Type[nn.Module] = BasicAttentionBlock):
        """
        __init__

        Initializes the U-Net model.

        Args:
            latent_dim: Dimension of the latent space.
            config_file: Path to a YAML configuration file for the encoder and decoder blocks.
            ResidualBlock: Class for the residual block, defaults to BasicResidualBlock.
            AttentionBlock: Class for the attention block, defaults to BasicAttentionBlock.
        """
        super(UNet, self).__init__()

        self.encoder = UNetEncoderBlock(
            latent_dim=latent_dim,
            config_file=config_file,
            ResidualBlock=ResidualBlock,
            AttentionBlock=AttentionBlock
        )
        self.decoder = UNetDecoderBlock(
            latent_dim=latent_dim,
            config_file=config_file,
            ResidualBlock=ResidualBlock,
            AttentionBlock=AttentionBlock
        )

    def forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        forward

        Forward pass of the U-Net model.
        This method processes the input tensor through the encoder and decoder blocks.

        Args:
            x: Input tensor.
            context: Context tensor for the encoder.
            time: Time step embedding for the decoder.

        Returns:
            The output tensor from the decoder.
        """
        # Pass through the encoder
        out, skip_connections = self.encoder(x, context=context, time=time)
        
        # Pass through the decoder with skip connections
        out = self.decoder(out, skip_connections=skip_connections, context=context, time=time)

        return out

class Diffusion(nn.Module):
    """
    Diffusion model.

    This model is basically a U-Net along with a time step embedding.
    """

    def __init__(self, latent_dim: int, d_time: int, config_file: str,
                 ResidualBlock: Type[nn.Module] = BasicResidualBlock,
                 AttentionBlock: Type[nn.Module] = BasicAttentionBlock,
                 TimeEncoder: Type[nn.Module] = TimeEncoding):
        """
        __init__

        Initializes the Diffusion model.

        Args:
            latent_dim: Dimension of the latent space.
            d_time: Dimension of the time step embedding.
            config_file: Path to a YAML configuration file for the encoder and decoder blocks.
            ResidualBlock: Class for the residual block, defaults to BasicResidualBlock.
            AttentionBlock: Class for the attention block, defaults to BasicAttentionBlock.
            TimeEncoder: Class for the time encoding, defaults to TimeEncoding.
        """
        super(Diffusion, self).__init__()

        self.time_encoder = TimeEncoder(dim=d_time)
        self.unet = UNet(
            latent_dim=latent_dim,
            config_file=config_file,
            ResidualBlock=ResidualBlock,
            AttentionBlock=AttentionBlock
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        forward

        Forward pass of the Diffusion model.
        This method processes the input tensor through the time encoding and U-Net.

        Args:
            x: Input tensor.
            t: Time step tensor.
            context: Context tensor for the U-Net.

        Returns:
            The output tensor from the U-Net.
        """
        # Get time encoding
        t_enc = self.time_encoder(t)

        # Pass through the U-Net
        out = self.unet(x, context=context, time=t_enc)

        return out