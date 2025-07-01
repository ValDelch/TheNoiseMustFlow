"""
decoder_blocks.py

The module implements the decoding path of VAE and UNet architectures.
"""


from __future__ import annotations
from typing import Type

import yaml
import copy
import os

import torch
from torch import nn

from core.basic_components.basic_blocks import BasicResidualBlock, BasicAttentionBlock
from core.basic_components.functional_blocks import Upsample



class VADecoderBlock(nn.Module):
    """
    The decoder path of a basic Variational Autoencoder (VAE) architecture.
    """

    def __init__(self, in_channels: int, config_file: str, 
                 ResidualBlock: Type[nn.Module] = BasicResidualBlock,
                 AttentionBlock: Type[nn.Module] = BasicAttentionBlock):
        """
        __init__

        Initializes the decoding path of a VAE architecture.

        Args:
            in_channels: Number of input channels, defaults to 1.
            config_file: Path to a YAML configuration file for the decoder block.
            ResidualBlock: Class for the residual block, defaults to BasicResidualBlock.
            AttentionBlock: Class for the attention block, defaults to BasicAttentionBlock.
        """
        super(VADecoderBlock, self).__init__()
        assert config_file.endswith('.yaml'), "Configuration file must be a YAML file."
        assert os.path.exists(config_file), f"Configuration file '{config_file}' does not exist."

        self.in_channels = in_channels
        self.residual_block = ResidualBlock
        self.attention_block = AttentionBlock

        # Load the configuration file and instantiate the decoder
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.decoder = self.parse_model_config(config)

    def parse_model_config(self, config: dict) -> nn.Sequential:
        """
        parse_model_config

        Parses the model configuration and builds the decoder as a sequential model.

        Args:
            config: Dictionary containing the model configuration.
        
        Returns:
            A sequential model constructed from the configuration.
        """
        required_fields = {
            "base_channels": int,
            "latent_dim": int,
            "activation": str,
            "groups": int,
            "dropout": float
        }
        if "AttBlock" in [k['use'] for k in config['decoder']]:
            required_fields["num_heads"] = int

        # Validate the global parameters
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required global field '{field}' in configuration file.")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Global field '{field}' must be of type {field_type.__name__}, got {type(config[field]).__name__}")

        # Assert value ranges
        assert config["base_channels"] > 0, "base_channels must be > 0"
        assert config["latent_dim"] > 0, "latent_dim must be > 0"
        assert config["num_heads"] > 0 if "num_heads" in config else True, "num_heads must be > 0" 
        assert 0 <= config["dropout"] <= 1, "dropout must be in [0, 1]"
        assert 0 < config["groups"] <= config["base_channels"], "groups must be in [0, base_channels["

        # Validate activation function
        act_string = config["activation"]
        try:
            activation_fn = getattr(nn, act_string)
            assert issubclass(activation_fn, nn.Module), \
                f"Activation '{act_string}' must be a subclass of nn.Module"
        except (AttributeError, AssertionError, ValueError) as e:
            raise ValueError(f"Invalid activation spec '{act_string}': {e}")
        
        # Validate and construct the decoder structure
        if "decoder" not in config:
            raise ValueError("Missing top-level 'decoder' in configuration file.")
        if "blocks" not in config:
            raise ValueError("Missing 'blocks' field in the configuration file. \
                             Please define the blocks used in the encoder/decoder.")
        
        decoder = config["decoder"]
        blocks = config["blocks"]

        def resolve_f(val):
            if isinstance(val, str) and 'f' in val:
                expr = val.replace('f', f'*{config["base_channels"]}')
                return int(eval(expr))
            return val

        def resolve_dict(d):
            return {k: resolve_f(v) for k, v in d.items()}
        
        layers = []
        in_channels = int(config["latent_dim"])
        for i, layer in enumerate(decoder):
            if 'use' not in layer:
                raise ValueError("Each layer must specify a 'use' field to indicate the type of block.")
            
            block_name = layer['use']
            override = layer.get("with", {})
            template = copy.deepcopy(blocks[block_name])
            kind = template.pop("type")
            for key in template:
                if isinstance(template[key], str) and template[key].startswith("$"):
                    var = template[key][1:]
                    template[key] = override[var]
            params = resolve_dict(template)

            print(f"Adding layer {i}: {kind} with parameters {params}")

            if i == 0:
                # Initialize the first two convolutional layers
                layers.append(nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0))
                layers.append(nn.Conv2d(in_channels, int(params["out_channels"]), kernel_size=3, padding=1))
                in_channels = int(params["out_channels"])

            n_repeat = int(params.pop("repeat", 1))
            for _ in range(n_repeat):
                if kind == "conv":
                    layers.append(nn.Conv2d(in_channels, **params))
                    in_channels = int(params["out_channels"])
                if kind == "upsample":
                    layers.append(Upsample(in_channels, **params))
                elif kind == "res":
                    layers.append(
                        self.residual_block(
                            in_channels, groups=config["groups"], activation=activation_fn(),
                            dropout=config["dropout"], **params
                        )
                    )
                    in_channels = int(params["out_channels"])
                elif kind == "att":
                    layers.append(
                        self.attention_block(
                            in_channels, num_heads=config["num_heads"], groups=config["groups"],
                            dropout=config["dropout"], **params
                        )
                    )
                else:
                    raise ValueError(f"Unknown layer type: {kind}. Supported types are 'conv', 'res', and 'att'.")
                
        layers.append(nn.GroupNorm(num_groups=config["groups"], num_channels=in_channels))
        layers.append(activation_fn())
        layers.append(nn.Conv2d(in_channels, self.in_channels, kernel_size=3, padding=1))
                
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, rescale: bool = True) -> torch.Tensor:
        """
        forward

        Forward pass through the decoder block.

        Args:
            x: Input tensor to the decoder.
            rescale: If True, rescales the output tensor by 1./0.18215 (default is False).

        Returns:
            torch.Tensor: Decoded tensor of shape (batch_size, in_channels, height, width).
        """
        x = self.decoder(x)

        if rescale:
            return x / 0.18215
        return x



if __name__ == "__main__":
    # Example usage
    config_path = '../../../configs/default_VAE.yaml'
    VADecoder = VADecoderBlock(in_channels=1, config_file=config_path)

    x = torch.randn(1, 4, 32, 32) # Example input tensor
    output = VADecoder(x, rescale = False)
    print("Output shape:", output.shape) # Encoded tensor