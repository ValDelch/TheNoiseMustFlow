"""
encoder_blocks.py

The module implements the encoding path of VAE and UNet architectures.
"""


from __future__ import annotations
from typing import Type, Union

import yaml
import copy
import os

import torch
from torch import nn

from core.basic_components.basic_blocks import BasicResidualBlock, BasicAttentionBlock


# Type alias
TensorOrMore = Union[
    torch.Tensor,
    tuple[torch.Tensor, tuple]
]


class VAEncoderBlock(nn.Module):
    """
    The encoder path of a basic Variational Autoencoder (VAE) architecture.
    """

    def __init__(self, in_channels: int, config_file: str, 
                 ResidualBlock: Type[nn.Module] = BasicResidualBlock,
                 AttentionBlock: Type[nn.Module] = BasicAttentionBlock):
        """
        __init__

        Initializes the encoding path of a VAE architecture.

        Args:
            in_channels: Number of input channels, defaults to 1.
            config_file: Path to a YAML configuration file for the encoder block.
            ResidualBlock: Class for the residual block, defaults to BasicResidualBlock.
            AttentionBlock: Class for the attention block, defaults to BasicAttentionBlock.
        """
        super(VAEncoderBlock, self).__init__()
        assert config_file.endswith('.yaml'), "Configuration file must be a YAML file."
        assert os.path.exists(config_file), f"Configuration file '{config_file}' does not exist."

        self.in_channels = in_channels
        self.residual_block = ResidualBlock
        self.attention_block = AttentionBlock

        # Load the configuration file and instantiate the encoder
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.encoder = self.parse_model_config(config)

    def parse_model_config(self, config: dict) -> nn.Sequential:
        """
        parse_model_config

        Parses the model configuration and builds the encoder as a sequential model.

        Args:
            config: Dictionary containing the model configuration.

        Returns:
            nn.Sequential: A sequential model representing the encoder.
        """
        required_fields = {
            "base_channels": int,
            "latent_dim": int,
            "activation": str,
            "groups": int,
            "dropout": float
        }
        if "AttBlock" in [k['use'] for k in config['encoder']]:
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
        
        # Validate and construct the encoder structure
        if "encoder" not in config:
            raise ValueError("Missing top-level 'encoder' in configuration file.")
        if "blocks" not in config:
            raise ValueError("Missing 'blocks' field in the configuration file. \
                             Please define the blocks used in the encoder/decoder.")
        
        encoder = config["encoder"]
        blocks = config["blocks"]

        def resolve_f(val):
            if isinstance(val, str) and 'f' in val:
                expr = val.replace('f', f'*{config["base_channels"]}')
                return int(eval(expr))
            return val

        def resolve_dict(d):
            return {k: resolve_f(v) for k, v in d.items()}

        layers = []
        in_channels = self.in_channels
        for layer in encoder:
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

            n_repeat = int(params.pop("repeat", 1))
            for _ in range(n_repeat):
                if kind == "conv":
                    layers.append(nn.Conv2d(in_channels, **params))
                    in_channels = int(params["out_channels"])
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
        layers.append(nn.Conv2d(in_channels, config["latent_dim"]*2, kernel_size=3, padding=1))
        layers.append(nn.Conv2d(config["latent_dim"]*2, config["latent_dim"]*2, kernel_size=1, padding=0))
                
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor, noise: torch.Tensor, 
                return_stats: bool = False, rescale: bool = False) -> TensorOrMore:
        """
        forward

        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            noise: Noise tensor of shape (batch_size, latent_dim, height', width').
            return_stats: If True, returns the mean and log variance statistics.
            rescale: If True, rescales the output tensor by 0.18215 (default is False).

        Returns:
            torch.Tensor: Encoded tensor of shape (batch_size, latent_dim, height', width').
        """
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, in_channels, height, width)"

        x = self.encoder(x) # (batch_size, latent_dim*2, height', width')

        mean, log_var = torch.chunk(x, 2, dim=1) # Split into mean and log variance
        assert mean.shape == noise.shape, \
            f"Mean shape {mean.shape} does not match noise shape {noise.shape}"

        log_variance = torch.clamp(log_var, min=-30.0, max=20.0)
        variance = torch.exp(log_variance)
        stdev = torch.sqrt(variance)

        x = mean + stdev * noise # Reparameterization trick

        # Rescale if not in training mode
        if rescale:
            x = x * 0.18215

        if return_stats:
            return x, (mean, log_variance)
        return x
    
class UNetEncoderBlock(nn.Module):
    """
    The encoder path of a basic U-Net architecture.
    """

    def __init__(self, latent_dim: int, config_file: str, 
                 ResidualBlock: Type[nn.Module] = BasicResidualBlock,
                 AttentionBlock: Type[nn.Module] = BasicAttentionBlock):
        """
        __init__

        Initializes the encoding path of a U-Net architecture.

        Args:
            latent_dim: Dimension of the latent space.
            config_file: Path to a YAML configuration file for the encoder block.
            ResidualBlock: Class for the residual block, defaults to BasicResidualBlock.
            AttentionBlock: Class for the attention block, defaults to BasicAttentionBlock.
        """
        super(UNetEncoderBlock, self).__init__()
        # Implementation similar to VAEncoderBlock but tailored for U-Net specifics
        assert config_file.endswith('.yaml'), "Configuration file must be a YAML file."
        assert os.path.exists(config_file), f"Configuration file '{config_file}' does not exist."

        self.latent_dim = latent_dim
        self.residual_block = ResidualBlock
        self.attention_block = AttentionBlock

        # Load the configuration file and instantiate the encoder
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)

        self.encoder, self.bottleneck = self.parse_model_config(config)

    def parse_model_config(self, config: dict) -> tuple[nn.Sequential, nn.Sequential]:
        """
        parse_model_config

        Parses the model configuration and builds the encoder and bottleneck as sequential models.

        Args:
            config: Dictionary containing the model configuration.

        Returns:
            A tuple of sequential models representing the encoder and bottleneck.
        """
        required_fields = {
            "latent_dim": int,
            "d_time": int,
            "d_context": int,
            "num_heads": int,
            "activation": str,
            "groups": int,
            "dropout": float
        }

        # Validate the global parameters
        for field, field_type in required_fields.items():
            if field not in config:
                raise ValueError(f"Missing required global field '{field}' in configuration file.")
            if not isinstance(config[field], field_type):
                raise TypeError(f"Global field '{field}' must be of type {field_type.__name__}, got {type(config[field]).__name__}")

        # Assert value ranges
        assert config["latent_dim"] > 0, "latent_dim must be > 0"
        assert config["d_time"] > 0, "d_time must be > 0"
        assert config["d_context"] > 0, "d_context must be > 0"
        assert config["num_heads"] > 0, "num_heads must be > 0"
        assert 0 <= config["dropout"] <= 1, "dropout must be in [0, 1]"

        # Validate activation function
        act_string = config["activation"]
        try:
            activation_fn = getattr(nn, act_string)
            assert issubclass(activation_fn, nn.Module), \
                f"Activation '{act_string}' must be a subclass of nn.Module"
        except (AttributeError, AssertionError, ValueError) as e:
            raise ValueError(f"Invalid activation spec '{act_string}': {e}")
        
        # Validate and construct the encoder structure
        if "encoder" not in config:
            raise ValueError("Missing top-level 'encoder' in configuration file.")
        if "bottleneck" not in config:
            raise ValueError("Missing top-level 'bottleneck' in configuration file.")
        if "blocks" not in config:
            raise ValueError("Missing 'blocks' field in the configuration file. \
                             Please define the blocks used in the encoder/bottleneck/decoder.")
        
        blocks = config["blocks"]

        # Encoder
        encoder = config["encoder"]
        layers = []
        in_channels = int(config["latent_dim"])
        for layer in encoder:
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
            params = template.copy()

            n_repeat = int(params.pop("repeat", 1))
            n_skip_channels = int(params.pop("skip_channels", 0))
            for _ in range(n_repeat):
                if kind == "conv":
                    layers.append(nn.Conv2d(in_channels, **params))
                    in_channels = int(params["out_channels"])
                elif kind == "res":
                    layers.append(
                        self.residual_block(
                            in_channels+n_skip_channels, groups=config["groups"], activation=activation_fn(),
                            dropout=config["dropout"], d_context=config["d_time"], **params
                        )
                    )
                    in_channels = int(params["out_channels"])
                elif kind == "att":
                    layers.append(
                        self.attention_block(
                            in_channels, num_heads=config["num_heads"], groups=config["groups"],
                            dropout=config["dropout"], d_context=config["d_context"], **params
                        )
                    )
                elif kind == "res_and_att":
                    _layer = nn.Sequential(
                        self.residual_block(
                            in_channels+n_skip_channels, groups=config["groups"], activation=activation_fn(),
                            dropout=config["dropout"], d_context=config["d_time"], **params
                        ),
                        self.attention_block(
                            channels=params["out_channels"], num_heads=config["num_heads"], groups=config["groups"],
                            dropout=config["dropout"], d_context=config["d_context"]
                        )
                    )
                    layers.append(_layer)
                    in_channels = int(params["out_channels"])
                else:
                    raise ValueError(f"Unknown layer type: {kind}. Supported types are 'conv', 'res', 'att', and 'res_and_att'.")
                
        encoder = nn.Sequential(*layers)
                    
        # Bottleneck
        bottleneck = config["bottleneck"]
        layers = []
        for layer in bottleneck:
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
            params = template.copy()

            n_repeat = int(params.pop("repeat", 1))
            n_skip_channels = int(params.pop("skip_channels", 0))
            if kind == "conv":
                layers.append(nn.Conv2d(in_channels, **params))
                in_channels = int(params["out_channels"])
            elif kind == "res":
                layers.append(
                    self.residual_block(
                        in_channels, groups=config["groups"], activation=activation_fn(),
                        dropout=config["dropout"], d_context=config["d_time"], **params
                    )
                )
                in_channels = int(params["out_channels"])
            elif kind == "att":
                layers.append(
                    self.attention_block(
                        in_channels, num_heads=config["num_heads"], groups=config["groups"],
                        dropout=config["dropout"], d_context=config["d_context"], **params
                    )
                )
            elif kind == "res_and_att":
                _layer = nn.Sequential(
                    self.residual_block(
                        in_channels, groups=config["groups"], activation=activation_fn(),
                        dropout=config["dropout"], d_context=config["d_time"], **params
                    ),
                    self.attention_block(
                        channels=params["out_channels"], num_heads=config["num_heads"], groups=config["groups"],
                        dropout=config["dropout"], d_context=config["d_context"]
                    )
                )
                layers.append(_layer)
                in_channels = int(params["out_channels"])
            else:
                raise ValueError(f"Unknown layer type: {kind}. Supported types are 'conv', 'res', 'att', and 'res_and_att'.")
            
        bottleneck = nn.Sequential(*layers)

        return (encoder, bottleneck)

    def forward(self, x: torch.Tensor, context: torch.Tensor,
                time: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """
        forward

        Forward pass through the encoder.

        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width).
            context: Context tensor for conditioning, shape (batch_size, seq_len, d_context).
            time: Time tensor for conditioning, shape (batch_size, d_time).

        Returns:
            The encoded tensor and the list of skip connections.
        """
        assert x.dim() == 4, "Input tensor must be 4D (batch_size, in_channels, height, width)"
        assert context.dim() == 3, "Context tensor must be 3D (batch_size, seq_len, d_context)"
        assert time.dim() == 2, "Time tensor must be 2D (batch_size, d_time)"

        def forward_cascade(model, x, context, time):
            skip_connections = []
            for layer in model:
                if isinstance(layer, nn.Conv2d):
                    x = layer(x)
                elif isinstance(layer, self.residual_block):
                    x = layer(x, context=time)
                elif isinstance(layer, self.attention_block):
                    x = layer(x, context=context)
                elif isinstance(layer, nn.Sequential):
                    for sublayer in layer:
                        if isinstance(sublayer, self.residual_block):
                            x = sublayer(x, context=time)
                        elif isinstance(sublayer, self.attention_block):
                            x = sublayer(x, context=context)

                skip_connections.append(x)
            return x, skip_connections

        # Pass through the encoder
        x, skip_connections = forward_cascade(self.encoder, x, context, time)

        # Pass through the bottleneck
        x, _ = forward_cascade(self.bottleneck, x, context, time)

        return (x, skip_connections)



if __name__ == "__main__":
    # Example usage

    # VAEncoderBlock
    config_path = '../../../configs/default_VAE.yaml'
    VAEncoder = VAEncoderBlock(in_channels=1, config_file=config_path)

    x = torch.randn(1, 1, 256, 256) # Example input tensor
    noise = torch.randn(1, 4, 32, 32) # Example noise tensor
    output = VAEncoder(x, noise, return_stats=True)
    print("Output shape:", output[0].shape) # Encoded tensor
    print("Mean shape:", output[1][0].shape) # Mean tensor
    print("Log variance shape:", output[1][1].shape) # Log variance tensor

    # UNetEncoderBlock
    config_path = '../../../configs/default_UNet.yaml'
    UNetEncoder = UNetEncoderBlock(latent_dim=4, config_file=config_path)

    x = torch.randn(1, 4, 32, 32) # Example input tensor
    context = torch.randn(1, 10, 768) # Example context tensor
    time = torch.randn(1, 1280) # Example time tensor
    output, skip_connections = UNetEncoder(x, context, time)
    print("Output shape:", output.shape) # Encoded tensor
    print("Skip connections length:", len(skip_connections))
    for i, skip in enumerate(skip_connections):
        print(f"Skip connection {i+1} shape:", skip.shape) # Shapes of skip connections