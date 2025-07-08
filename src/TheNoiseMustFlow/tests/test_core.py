# tests/test_core.py

import warnings
import os

import pytest
import torch
from torch import nn

from TheNoiseMustFlow.core.schedulers import NoiseScheduler
from TheNoiseMustFlow.core.samplers import DDPMSampler, DDIMSampler
from TheNoiseMustFlow.core.basic_components.encodings import TimeEncoding
from TheNoiseMustFlow.core.basic_components.basic_blocks import (
    BasicResidualBlock,
    BasicAttentionBlock,
    BasicFeedForwardBlock,
)
from TheNoiseMustFlow.core.basic_components.functional_blocks import (
    LayerNorm,
    MultiWaveletAct,
    SelfAttention,
    CrossAttention,
    GEGLU,
    Upsample,
)
from TheNoiseMustFlow.core.basic_components.encoder_blocks import (
    VAEncoderBlock,
    UNetEncoderBlock,
)
from TheNoiseMustFlow.core.basic_components.decoder_blocks import (
    VADecoderBlock,
    UNetDecoderBlock,
)
from TheNoiseMustFlow.core.models import VAE, UNet, Diffusion

#
# Tests for the schedulers
#

# NoiseScheduler


@pytest.mark.parametrize(
    "schedule", ["linear", "cosine", "quadratic", "sigmoid", "geometric"]
)
def test_scheduler_init_valid(schedule):
    scheduler = NoiseScheduler(steps=10, betas=(1e-4, 0.02), schedule=schedule)
    assert scheduler.steps == 10
    assert scheduler.schedule == schedule
    assert scheduler.alphas_cumprod.shape[0] == 10


def test_scheduler_invalid_schedule():
    with pytest.raises(AssertionError, match="schedule must be one of"):
        NoiseScheduler(schedule="invalid")


def test_scheduler_invalid_steps():
    with pytest.raises(AssertionError, match="steps must be a positive integer"):
        NoiseScheduler(steps=0)


def test_noise_add_step_single():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(3, 32, 32)
    t = 10
    noisy_x = scheduler.add_noise_step(x, t)
    assert noisy_x.shape == x.shape
    assert not torch.equal(noisy_x, x)


def test_noise_add_step_batch():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(4, 3, 32, 32)
    t = torch.tensor([0, 1, 2, 3])
    noisy_x = scheduler.add_noise_step(x, t)
    assert noisy_x.shape == x.shape


def test_noise_add_cumulative_single():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(3, 32, 32)
    t = 5
    noisy_x = scheduler.add_noise_cumulative(x, t)
    assert noisy_x.shape == x.shape


def test_noise_add_cumulative_batch():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(4, 3, 32, 32)
    t = torch.tensor([0, 1, 2, 3])
    noisy_x = scheduler.add_noise_cumulative(x, t)
    assert noisy_x.shape == x.shape


def test_invalid_t_out_of_bounds():
    scheduler = NoiseScheduler(steps=10, device="cpu")
    x = torch.ones(2, 3, 32, 32)
    t = 100  # out of bounds
    with pytest.raises(AssertionError):
        scheduler.add_noise_step(x, t)


def test_invalid_t_mismatch_batch_size():
    scheduler = NoiseScheduler(steps=10, device="cpu")
    x = torch.ones(2, 3, 32, 32)
    t = torch.tensor([1])  # batch size mismatch
    with pytest.raises(AssertionError):
        scheduler.add_noise_step(x, t)


def test_invalid_t_type():
    scheduler = NoiseScheduler(steps=10, device="cpu")
    x = torch.ones(2, 3, 32, 32)
    with pytest.raises(TypeError):
        scheduler.add_noise_step(x, t="invalid")


def test_cosine_schedule_betas_less_than_1():
    scheduler = NoiseScheduler(steps=10, schedule="cosine", device="cpu")
    assert torch.all(scheduler.betas >= 0)
    assert torch.all(scheduler.betas <= 0.999)


def test_broadcasting_step_batch_scalar_t():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(4, 3, 32, 32)
    t = 5  # scalar t, should broadcast to all
    out = scheduler.add_noise_step(x, t)
    assert out.shape == x.shape


def test_broadcasting_cumulative_batch_scalar_t():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(4, 3, 32, 32)
    t = 5
    out = scheduler.add_noise_cumulative(x, t)
    assert out.shape == x.shape


def test_3d_input_with_batch_t_raises():
    scheduler = NoiseScheduler(steps=100, device="cpu")
    x = torch.ones(3, 32, 32)  # 3D input (no batch)
    t = torch.tensor([1, 2])  # Batched t
    with pytest.raises(AssertionError, match="Batch size of t must match"):
        scheduler.add_noise_step(x, t)


def test_add_noise_step_with_external_noise():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(3, 32, 32)
    t = 10
    noise = torch.randn_like(x)
    out = scheduler.add_noise_step(x, t, noise=noise)
    expected = (
        torch.sqrt(1 - scheduler.betas[t]) * x + torch.sqrt(scheduler.betas[t]) * noise
    )
    assert torch.allclose(out, expected)


def test_add_noise_step_batch_with_external_noise():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(2, 3, 32, 32)
    t = torch.tensor([5, 10])
    noise = torch.randn_like(x)
    out = scheduler.add_noise_step(x, t, noise=noise)
    betas = scheduler.betas[t].view(-1, 1, 1, 1)
    expected = torch.sqrt(1 - betas) * x + torch.sqrt(betas) * noise
    assert torch.allclose(out, expected)


def test_add_noise_step_wrong_noise_shape_raises():
    scheduler = NoiseScheduler(steps=100, device="cpu")
    x = torch.ones(3, 32, 32)
    t = 10
    noise = torch.randn(1, 3, 32, 32)  # mismatched shape
    with pytest.raises(AssertionError, match="must have the same shape"):
        scheduler.add_noise_step(x, t, noise=noise)


def test_add_noise_cumulative_with_external_noise():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(3, 32, 32)
    t = 10
    noise = torch.randn_like(x)
    out = scheduler.add_noise_cumulative(x, t, noise=noise)
    expected = (
        torch.sqrt(scheduler.alphas_cumprod[t]) * x
        + torch.sqrt(1 - scheduler.alphas_cumprod[t]) * noise
    )
    assert torch.allclose(out, expected)


def test_add_noise_cumulative_batch_with_external_noise():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42, device="cpu")
    x = torch.ones(2, 3, 32, 32)
    t = torch.tensor([10, 20])
    noise = torch.randn_like(x)
    alphas = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
    expected = torch.sqrt(alphas) * x + torch.sqrt(1 - alphas) * noise
    out = scheduler.add_noise_cumulative(x, t, noise=noise)
    assert torch.allclose(out, expected)


def test_add_noise_cumulative_wrong_noise_shape_raises():
    scheduler = NoiseScheduler(steps=100, device="cpu")
    x = torch.ones(3, 32, 32)
    t = 10
    noise = torch.randn(1, 3, 32, 32)
    with pytest.raises(AssertionError, match="must have the same shape"):
        scheduler.add_noise_cumulative(x, t, noise=noise)


def test_linear_noise_one_image_full_process():
    scheduler = NoiseScheduler(steps=10, schedule="linear", seed=42, device="cpu")
    x = torch.ones(3, 32, 32)
    for i in range(scheduler.steps):
        x = scheduler.add_noise_step(x, i)


def test_cosine_noise_one_image_full_process():
    scheduler = NoiseScheduler(steps=10, schedule="cosine", seed=42, device="cpu")
    x = torch.ones(3, 32, 32)
    for i in range(scheduler.steps):
        x = scheduler.add_noise_step(x, i)


def test_cosine_noise_snr():
    scheduler = NoiseScheduler(steps=10, schedule="cosine", seed=42, device="cpu")
    snr_1 = scheduler.compute_snr(1)
    snr_2 = scheduler.compute_snr(5)
    assert snr_1 > snr_2, "SNR should decrease as t increases"


def test_linear_noise_snr():
    scheduler = NoiseScheduler(steps=10, schedule="linear", seed=42, device="cpu")
    snr_1 = scheduler.compute_snr(1)
    snr_2 = scheduler.compute_snr(5)
    assert snr_1 > snr_2, "SNR should decrease as t increases"


def test_cosine_noise_snr_tensor():
    scheduler = NoiseScheduler(steps=10, schedule="cosine", seed=42, device="cpu")
    t = torch.tensor([1, 5, 9])
    snr = scheduler.compute_snr(t)
    assert snr.shape == (3,), "SNR should return a tensor with the same shape as t"
    assert snr[0] > snr[1] > snr[2], "SNR should decrease as t increases"


#
# Tests for the samplers
#


@pytest.fixture
def scheduler():
    return NoiseScheduler(
        steps=100, betas=(1e-4, 0.02), schedule="linear", seed=123, device="cpu"
    )


@pytest.fixture
def image():
    return torch.ones(3, 32, 32)


@pytest.fixture
def batch_image():
    return torch.ones(2, 3, 32, 32)


@pytest.fixture
def dummy_pred_noise():
    return lambda x, t: torch.zeros_like(x)


# DDPMSampler


def test_ddpm_sample_prev_step_shape(scheduler, image):
    sampler = DDPMSampler(scheduler, use_tqdm=False)
    noise = torch.randn_like(image)
    out = sampler.sample_prev_step(image, t=10, pred_noise=noise)
    assert out.shape == image.shape


def test_ddpm_sample_final_output_shape(scheduler, image, dummy_pred_noise):
    sampler = DDPMSampler(scheduler)
    out = sampler.sample(image.clone(), dummy_pred_noise, return_intermediates=False)
    assert out.shape == image.shape


def test_ddpm_sample_from_start(scheduler, image, dummy_pred_noise):
    sampler = DDPMSampler(scheduler)
    out = sampler.sample(image.clone(), dummy_pred_noise, return_intermediates=False)
    assert out.shape == image.shape


def test_ddpm_sample_returns_intermediates(scheduler, image, dummy_pred_noise):
    sampler = DDPMSampler(scheduler)
    result = sampler.sample(
        image.clone(), dummy_pred_noise, return_intermediates=True, return_step=2
    )
    assert isinstance(result, list)
    for r in result:
        assert r.shape == image.shape


# DDIMSampler


def test_ddim_sample_prev_step_shape(scheduler, image):
    sampler = DDIMSampler(scheduler, steps=10, eta=0.0)
    noise = torch.randn_like(image)
    out = sampler.sample_prev_step(image, t=5, pred_noise=noise)
    assert out.shape == image.shape


def test_ddim_sample_final_output_shape(scheduler, image, dummy_pred_noise):
    sampler = DDIMSampler(scheduler, steps=10, eta=0.0)
    out = sampler.sample(image.clone(), dummy_pred_noise)
    assert out.shape == image.shape


def test_ddim_sample_from_start(scheduler, image, dummy_pred_noise):
    sampler = DDIMSampler(scheduler, steps=10, eta=0.0)
    out = sampler.sample(image.clone(), dummy_pred_noise, return_intermediates=False)
    assert out.shape == image.shape


def test_ddim_sample_returns_intermediates(scheduler, image, dummy_pred_noise):
    sampler = DDIMSampler(scheduler, steps=10)
    result = sampler.sample(
        image.clone(), dummy_pred_noise, return_intermediates=True, return_step=2
    )
    assert isinstance(result, list)
    for r in result:
        assert r.shape == image.shape


def test_ddim_invalid_step_divisor():
    with pytest.raises(AssertionError, match="steps must be a divisor"):
        DDIMSampler(NoiseScheduler(steps=100), steps=33)


#
# Tests for the encodings
#

# TimeEncoding


def test_time_encoding_output_shape_batch():
    dim = 1280
    model = TimeEncoding(dim=dim)
    t = torch.tensor([0, 1, 2, 3])
    out = model(t)
    assert out.shape == (4, dim)


def test_time_encoding_output_shape_single():
    dim = 1280
    model = TimeEncoding(dim=dim)
    t = 5
    out = model(t)
    assert out.shape == (1, dim)


def test_time_encoding_invalid_negative_t():
    model = TimeEncoding(dim=1280)
    t = -1
    with pytest.raises(AssertionError, match="t must be a non-negative integer"):
        model(t)


def test_time_encoding_invalid_dim_odd():
    with pytest.raises(AssertionError, match="dim must be divisible by 8"):
        TimeEncoding(dim=1281)


def test_time_encoding_get_time_encoding_batch_shape():
    dim = 128
    model = TimeEncoding(dim=dim)
    t = torch.tensor([0, 1, 2])
    enc = model.get_time_encoding(t)
    assert enc.shape == (3, dim // 4)


def test_time_encoding_get_time_encoding_single_shape():
    dim = 128
    model = TimeEncoding(dim=dim)
    t = 7
    enc = model.get_time_encoding(t)
    assert enc.shape == (1, dim // 4)


def test_time_encoding_get_time_encoding_invalid_tensor_dim():
    model = TimeEncoding(dim=64)
    t = torch.tensor([[1, 2]])
    with pytest.raises(AssertionError, match="t must be a 1D tensor"):
        model.get_time_encoding(t)


def test_time_encoding_get_time_encoding_invalid_tensor_negative():
    model = TimeEncoding(dim=64)
    t = torch.tensor([1, -2])
    with pytest.raises(AssertionError, match="t must contain non-negative integers"):
        model.get_time_encoding(t)


#
# Tests for the basic blocks
#

# BasicResidualBlock


@pytest.mark.parametrize(
    "in_channels,out_channels,kernel_size,groups,activation,padding,use_bias,padding_mode,dropout",
    [
        (8, 8, 3, 4, nn.ReLU(), "same", True, "zeros", 0.0),
        (8, 16, 3, 4, nn.GELU(), "same", False, "zeros", 0.1),
        (16, 16, 1, 4, None, 0, True, "zeros", 0.2),
        (4, 4, 5, 2, nn.SiLU(), "same", True, "zeros", 0.0),
    ],
)
def test_basic_residual_block_forward_shapes(
    in_channels,
    out_channels,
    kernel_size,
    groups,
    activation,
    padding,
    use_bias,
    padding_mode,
    dropout,
):
    block = BasicResidualBlock(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        groups=groups,
        activation=activation,
        padding=padding,
        use_bias=use_bias,
        padding_mode=padding_mode,
        dropout=dropout,
    )
    x = torch.randn(2, in_channels, 16, 16)
    y = block(x)
    assert y.shape == (2, out_channels, 16, 16)


def test_basic_residual_block_skip_connection_identity():
    block = BasicResidualBlock(8, 8, 3, 4)
    x = torch.randn(1, 8, 8, 8)
    y = block(x)
    assert y.shape == x.shape


def test_basic_residual_block_skip_connection_conv():
    block = BasicResidualBlock(8, 16, 3, 4)
    x = torch.randn(1, 8, 8, 8)
    y = block(x)
    assert y.shape == (1, 16, 8, 8)


def test_basic_residual_block_forward_raises_on_wrong_dim():
    block = BasicResidualBlock(4, 4, 3, 2)
    x = torch.randn(4, 4, 8)  # 3D input
    with pytest.raises(AssertionError, match="Input tensor must be 4D"):
        block(x)


def test_basic_residual_block_conditioning_shape():
    block = BasicResidualBlock(in_channels=8, out_channels=8, groups=4, d_context=16)
    x = torch.randn(4, 8, 16, 16)
    context = torch.randn(4, 16)  # batch size matches
    y = block(x, context)
    assert y.shape == x.shape


def test_basic_residual_block_conditioning_broadcasting():
    block = BasicResidualBlock(in_channels=4, out_channels=4, groups=4, d_context=10)
    x = torch.randn(2, 4, 8, 8)
    context = torch.randn(2, 10)
    out = block(x, context)
    assert out.shape == x.shape


def test_basic_residual_block_conditioning_wrong_shape():
    block = BasicResidualBlock(in_channels=4, out_channels=4, groups=4, d_context=10)
    x = torch.randn(2, 4, 8, 8)
    context = torch.randn(2, 10, 1)  # wrong shape
    with pytest.raises(AssertionError, match="Conditioning tensor must be 2D"):
        block(x, context)


def test_basic_residual_block_conditioning_none_warns_once():
    block = BasicResidualBlock(in_channels=4, out_channels=4, groups=4, d_context=10)
    x = torch.randn(2, 4, 8, 8)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = block(x, context=None)
        warning_msgs = [str(warn.message) for warn in w]
        assert any("Context tensor is None" in msg for msg in warning_msgs)


# BasicAttentionBlock


@pytest.mark.parametrize(
    "channels,num_heads,groups,dropout",
    [
        (16, 4, 4, 0.0),
        (32, 8, 8, 0.1),
    ],
)
def test_basic_attention_block_output_shape(channels, num_heads, groups, dropout):
    block = BasicAttentionBlock(
        channels, num_heads=num_heads, groups=groups, dropout=dropout
    )
    x = torch.randn(2, channels, 16, 16)
    out = block(x)
    assert out.shape == x.shape


def test_basic_attention_block_forward_raises_on_wrong_dim():
    block = BasicAttentionBlock(8, num_heads=2, groups=4)
    x = torch.randn(2, 8, 16)  # 3D input
    with pytest.raises(AssertionError, match="Input tensor must be 4D"):
        block(x)


def test_basic_attention_block_context_path():
    block = BasicAttentionBlock(16, num_heads=4, groups=4, d_context=32)
    x = torch.randn(2, 16, 8, 8)
    context = torch.randn(2, 64, 32)  # (batch_size, seq_len, d_context)
    out = block(x, context)
    assert out.shape == x.shape


def test_basic_attention_block_context_wrong_shape():
    block = BasicAttentionBlock(16, num_heads=4, groups=4, d_context=32)
    x = torch.randn(2, 16, 8, 8)
    context = torch.randn(2, 64, 32, 1)  # too many dims
    with pytest.raises(AssertionError, match="Context tensor must be 2D"):
        block.apply_context(torch.randn(2, 64, 16), context)


def test_basic_attention_block_context_none_warns_once():
    block = BasicAttentionBlock(16, num_heads=4, groups=4, d_context=32)
    x = torch.randn(2, 16, 8, 8)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = block(x, context=None)
        assert any("Context tensor is None" in str(warning.message) for warning in w)


def test_basic_attention_block_activation_custom():
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

    block = BasicAttentionBlock(16, num_heads=4, groups=4, activation=Swish())
    x = torch.randn(2, 16, 8, 8)
    out = block(x)
    assert out.shape == x.shape


# BasicFeedForwardBlock


@pytest.mark.parametrize("layer_norm", [True, False])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_feedforward_output_shape(layer_norm, dropout):
    dim = 64
    d_ff = 128
    out_features = 32
    batch_size = 4
    seq_len = 10

    ff = BasicFeedForwardBlock(
        dim=dim,
        d_ff=d_ff,
        out_features=out_features,
        activation=nn.GELU(),
        layer_norm=layer_norm,
        dropout=dropout,
    )

    x = torch.randn(batch_size, seq_len, dim)
    y = ff(x)

    assert y.shape == (batch_size, seq_len, out_features), (
        f"Expected output shape {(batch_size, seq_len, out_features)}, got {y.shape}"
    )


def test_feedforward_no_out_features_defaults_to_dim():
    dim = 64
    d_ff = 128
    ff = BasicFeedForwardBlock(dim=dim, d_ff=d_ff)
    x = torch.randn(2, 8, dim)
    y = ff(x)
    assert y.shape[-1] == dim, (
        "Output dim should default to input dim when out_features is None"
    )


def test_feedforward_gradient_flow():
    dim = 32
    d_ff = 64
    ff = BasicFeedForwardBlock(dim=dim, d_ff=d_ff)
    x = torch.randn(2, 8, dim, requires_grad=True)
    y = ff(x)
    y.mean().backward()

    assert x.grad is not None, "Gradients should flow to the input"
    assert any(p.grad is not None for p in ff.parameters()), (
        "All trainable parameters should receive gradients"
    )


def test_feedforward_dropout_effect():
    dim = 32
    d_ff = 64
    dropout = 0.9
    ff = BasicFeedForwardBlock(dim=dim, d_ff=d_ff, dropout=dropout)
    ff.train()  # Enable dropout

    x = torch.randn(10, 5, dim)
    out1 = ff(x)
    out2 = ff(x)

    # Outputs should differ due to dropout
    assert not torch.allclose(out1, out2), (
        "Dropout should cause output variance in training mode"
    )


def test_feedforward_invalid_input_dimension():
    dim = 16
    d_ff = 32
    ff = BasicFeedForwardBlock(dim=dim, d_ff=d_ff)
    x = torch.randn(16)  # Only 1D, invalid input
    with pytest.raises(AssertionError):
        ff(x)


#
# Tests for the functional blocks
#

# SelfAttention


@pytest.mark.parametrize("causal_mask", [False, True])
@pytest.mark.parametrize("return_attn", [False, True])
def test_self_attention_shape(causal_mask, return_attn):
    dim = 64
    model = SelfAttention(dim=dim, num_heads=8)
    x = torch.randn(2, 10, dim)

    output = model(x, causal_mask=causal_mask, return_attn=return_attn)
    if return_attn:
        y, attn = output
        assert y.shape == (2, 10, dim)
        assert attn.shape == (2, 8, 10, 10)
    else:
        assert output.shape == (2, 10, dim)


def test_self_attention_key_padding_mask():
    dim = 32
    model = SelfAttention(dim=dim, num_heads=4)
    x = torch.randn(2, 5, dim)
    mask = torch.tensor(
        [[False, False, True, True, True], [False, True, False, False, True]]
    )
    output = model(x, key_padding_mask=mask)
    assert output.shape == x.shape


# CrossAttention


def test_cross_attention_shape_and_output():
    dim = 64
    cross_dim = 64
    model = CrossAttention(dim=dim, cross_dim=cross_dim, num_heads=8)
    x = torch.randn(2, 6, dim)
    context = torch.randn(2, 10, cross_dim)

    output = model(x, context)
    assert output.shape == (2, 6, dim)


def test_cross_attention_key_padding_mask():
    dim = 32
    cross_dim = 32
    model = CrossAttention(dim=dim, cross_dim=cross_dim, num_heads=4)
    x = torch.randn(2, 4, dim)
    context = torch.randn(2, 6, cross_dim)
    mask = torch.tensor(
        [
            [False, False, True, True, True, True],
            [False, True, False, False, True, False],
        ]
    )

    output = model(x, context, key_padding_mask=mask)
    assert output.shape == (2, 4, dim)


def test_cross_attention_return_attn():
    model = CrossAttention(dim=32, cross_dim=32, num_heads=4)
    x = torch.randn(2, 4, 32)
    context = torch.randn(2, 4, 32)
    out, attn = model(x, context, return_attn=True)
    assert out.shape == (2, 4, 32)
    assert attn.shape == (2, 4, 4, 4)  # (batch, heads, seq_len, cross_seq_len)


def test_cross_attention_causal_warns_once():
    model = CrossAttention(dim=32, cross_dim=32, num_heads=4)
    x = torch.randn(1, 4, 32)
    context = torch.randn(1, 4, 32)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model(x, context, causal_mask=True)

        warning_msgs = [str(warn.message) for warn in w]
        matching = [
            msg
            for msg in warning_msgs
            if "Causal masking is typically only used in self-attention" in msg
        ]

        assert len(matching) == 1


def test_cross_attention_invalid_padding_mask_shape():
    model = CrossAttention(dim=32, cross_dim=32, num_heads=4)
    x = torch.randn(2, 4, 32)
    context = torch.randn(2, 6, 32)
    bad_mask = torch.ones(2, 5, dtype=torch.bool)  # wrong shape

    with pytest.raises(AssertionError):
        model(x, context, key_padding_mask=bad_mask)


def test_cross_attention_invalid_causal_shape():
    model = CrossAttention(dim=32, cross_dim=32, num_heads=4)
    x = torch.randn(2, 4, 32)
    context = torch.randn(2, 6, 32)  # cross_seq_len != seq_len

    with pytest.raises(AssertionError):
        model(x, context, causal_mask=True)


# LayerNorm


def test_layernorm_shape_and_type():
    ln = LayerNorm(features=32)
    x = torch.randn(8, 32)
    out = ln(x)
    assert out.shape == x.shape, "Output shape should match input shape"
    assert isinstance(out, torch.Tensor), "Output should be a torch.Tensor"


def test_layernorm_normalization():
    ln = LayerNorm(features=16)
    x = torch.randn(4, 16)
    out = ln(x)

    mean = out.mean(dim=-1)
    std = out.std(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5), (
        "Mean should be ~0 after normalization"
    )
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4), (
        "Std should be ~1 after normalization"
    )


def test_layernorm_gradient_flow():
    ln = LayerNorm(features=64)
    x = torch.randn(3, 64, requires_grad=True)
    out = ln(x)
    loss = out.sum()
    loss.backward()

    assert x.grad is not None, "Input should receive gradients"
    assert ln.alphas.grad is not None, "alphas should receive gradients"
    assert ln.betas.grad is not None, "betas should receive gradients"


def test_layernorm_multi_dim_input():
    ln = LayerNorm(features=32)
    x = torch.randn(2, 4, 8, 32)  # (batch, H, W, C)
    out = ln(x)
    assert out.shape == x.shape

    mean = out.mean(dim=-1)
    std = out.std(dim=-1, unbiased=False)

    assert torch.allclose(mean, torch.zeros_like(mean), atol=1e-5)
    assert torch.allclose(std, torch.ones_like(std), atol=1e-4)


def test_invalid_input_dimension():
    ln = LayerNorm(features=10)
    x = torch.randn(10)  # Only 1D
    with pytest.raises(AssertionError):
        ln(x)


def test_parameter_shapes():
    features = 24
    ln = LayerNorm(features=features)
    assert ln.alphas.shape == (features,), "alphas parameter shape is incorrect"
    assert ln.betas.shape == (features,), "betas parameter shape is incorrect"


# Upsample


@pytest.mark.parametrize("scale_factor", [2, 3, 0.5])
@pytest.mark.parametrize("mode", ["bilinear", "nearest"])
@pytest.mark.parametrize("align_corners", [True, False])
def test_upsample_shape(scale_factor, mode, align_corners):
    block = Upsample(
        in_channels=8,
        out_channels=8,
        scale_factor=scale_factor,
        mode=mode,
        align_corners=align_corners,
    )
    x = torch.randn(2, 8, 16, 16)
    y = block(x)

    expected_size = int(16 * scale_factor)
    assert y.shape == (2, 8, expected_size, expected_size), (
        f"Expected output shape (2, 8, {expected_size}, {expected_size}), got {y.shape}"
    )


def test_upsample_forward_invalid_dim():
    block = Upsample(in_channels=4, out_channels=4, scale_factor=2)
    x = torch.randn(4, 4, 16)  # invalid shape
    with pytest.raises(RuntimeError):
        _ = block(x)


def test_upsample_gradients():
    block = Upsample(in_channels=4, out_channels=4, scale_factor=2)
    x = torch.randn(2, 4, 8, 8, requires_grad=True)
    y = block(x)
    loss = y.mean()
    loss.backward()
    assert x.grad is not None, "Gradients did not flow back through Upsample block"


def test_upsample_with_different_batch_size():
    block = Upsample(in_channels=6, out_channels=12, scale_factor=1.5, mode="bilinear")
    x = torch.randn(3, 6, 20, 20)
    y = block(x)
    expected_size = int(20 * 1.5)
    assert y.shape == (3, 12, expected_size, expected_size), (
        "Upsample block did not produce expected shape for fractional scale"
    )


# GEGLU


def test_geglu_output_shape():
    module = GEGLU(in_dim=32, inter_dim=64)
    x = torch.randn(4, 10, 32)
    y = module(x)
    assert y.shape == (4, 10, 32), "Output shape mismatch for GEGLU forward pass."


def test_geglu_no_bias():
    module = GEGLU(in_dim=16, inter_dim=32, bias=False)
    x = torch.randn(2, 8, 16)
    y = module(x)
    assert y.shape == (2, 8, 16), "GEGLU without bias gave incorrect output shape."


def test_geglu_gradients():
    module = GEGLU(in_dim=16, inter_dim=8)
    x = torch.randn(5, 16, requires_grad=True)
    y = module(x)
    loss = y.sum()
    loss.backward()
    assert x.grad is not None, "Gradients did not flow through GEGLU."
    assert x.grad.shape == x.shape, "Gradient shape mismatch."


def test_geglu_invalid_input_dim():
    module = GEGLU(in_dim=16, inter_dim=8)
    x = torch.randn(5, 16, 8)  # 3D, but last dim != in_dim
    with pytest.raises(RuntimeError):
        module(x)


def test_geglu_asserts_on_invalid_rank():
    module = GEGLU(in_dim=16, inter_dim=8)
    x = torch.randn(16)  # 1D tensor
    with pytest.raises(AssertionError, match="Input tensor must be at least 2D"):
        module(x)


# MultiWaveletAct


@pytest.mark.parametrize("wavelet_only", [True, False])
@pytest.mark.parametrize("normalize", [True, False])
def test_output_shape(wavelet_only, normalize):
    dim = 64
    batch, seq = 8, 10
    model = MultiWaveletAct(dim=dim, wavelet_only=wavelet_only, normalize=normalize)
    x = torch.randn(batch, seq, dim)
    y = model(x)
    assert y.shape == x.shape, f"Expected output shape {x.shape}, got {y.shape}"


def test_gradient_flow():
    dim = 32
    model = MultiWaveletAct(dim=dim)
    x = torch.randn(4, 16, dim, requires_grad=True)
    y = model(x)
    loss = y.mean()
    loss.backward()

    # Check that gradients flow through freq_scales, log_scales, and weights
    assert model.freq_scales.grad is not None, "No gradient through freq_scales"
    assert model.log_scales.grad is not None, "No gradient through log_scales"
    assert model.weights.grad is not None, "No gradient through weights"
    assert x.grad is not None, "No gradient through input"


def test_input_shape_variants():
    dim = 32
    model = MultiWaveletAct(dim=dim)

    x_2d = torch.randn(5, dim)
    x_3d = torch.randn(3, 7, dim)
    x_4d = torch.randn(2, 3, 4, dim)

    assert model(x_2d).shape == x_2d.shape
    assert model(x_3d).shape == x_3d.shape
    assert model(x_4d).shape == x_4d.shape


def test_invalid_input_dim():
    model = MultiWaveletAct(dim=16)
    x = torch.randn(10)  # 1D input, invalid
    with pytest.raises(AssertionError):
        model(x)


def test_invalid_input_feature_size():
    model = MultiWaveletAct(dim=64)
    x = torch.randn(2, 10, 32)  # Last dim != model.dim
    with pytest.raises(AssertionError):
        model(x)


#
# Tests for the encoder blocks
#

# VAEncoderBlock


# Fixture to locate your actual config file
@pytest.fixture
def default_config_path():
    here = os.path.dirname(__file__)
    config_path = os.path.abspath(
        os.path.join(here, "..", "..", "..", "configs", "default_VAE.yaml")
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    return config_path


def test_encoder_instantiates_with_real_config(default_config_path):
    encoder = VAEncoderBlock(
        in_channels=1,
        config_file=default_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    assert isinstance(encoder.encoder, nn.Sequential)


def test_encoder_forward_output_shape(default_config_path):
    encoder = VAEncoderBlock(
        in_channels=1,
        config_file=default_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    x = torch.randn(2, 1, 32, 32)
    noise = torch.randn(2, 4, 4, 4)
    out = encoder(x, noise)
    assert out.shape == noise.shape


def test_encoder_forward_with_stats(default_config_path):
    encoder = VAEncoderBlock(
        in_channels=1,
        config_file=default_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    x = torch.randn(1, 1, 16, 16)
    noise = torch.randn(1, 4, 2, 2)
    out, (mean, logvar) = encoder(x, noise, return_stats=True)
    assert out.shape == noise.shape
    assert mean.shape == logvar.shape == noise.shape


def test_encoder_invalid_config_path():
    with pytest.raises(AssertionError, match="does not exist"):
        VAEncoderBlock(in_channels=1, config_file="nonexistent.yaml")


def test_encoder_invalid_yaml_extension():
    with pytest.raises(AssertionError, match="must be a YAML file"):
        VAEncoderBlock(in_channels=1, config_file="config.txt")


# UNetEncoderBlock


# Fixture to locate your actual config file
@pytest.fixture
def default_unet_config_path():
    here = os.path.dirname(__file__)
    config_path = os.path.abspath(
        os.path.join(here, "..", "..", "..", "configs", "default_UNet.yaml")
    )
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    return config_path


def test_unet_encoder_instantiates_with_real_config(default_unet_config_path):
    encoder = UNetEncoderBlock(
        latent_dim=4,
        config_file=default_unet_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    assert isinstance(encoder.encoder, nn.Sequential)


def test_unet_encoder_forward_output_shape(default_unet_config_path):
    encoder = UNetEncoderBlock(
        latent_dim=4,
        config_file=default_unet_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    x = torch.randn(2, 4, 32, 32)
    context = torch.randn(2, 10, 768)
    time = torch.randn(2, 1280)
    out, skip = encoder(x, context=context, time=time)
    assert isinstance(out, torch.Tensor)
    assert isinstance(skip, list)


def test_unet_encoder_invalid_config_path():
    with pytest.raises(AssertionError, match="does not exist"):
        UNetEncoderBlock(latent_dim=1, config_file="nonexistent.yaml")


def test_unet_encoder_invalid_yaml_extension():
    with pytest.raises(AssertionError, match="must be a YAML file"):
        UNetEncoderBlock(latent_dim=1, config_file="config.txt")


# VADecoderBlock


def test_decoder_instantiates_with_real_config(default_config_path):
    decoder = VADecoderBlock(
        in_channels=1,
        config_file=default_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    assert isinstance(decoder.decoder, nn.Sequential)


def test_decoder_forward_output_shape(default_config_path):
    decoder = VADecoderBlock(
        in_channels=1,
        config_file=default_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    x = torch.randn(2, 4, 32, 32)
    out = decoder(x)
    assert out.shape == (2, 1, 256, 256)


def test_decoder_invalid_config_path():
    with pytest.raises(AssertionError, match="does not exist"):
        VADecoderBlock(in_channels=1, config_file="nonexistent.yaml")


def test_decoder_invalid_yaml_extension():
    with pytest.raises(AssertionError, match="must be a YAML file"):
        VADecoderBlock(in_channels=1, config_file="config.txt")


# UNetDecoderBlock


def test_unet_decoder_instantiates_with_real_config(default_unet_config_path):
    decoder = UNetDecoderBlock(
        latent_dim=4,
        config_file=default_unet_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    assert isinstance(decoder.decoder, nn.Sequential)


def test_unet_decoder_forward_output_shape(default_unet_config_path):
    decoder = UNetDecoderBlock(
        latent_dim=4,
        config_file=default_unet_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    x = torch.randn(1, 1280, 4, 4)
    context = torch.randn(1, 10, 768)  # Example context tensor
    time = torch.randn(1, 1280)  # Example time tensor
    skip_connections = [
        torch.randn(1, 320, 32, 32),
        torch.randn(1, 320, 32, 32),
        torch.randn(1, 320, 32, 32),
        torch.randn(1, 320, 16, 16),
        torch.randn(1, 640, 16, 16),
        torch.randn(1, 640, 16, 16),
        torch.randn(1, 640, 8, 8),
        torch.randn(1, 1280, 8, 8),
        torch.randn(1, 1280, 8, 8),
        torch.randn(1, 1280, 4, 4),
        torch.randn(1, 1280, 4, 4),
        torch.randn(1, 1280, 4, 4),
    ]  # Example skip connections
    out = decoder(x, context=context, time=time, skip_connections=skip_connections)
    assert out.shape == (1, 4, 32, 32)


def test_unet_decoder_invalid_config_path():
    with pytest.raises(AssertionError, match="does not exist"):
        VADecoderBlock(in_channels=1, config_file="nonexistent.yaml")


def test_unet_decoder_invalid_yaml_extension():
    with pytest.raises(AssertionError, match="must be a YAML file"):
        VADecoderBlock(in_channels=1, config_file="config.txt")


# VAE


def test_vae_initialization(default_config_path):
    vae = VAE(
        input_shape=(1, 32, 32),
        config_file=default_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    assert isinstance(vae.encoder, nn.Module)
    assert isinstance(vae.decoder, nn.Module)


def test_vae_forward_basic(default_config_path):
    vae = VAE((1, 32, 32), default_config_path, BasicResidualBlock, BasicAttentionBlock)
    x = torch.randn(4, 1, 32, 32)
    noise = torch.randn(4, 4, 4, 4)  # Match latent dim
    out = vae(x, noise)
    assert isinstance(out, torch.Tensor)
    assert out.shape == noise.shape


def test_vae_forward_with_stats(default_config_path):
    vae = VAE((1, 32, 32), default_config_path, BasicResidualBlock, BasicAttentionBlock)
    x = torch.randn(2, 1, 32, 32)
    noise = torch.randn(2, 4, 4, 4)
    out, (mean, logvar) = vae(x, noise, return_stats=True)
    assert out.shape == noise.shape
    assert mean.shape == noise.shape
    assert logvar.shape == noise.shape


def test_vae_forward_with_rec(default_config_path):
    vae = VAE((1, 32, 32), default_config_path, BasicResidualBlock, BasicAttentionBlock)
    x = torch.randn(2, 1, 32, 32)
    noise = torch.randn(2, 4, 4, 4)
    out, rec = vae(x, noise, return_rec=True)
    assert out.shape == noise.shape
    assert rec.shape[0] == x.shape[0]
    assert rec.shape[2:] == x.shape[2:]


def test_vae_forward_with_all(default_config_path):
    vae = VAE((1, 16, 16), default_config_path, BasicResidualBlock, BasicAttentionBlock)
    x = torch.randn(1, 1, 16, 16)
    noise = torch.randn(1, 4, 2, 2)
    out, (mean, logvar), rec = vae(x, noise, return_stats=True, return_rec=True)
    assert out.shape == noise.shape
    assert mean.shape == logvar.shape == noise.shape
    assert rec.shape == x.shape


# UNet


def test_unet_initialization(default_unet_config_path):
    unet = UNet(
        latent_dim=4,
        config_file=default_unet_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )
    assert isinstance(unet.encoder, nn.Module)
    assert isinstance(unet.decoder, nn.Module)


def test_unet_forward_output_shape(default_unet_config_path):
    unet = UNet(
        latent_dim=4,
        config_file=default_unet_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
    )

    x = torch.randn(2, 4, 32, 32)
    context = torch.randn(2, 10, 768)  # Example context tensor
    time = torch.randn(2, 1280)  # Example time tensor
    out = unet(x, context=context, time=time)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4, 32, 32)


# Diffusion


def test_diffusion_initialization(default_unet_config_path):
    diffusion = Diffusion(
        latent_dim=4,
        config_file=default_unet_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
        TimeEncoder=TimeEncoding,
    )
    assert isinstance(diffusion.unet, UNet)
    assert isinstance(diffusion.time_encoder, TimeEncoding)


def test_diffusion_forward_output_shape(default_unet_config_path):
    diffusion = Diffusion(
        latent_dim=4,
        config_file=default_unet_config_path,
        ResidualBlock=BasicResidualBlock,
        AttentionBlock=BasicAttentionBlock,
        TimeEncoder=TimeEncoding,
    )

    x = torch.randn(2, 4, 32, 32)
    context = torch.randn(2, 10, 768)  # Example context tensor
    time = torch.tensor([756, 123])  # Example time tensor
    out = diffusion(x, t=time, context=context)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (2, 4, 32, 32)
