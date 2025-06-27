# tests/test_core.py

import pytest
import torch
from torch import nn

#
# Tests for the schedulers
#

from core.schedulers import NoiseScheduler
from core.samplers import DDPMSampler, DDIMSampler
from core.basic_components.encodings import TimeEncoding
from core.basic_components.basic_blocks import BasicResidualBlock, BasicAttentionBlock

@pytest.mark.parametrize("schedule", ["linear", "cosine", "quadratic", "sigmoid", "geometric"])
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
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(3, 32, 32)
    t = 10
    noisy_x = scheduler.add_noise_step(x, t)
    assert noisy_x.shape == x.shape
    assert not torch.equal(noisy_x, x)

def test_noise_add_step_batch():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(4, 3, 32, 32)
    t = torch.tensor([0, 1, 2, 3])
    noisy_x = scheduler.add_noise_step(x, t)
    assert noisy_x.shape == x.shape

def test_noise_add_cumulative_single():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(3, 32, 32)
    t = 5
    noisy_x = scheduler.add_noise_cumulative(x, t)
    assert noisy_x.shape == x.shape

def test_noise_add_cumulative_batch():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(4, 3, 32, 32)
    t = torch.tensor([0, 1, 2, 3])
    noisy_x = scheduler.add_noise_cumulative(x, t)
    assert noisy_x.shape == x.shape

def test_invalid_t_out_of_bounds():
    scheduler = NoiseScheduler(steps=10)
    x = torch.ones(2, 3, 32, 32)
    t = 100  # out of bounds
    with pytest.raises(AssertionError):
        scheduler.add_noise_step(x, t)

def test_invalid_t_mismatch_batch_size():
    scheduler = NoiseScheduler(steps=10)
    x = torch.ones(2, 3, 32, 32)
    t = torch.tensor([1])  # batch size mismatch
    with pytest.raises(AssertionError):
        scheduler.add_noise_step(x, t)

def test_invalid_t_type():
    scheduler = NoiseScheduler(steps=10)
    x = torch.ones(2, 3, 32, 32)
    with pytest.raises(TypeError):
        scheduler.add_noise_step(x, t="invalid")

def test_cosine_schedule_betas_less_than_1():
    scheduler = NoiseScheduler(steps=10, schedule="cosine")
    assert torch.all(scheduler.betas >= 0)
    assert torch.all(scheduler.betas <= 0.999)

def test_broadcasting_step_batch_scalar_t():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(4, 3, 32, 32)
    t = 5  # scalar t, should broadcast to all
    out = scheduler.add_noise_step(x, t)
    assert out.shape == x.shape

def test_broadcasting_cumulative_batch_scalar_t():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(4, 3, 32, 32)
    t = 5
    out = scheduler.add_noise_cumulative(x, t)
    assert out.shape == x.shape

def test_3d_input_with_batch_t_raises():
    scheduler = NoiseScheduler(steps=100)
    x = torch.ones(3, 32, 32)  # 3D input (no batch)
    t = torch.tensor([1, 2])   # Batched t
    with pytest.raises(AssertionError, match="Batch size of t must match"):
        scheduler.add_noise_step(x, t)

def test_add_noise_step_with_external_noise():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(3, 32, 32)
    t = 10
    noise = torch.randn_like(x)
    out = scheduler.add_noise_step(x, t, noise=noise)
    expected = torch.sqrt(1 - scheduler.betas[t]) * x + torch.sqrt(scheduler.betas[t]) * noise
    assert torch.allclose(out, expected)

def test_add_noise_step_batch_with_external_noise():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(2, 3, 32, 32)
    t = torch.tensor([5, 10])
    noise = torch.randn_like(x)
    out = scheduler.add_noise_step(x, t, noise=noise)
    betas = scheduler.betas[t].view(-1, 1, 1, 1)
    expected = torch.sqrt(1 - betas) * x + torch.sqrt(betas) * noise
    assert torch.allclose(out, expected)

def test_add_noise_step_wrong_noise_shape_raises():
    scheduler = NoiseScheduler(steps=100)
    x = torch.ones(3, 32, 32)
    t = 10
    noise = torch.randn(1, 3, 32, 32)  # mismatched shape
    with pytest.raises(AssertionError, match="must have the same shape"):
        scheduler.add_noise_step(x, t, noise=noise)

def test_add_noise_cumulative_with_external_noise():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(3, 32, 32)
    t = 10
    noise = torch.randn_like(x)
    out = scheduler.add_noise_cumulative(x, t, noise=noise)
    expected = torch.sqrt(scheduler.alphas_cumprod[t]) * x + torch.sqrt(1 - scheduler.alphas_cumprod[t]) * noise
    assert torch.allclose(out, expected)

def test_add_noise_cumulative_batch_with_external_noise():
    scheduler = NoiseScheduler(steps=100, schedule="linear", seed=42)
    x = torch.ones(2, 3, 32, 32)
    t = torch.tensor([10, 20])
    noise = torch.randn_like(x)
    alphas = scheduler.alphas_cumprod[t].view(-1, 1, 1, 1)
    expected = torch.sqrt(alphas) * x + torch.sqrt(1 - alphas) * noise
    out = scheduler.add_noise_cumulative(x, t, noise=noise)
    assert torch.allclose(out, expected)

def test_add_noise_cumulative_wrong_noise_shape_raises():
    scheduler = NoiseScheduler(steps=100)
    x = torch.ones(3, 32, 32)
    t = 10
    noise = torch.randn(1, 3, 32, 32)
    with pytest.raises(AssertionError, match="must have the same shape"):
        scheduler.add_noise_cumulative(x, t, noise=noise)

def test_linear_noise_one_image_full_process():
    scheduler = NoiseScheduler(steps=10, schedule="linear", seed=42)
    x = torch.ones(3, 32, 32)
    for i in range(scheduler.steps):
        x = scheduler.add_noise_step(x, i)
    
def test_cosine_noise_one_image_full_process():
    scheduler = NoiseScheduler(steps=10, schedule="cosine", seed=42)
    x = torch.ones(3, 32, 32)
    for i in range(scheduler.steps):
        x = scheduler.add_noise_step(x, i)

#
# Tests for the samplers
#

@pytest.fixture
def scheduler():
    return NoiseScheduler(steps=100, betas=(1e-4, 0.02), schedule="linear", seed=123)

@pytest.fixture
def image():
    return torch.ones(3, 32, 32)

@pytest.fixture
def batch_image():
    return torch.ones(2, 3, 32, 32)

@pytest.fixture
def dummy_pred_noise():
    return lambda x, t: torch.zeros_like(x)

def test_ddpm_sample_prev_step_shape(scheduler, image):
    sampler = DDPMSampler(scheduler)
    noise = torch.randn_like(image)
    out = sampler.sample_prev_step(image, t=10, pred_noise=noise)
    assert out.shape == image.shape

def test_ddpm_sample_final_output_shape(scheduler, image, dummy_pred_noise):
    sampler = DDPMSampler(scheduler)
    out = sampler.sample(image.clone(), dummy_pred_noise, t=10, return_intermediates=False)
    assert out.shape == image.shape

def test_ddpm_sample_from_start(scheduler, image, dummy_pred_noise):
    sampler = DDPMSampler(scheduler)
    out = sampler.sample(image.clone(), dummy_pred_noise, return_intermediates=False)
    assert out.shape == image.shape

def test_ddpm_sample_returns_intermediates(scheduler, image, dummy_pred_noise):
    sampler = DDPMSampler(scheduler)
    result = sampler.sample(image.clone(), dummy_pred_noise, t=10, return_intermediates=True, return_step=2)
    assert isinstance(result, list)
    for r in result:
        assert r.shape == image.shape

def test_ddim_sample_prev_step_shape(scheduler, image):
    sampler = DDIMSampler(scheduler, steps=10, eta=0.0)
    noise = torch.randn_like(image)
    out = sampler.sample_prev_step(image, t=5, pred_noise=noise)
    assert out.shape == image.shape

def test_ddim_sample_final_output_shape(scheduler, image, dummy_pred_noise):
    sampler = DDIMSampler(scheduler, steps=10, eta=0.0)
    out = sampler.sample(image.clone(), dummy_pred_noise, t=9)
    assert out.shape == image.shape

def test_ddim_sample_from_start(scheduler, image, dummy_pred_noise):
    sampler = DDIMSampler(scheduler, steps=10, eta=0.0)
    out = sampler.sample(image.clone(), dummy_pred_noise, return_intermediates=False)
    assert out.shape == image.shape

def test_ddim_sample_returns_intermediates(scheduler, image, dummy_pred_noise):
    sampler = DDIMSampler(scheduler, steps=10)
    result = sampler.sample(image.clone(), dummy_pred_noise, t=9, return_intermediates=True, return_step=2)
    assert isinstance(result, list)
    for r in result:
        assert r.shape == image.shape

def test_ddim_invalid_step_divisor():
    with pytest.raises(AssertionError, match="steps must be a divisor"):
        DDIMSampler(NoiseScheduler(steps=100), steps=33)

#
# Tests for the encodings
#

def test_time_encoding_output_shape_batch():
    dim = 320
    model = TimeEncoding(dim=dim)
    t = torch.tensor([0, 1, 2, 3])
    out = model(t)
    assert out.shape == (4, 4 * dim)

def test_time_encoding_output_shape_single():
    dim = 320
    model = TimeEncoding(dim=dim)
    t = 5
    out = model(t)
    assert out.shape == (1, 4 * dim)

def test_time_encoding_invalid_negative_t():
    model = TimeEncoding(dim=320)
    t = -1
    with pytest.raises(AssertionError, match="t must be a non-negative integer"):
        model(t)

def test_time_encoding_invalid_dim_odd():
    with pytest.raises(AssertionError, match="dim must be an even integer"):
        TimeEncoding(dim=321)

def test_time_encoding_get_time_encoding_batch_shape():
    dim = 128
    model = TimeEncoding(dim=dim)
    t = torch.tensor([0, 1, 2])
    enc = model.get_time_encoding(t)
    assert enc.shape == (3, dim)

def test_time_encoding_get_time_encoding_single_shape():
    dim = 128
    model = TimeEncoding(dim=dim)
    t = 7
    enc = model.get_time_encoding(t)
    assert enc.shape == (1, dim)

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

@pytest.mark.parametrize("in_channels,out_channels,kernel_size,groups,activation,padding,use_bias,padding_mode,dropout", [
    (8, 8, 3, 4, nn.ReLU(), 'same', True, 'zeros', 0.0),
    (8, 16, 3, 4, nn.GELU(), 'same', False, 'zeros', 0.1),
    (16, 16, 1, 4, None, 0, True, 'zeros', 0.2),
    (4, 4, 5, 2, nn.SiLU(), 'same', True, 'zeros', 0.0),
])
def test_basic_residual_block_forward_shapes(
    in_channels, out_channels, kernel_size, groups, activation, padding, use_bias, padding_mode, dropout
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

@pytest.mark.parametrize("in_channels,num_heads,groups,dropout", [
    (16, 4, 8, 0.0),
    (16, 8, 16, 0.1),
    (32, 16, 8, 0.2),
])
def test_basic_attention_block_forward_shapes(in_channels, num_heads, groups, dropout):
    block = BasicAttentionBlock(in_channels=in_channels, num_heads=num_heads, groups=groups, dropout=dropout)
    x = torch.randn(2, in_channels, 16, 16)
    y = block(x)
    assert y.shape == (2, in_channels, 16, 16)

def test_basic_attention_block_forward_raises_on_wrong_dim():
    block = BasicAttentionBlock(4, 2, 2)
    x = torch.randn(4, 4, 8)  # 3D input
    with pytest.raises(AssertionError, match="Input tensor must be 4D"):
        block(x)