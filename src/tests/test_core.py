# tests/test_core.py

import pytest
import torch

#
# Tests for the schedulers
#

from core.schedulers import NoiseScheduler
from core.samplers import DDPMSampler, DDIMSampler

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