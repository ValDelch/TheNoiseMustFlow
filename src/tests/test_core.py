# tests/test_core.py

import pytest
import torch

#
# Tests for the schedulers
#

from core.schedulers import NoiseScheduler

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