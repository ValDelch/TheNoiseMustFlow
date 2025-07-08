# tests/test_trainer.py

import pytest
import torch
from torch.optim import SGD

from TheNoiseMustFlow.trainer.losses import (
    mse_loss,
    huber_noise_loss,
    snr_weighted_mse_loss,
    kl_divergence,
    VAE_loss,
    cross_entropy,
)
from TheNoiseMustFlow.trainer.metrics import PSNR, SSIM
from TheNoiseMustFlow.trainer.custom_lr_schedulers import CosineLRScheduler

#
# Tests for the losses
#


@pytest.fixture
def tensors():
    B, C, H, W = 4, 3, 32, 32
    x = torch.randn(B, C, H, W)
    x_recon = torch.randn_like(x)
    mu = torch.randn(B, 4, H // 8, W // 8)  # Example shape for latent space
    logvar = torch.randn_like(mu)
    snr = torch.linspace(1e8, 1e-5, B)
    return x, x_recon, mu, logvar, snr


def test_mse_loss(tensors):
    x, x_recon, *_ = tensors
    loss = mse_loss(x, x_recon)
    assert loss.ndim == 0 and loss >= 0


def test_huber_noise_loss(tensors):
    x, x_recon, *_ = tensors
    loss = huber_noise_loss(x, x_recon, delta=1.0)
    assert loss.ndim == 0 and loss >= 0


def test_snr_weighted_mse_loss(tensors):
    x, x_recon, _, _, snr = tensors
    loss = snr_weighted_mse_loss(x, x_recon, snr, gamma=5.0)
    assert loss.ndim == 0 and loss >= 0


def test_kl_divergence(tensors):
    _, _, mu, logvar, _ = tensors
    kld = kl_divergence(mu, logvar)
    assert kld.ndim == 0 and kld >= 0


def test_vae_loss(tensors):
    x, x_recon, mu, logvar, _ = tensors
    loss = VAE_loss(x, x_recon, mu, logvar)
    assert loss.ndim == 0 and loss >= 0


def test_cross_entropy_loss():
    logits = torch.randn(4, 1, 32, 32)
    targets = torch.randint(0, 1, (4, 1, 32, 32))
    loss = cross_entropy(logits, targets)
    assert loss.ndim == 0 and loss >= 0


#
# Tests for the metrics
#

# PSNR


def test_psnr_same_images():
    x = torch.ones(1, 3, 64, 64)
    y = torch.ones(1, 3, 64, 64)
    psnr = PSNR(x, y)
    assert psnr == float("inf"), "PSNR should be infinite for identical images"


def test_psnr_different_images():
    x = torch.zeros(1, 3, 64, 64)
    y = torch.ones(1, 3, 64, 64) + (0.05 * torch.randn_like(x))
    y = y.clamp(0, 1)
    psnr = PSNR(x, y)
    assert psnr > 0 and psnr < 10, "PSNR should be low for completely different images"


def test_psnr_assert_shape_mismatch():
    x = torch.ones(1, 3, 64, 64)
    y = torch.ones(1, 3, 32, 32)
    with pytest.raises(AssertionError):
        PSNR(x, y)


def test_batch_psnr():
    x = torch.rand(8, 3, 64, 64)
    y = x + 0.05 * torch.randn_like(x)
    y = y.clamp(0, 1)
    psnr = PSNR(x, y)
    assert isinstance(psnr, float)
    assert psnr > 10, "PSNR should be >10 for similar images"


# SSIM


def test_ssim_same_images():
    x = torch.rand(1, 3, 64, 64)
    ssim = SSIM(x, x.clone())
    assert abs(ssim - 1.0) < 1e-5, "SSIM should be 1 for identical images"


def test_ssim_different_images():
    x = torch.zeros(1, 3, 64, 64)
    y = torch.ones(1, 3, 64, 64)
    ssim = SSIM(x, y)
    assert ssim < 0.1, "SSIM should be low for completely different images"


def test_ssim_assert_shape_mismatch():
    x = torch.rand(1, 3, 64, 64)
    y = torch.rand(1, 3, 32, 32)
    with pytest.raises(AssertionError):
        SSIM(x, y)


def test_batch_ssim():
    x = torch.rand(8, 3, 64, 64)
    y = x + 0.05 * torch.randn_like(x)
    ssim = SSIM(x.clamp(0, 1), y.clamp(0, 1))
    assert isinstance(ssim, float)
    assert 0.0 < ssim < 1.0


#
# Tests fir the lr schedulers
#

# CosineLRScheduler


@pytest.fixture
def dummy_optimizer():
    model = torch.nn.Linear(10, 1)
    return SGD(model.parameters(), lr=0.1)


def test_cosine_annealing_phase(dummy_optimizer):
    scheduler = CosineLRScheduler(
        optimizer=dummy_optimizer,
        base_lr=0.1,
        min_lr=0.01,
        total_epochs=10,
        warmup_epochs=2,
    )

    # warmup
    for _ in range(2):
        dummy_optimizer.step()
        scheduler.step()

    lrs = []
    for _ in range(8):
        dummy_optimizer.step()
        scheduler.step()
        lrs.append(scheduler.get_last_lr()[0])

    # Ensure LR decreases and stays above min_lr
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1]
        assert lrs[i] >= 0.01


def test_learning_rate_bounds(dummy_optimizer):
    scheduler = CosineLRScheduler(
        optimizer=dummy_optimizer,
        base_lr=0.05,
        min_lr=0.01,
        total_epochs=4,
        warmup_epochs=0,
    )

    for _ in range(10):  # go beyond total_epochs
        dummy_optimizer.step()
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        assert 0.01 <= lr <= 0.05


def test_step_with_metrics_argument(dummy_optimizer):
    scheduler = CosineLRScheduler(
        optimizer=dummy_optimizer,
        base_lr=0.1,
        min_lr=0.01,
        total_epochs=5,
        warmup_epochs=1,
    )

    dummy_optimizer.step()
    try:
        scheduler.step(metrics=0.5)  # should be ignored
    except Exception as e:
        pytest.fail(f"Scheduler should ignore metrics argument, but failed: {e}")
