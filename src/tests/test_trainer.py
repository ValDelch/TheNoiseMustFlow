# tests/test_trainer.py

import pytest
import torch

from trainer.losses import (
    mse_loss, huber_noise_loss, snr_weighted_mse_loss, kl_divergence, VAE_loss, cross_entropy
)

#
# Tests for the losses
#

@pytest.fixture
def tensors():
    B, C, H, W = 4, 3, 32, 32
    x = torch.randn(B, C, H, W)
    x_recon = torch.randn_like(x)
    mu = torch.randn(B, 4, H // 8, W // 8) # Example shape for latent space
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