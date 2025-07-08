"""
losses.py

This module implements various loss functions used in the training of diffusion models.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def mse_loss(
    x: torch.Tensor, x_recon: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """
    mse_loss

    Computes the reconstruction loss between the original and
    reconstructed images as the mean squared error (MSE).

    Args:
        x: Original image tensor.
        x_recon: Reconstructed image tensor.
        reduction: Specifies the reduction to apply to the output:
            can be 'mean' or 'sum'. Default is 'mean'.

    Returns:
        A tensor containing the computed MSE loss.
    """
    assert x.shape == x_recon.shape, "Input tensors must have the same shape"
    return F.mse_loss(x_recon, x, reduction=reduction)


def huber_noise_loss(
    x: torch.Tensor, x_recon: torch.Tensor, delta: float = 1.0, reduction: str = "mean"
) -> torch.Tensor:
    """
    huber_noise_loss

    Computes the Huber loss between the original and reconstructed images.
    This loss is less sensitive to outliers than MSE and is useful for noise estimation.

    Args:
        x: Original image tensor.
        x_recon: Reconstructed image tensor.
        delta: The point where the loss function changes from quadratic to linear. Default is 1.0.
        reduction: Specifies the reduction to apply to the output:
            can be 'mean' or 'sum'. Default is 'mean'.

    Returns:
        A tensor containing the computed Huber loss.
    """
    assert x.shape == x_recon.shape, "Input tensors must have the same shape"
    return F.smooth_l1_loss(x_recon, x, beta=delta, reduction=reduction)


def snr_weighted_mse_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    snr: torch.Tensor,
    gamma: float = 5.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    snr_weighted_mse_loss

    Computes the SNR-weighted MSE loss between the original and reconstructed images.
    The loss is scaled by the SNR value, which can help in focusing on more significant
    steps in the denoising process.

    Args:
        x: Original image tensor.
        x_recon: Reconstructed image tensor.
        snr: Signal-to-noise ratio tensor, obtained from the noise scheduler.
        gamma: Scaling factor for the SNR. Default is 5.0. Should be greater than 1.
        reduction: Specifies the reduction to apply to the output:
            can be 'mean' or 'sum'. Default is 'mean'.

    Returns:
        A tensor containing the computed SNR-weighted MSE loss.
    """
    assert x.shape == x_recon.shape, "Input tensors must have the same shape"
    assert snr.dim() == 1, "SNR must be a 1D tensor"
    assert snr.shape[0] == x.shape[0], "SNR must match the batch size of x and x_recon"
    assert gamma >= 1.0, "gamma must be greater than 1"

    weights = gamma / (snr + gamma)
    mse = F.mse_loss(x_recon, x, reduction="none").mean(dim=[1, 2, 3])

    if reduction == "mean":
        return (weights * mse).mean()
    elif reduction == "sum":
        return (weights * mse).sum()
    else:
        raise ValueError("Reduction must be 'mean' or 'sum'")


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    kl_divergence

    Computes the Kullback-Leibler divergence loss between the
    learned distribution and a standard normal distribution.

    Args:
        mu: Mean of the learned distribution.
        logvar: Log variance of the learned distribution.

    Returns:
        A tensor containing the computed KL divergence loss.
    """
    assert mu.shape == logvar.shape, "mu and logvar must have the same shape"
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())


def VAE_loss(
    x: torch.Tensor,
    x_recon: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    beta: float = 1.0,
) -> torch.Tensor:
    """
    VAE_loss

    Computes the total loss for a Variational Autoencoder (VAE),
    which is the sum of the reconstruction loss and the KL divergence.

    Args:
        x: Original image tensor.
        x_recon: Reconstructed image tensor.
        mu: Mean of the learned distribution.
        logvar: Log variance of the learned distribution.
        beta: Weighting factor for the KL divergence term. Default is 1.0.

    Returns:
        A tensor containing the total VAE loss.
    """
    recon_loss = mse_loss(x, x_recon, reduction="sum")
    kld_loss = kl_divergence(mu, logvar)
    return recon_loss + beta * kld_loss


def cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
    weight: float = 1.0,
) -> torch.Tensor:
    """
    cross_entropy

    Computes the cross-entropy loss between logits and target labels.
    Useful for tasks where images can be binarized into 2 classes (e.g., foreground vs background).

    Args:
        logits: Logits tensor (unnormalized scores).
        targets: Target labels tensor (ground truth).
        reduction: Specifies the reduction to apply to the output:
            can be 'mean' or 'sum'. Default is 'mean'.
        weight: Optional float for the positive class weight.

    Returns:
        A tensor containing the computed cross-entropy loss.
    """
    assert logits.shape == targets.shape, "Logits and targets must have the same shape"
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.ones(1, 1, 1, device=logits.device) * weight,
        reduction=reduction,
    )
    return loss_fn(logits, targets.float())
