"""
metrics.py

This module implements various metrics commonly used to test diffusion models.
"""


from __future__ import annotations

import warnings

import torch
from torch import nn
from torch.nn import functional as F

from torchmetrics.image.ssim import StructuralSimilarityIndexMeasure


def FID(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    FID

    Computes the Fréchet Inception Distance (FID) between two sets of images.

    Recommended for class-conditional image generation or general diffusion
    image generation tasks.
    
    Args:
        x: Tensor of shape (B, C, H, W) representing the first batch of images.
        y: Tensor of shape (B, C, H, W) representing the second batch of images.
        eps: Small value to avoid division by zero in the square root.

    Returns:
        A float representing the mean FID value for the batch.
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"
    raise NotImplementedError("FID computation is not implemented yet.")

def PSNR(x: torch.Tensor, y: torch.Tensor, max_val: float = 1.0) -> float:
    """
    PSNR

    Computes the Peak Signal-to-Noise Ratio (PSNR) between two images.
    
    Recommended for image denoising/inpainting tasks.

    Args:
        x: Tensor of shape (B, C, H, W) representing the first batch of images.
        y: Tensor of shape (B, C, H, W) representing the second batch of images.
        max_val: Maximum possible pixel value of the images. Default is 1.0.

    Returns:
        A float representing the mean PSNR value for the batch.
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"

    if x.min() < 0 or x.max() > max_val or y.min() < 0 or y.max() > max_val:
        warnings.warn("Input tensors contain values outside the expected range [0, max_val]. \
                      Clamping to [0, max_val].")
        x = x.clamp(0, max_val)
        y = y.clamp(0, max_val)
    
    mse = F.mse_loss(x, y, reduction='mean')
    if mse == 0:
        return float('inf') # Infinite PSNR if there is no noise
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()
    
def SSIM(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    SSIM

    Computes the Structural Similarity Index (SSIM) between two images.
    
    Recommended for image denoising/inpainting tasks.

    Args:
        x: Tensor of shape (B, C, H, W) representing the first batch of images.
        y: Tensor of shape (B, C, H, W) representing the second batch of images.

    Returns:
        A float representing the mean SSIM value for the batch.
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"
    
    ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0)
    return ssim_metric(x, y).mean().item()

def LPIPS(x: torch.Tensor, y: torch.Tensor) -> float:
    """
    LPIPS

    Computes the Learned Perceptual Image Patch Similarity (LPIPS) between two images.
    
    Recommended for all types of image generation tasks.

    Args:
        x: Tensor of shape (B, C, H, W) representing the first batch of images.
        y: Tensor of shape (B, C, H, W) representing the second batch of images.

    Returns:
        A float representing the mean LPIPS value for the batch.
    """
    assert x.shape == y.shape, "Input tensors must have the same shape"
    raise NotImplementedError("LPIPS computation is not implemented yet.")

def inception_score(x: torch.Tensor, inception_model: nn.Module, 
                    splits: int = 10) -> float:
    """
    Inception Score

    Computes the Inception Score for a batch of images using a pre-trained Inception model.
    
    Generally not recommended as inception score is biased on the Inception model's
    training data and may not generalize well to other datasets. Also, it is not
    considering real images in the dataset.

    Args:
        x: Tensor of shape (B, C, H, W) representing the batch of images.
        inception_model: Pre-trained Inception model for feature extraction.
        splits: Number of splits to use for computing the score.

    Returns:
        A float representing the mean Inception Score for the batch.
    """
    assert x.dim() == 4, "Input tensor must be of shape (B, C, H, W)"
    raise NotImplementedError("Inception Score computation is not implemented yet.")
    
    