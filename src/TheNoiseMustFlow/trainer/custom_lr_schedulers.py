"""
custom_lr_schedulers.py

The module implements different learning rate schedulers that can be used during
the training of deep learning models.
"""


from __future__ import annotations
from typing import Optional

import math
from torch.optim import Optimizer, lr_scheduler


class CosineLRScheduler(lr_scheduler._LRScheduler):
    """
    Cosine learning rate scheduler.

    This scheduler adjusts the learning rate using a cosine annealing strategy
    with a warmup phase.
    """

    def __init__(self, optimizer: Optimizer, base_lr: float, total_epochs: int, warmup_epochs: int = 0, 
                 min_lr: float = 0.0, last_epoch: int = -1):
        """
        Initializes the CosineLRScheduler.

        Args:
            optimizer: Optimizer for which to adjust the learning rate.
            base_lr: Base learning rate to use after warmup.
            total_epochs: Total number of epochs for the training.
            warmup_epochs: Number of warmup epochs before cosine annealing starts.
            min_lr: Minimum learning rate after annealing.
            last_epoch: The index of the last epoch.
                Default is -1, which means the scheduler starts from the beginning.
        """
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup from min_lr to base_lr
            warmup_ratio = (self.last_epoch + 1) / self.warmup_epochs
            return [
                self.min_lr + warmup_ratio * (self.base_lr - self.min_lr)
                for _ in self.base_lrs
            ]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(1, self.total_epochs - self.warmup_epochs)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            return [
                self.min_lr + cosine_factor * (self.base_lr - self.min_lr)
                for _ in self.base_lrs
            ]
        
    def step(self, metrics: Optional[float] = None):
        """
        Override step to accept a metrics argument for compatibility with several schedulers.
        Args:
            metrics: Optional metric to use for step adjustment.
                Unused in this scheduler.
        """
        return super().step()