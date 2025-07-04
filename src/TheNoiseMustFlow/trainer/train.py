"""
train.py

This module implements the training loops for the VAE and diffusion model.
"""


from __future__ import annotations
from typing import Optional
from contextlib import nullcontext

import os
from tqdm import tqdm

import torch
from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter

from TheNoiseMustFlow.core.models import VAE, Diffusion

from TheNoiseMustFlow.core.schedulers import NoiseScheduler
from TheNoiseMustFlow.core.samplers import DDPMSampler, DDIMSampler

import matplotlib.pyplot as plt


def print_epoch_summary(epoch: int, epochs: int, current_lr: float, 
                        train_losses: dict[str, float],
                        test_losses: dict[str, float], best_loss: float) -> None:
    """
    Utility function to print the summary of the training epoch.
    """
    box_width = 60
    line = "═" * box_width
    content_width = box_width - 2  # account for border characters

    def format_line(content: str = "") -> str:
        return f"║ {content:<{content_width - 1}}║"

    def format_kv(key: str, value: str) -> str:
        if len(key) > 20:
            key = key[:20] + "..."
        text = f"{key:<25}: {value:>28}"
        return format_line(text[:content_width - 1])  # Trim if necessary

    print(f"\n╔{line[:-2]}╗")
    print(format_line(f"Epoch {epoch + 1}/{epochs}"))
    print(f"╠{line[:-2]}╣")
    print(format_kv("Learning Rate", f"{current_lr:.2e}"))
    print(format_line())
    print(format_kv("Total Train Loss", f"{sum(train_losses.values()):.3e}"))
    for k, v in train_losses.items():
        print(format_kv(f"• Train {k}", f"{v:.6f}"))
    print(format_line())
    print(format_kv("Total Test Loss", f"{sum(test_losses.values()):.3e}"))
    for k, v in test_losses.items():
        print(format_kv(f"• Test {k}", f"{v:.6f}"))
    print(format_line())
    print(format_kv("Best Test Loss", f"{best_loss:.3e}"))
    print(f"╚{line[:-2]}╝\n")

def write_tensorboard_summary(writer: SummaryWriter, epoch: int, current_lr: float,
                              train_losses: dict[str, float], test_losses: dict[str, float]) -> None:
    """
    Utility function to write the training and testing losses to TensorBoard.
    """
    writer.add_scalar('Train', sum(train_losses.values()), epoch)
    writer.add_scalar('Test', sum(test_losses.values()), epoch)
    writer.add_scalar('Learning Rate', current_lr, epoch)

    for k, v in train_losses.items():
        writer.add_scalar(f'Train/{k}', v, epoch)
    for k, v in test_losses.items():
        writer.add_scalar(f'Test/{k}', v, epoch)

def compute_loss(loss_fn_inputs: dict, losses: list[dict]) -> dict:
    """
    Compute the weighted loss for the given inputs and loss functions.

    Args:
        loss_fn_inputs: Dictionary containing the inputs for the loss functions.
            keys can be 'x', 'x_hat', 'stats', 'snr', 'logits', 'targets'.
        losses: List of dictionaries, each containing:
            - 'loss_name': Name of the loss function (e.g., 'mse_loss', 'huber_noise_loss', etc.)
            - 'callable': The loss function to call
            - 'weight': Weight to apply to the loss
            - 'kwargs': Additional keyword arguments for the loss function.
    """
    _loss = {}
    for loss in losses:
        if loss['loss_name'] in ['mse_loss', 'huber_noise_loss']:
            assert 'x' in loss_fn_inputs and 'x_hat' in loss_fn_inputs, \
                f"Inputs for {loss['loss_name']} must contain 'x' and 'x_hat'."
            
            loss_value = loss['callable'](
                loss_fn_inputs['x'],
                loss_fn_inputs['x_hat'],
                **loss['kwargs']
            )
            _loss[loss['loss_name']] = loss_value * loss['weight']

        elif loss['loss_name'] == 'snr_weighted_mse_loss':
            assert 'x' in loss_fn_inputs and 'x_hat' in loss_fn_inputs and 'snr' in loss_fn_inputs, \
                f"Inputs for {loss['loss_name']} must contain 'x', 'x_hat', and 'snr'."
            
            loss_value = loss['callable'](
                loss_fn_inputs['x'],
                loss_fn_inputs['x_hat'],
                snr=loss_fn_inputs['snr'],
                **loss['kwargs']
            )
            _loss[loss['loss_name']] = loss_value * loss['weight']

        elif loss['loss_name'] == 'kl_divergence':
            assert 'stats' in loss_fn_inputs, \
                f"Inputs for {loss['loss_name']} must contain 'stats'."
            
            loss_value = loss['callable'](
                *loss_fn_inputs['stats'],
                **loss['kwargs']
            )
            _loss[loss['loss_name']] = loss_value * loss['weight']

        elif loss['loss_name'] == 'VAE_loss':
            assert 'x' in loss_fn_inputs and 'x_hat' in loss_fn_inputs and 'stats' in loss_fn_inputs, \
                f"Inputs for {loss['loss_name']} must contain 'x', 'x_hat', and 'stats'."
            
            loss_value = loss['callable'](
                loss_fn_inputs['x'],
                loss_fn_inputs['x_hat'],
                *loss_fn_inputs['stats'],
                **loss['kwargs']
            )
            _loss[loss['loss_name']] = loss_value * loss['weight']

        elif loss['loss_name'] == 'cross_entropy_loss':
            assert 'logits' in loss_fn_inputs and 'targets' in loss_fn_inputs, \
                f"Inputs for {loss['loss_name']} must contain 'logits' and 'targets'."
            
            loss_value = loss['callable'](
                loss_fn_inputs['logits'],
                loss_fn_inputs['targets'],
                **loss['kwargs']
            )
            _loss[loss['loss_name']] = loss_value * loss['weight']

        else:
            raise ValueError(f"Unknown loss function: {loss['loss_name']}")
        
    return _loss

def train_vae(vae: VAE, train_dataloader: torch.utils.data.DataLoader,
              test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer,
              losses: list[dict], epochs: int,  scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
              checkpoint_folder: str = "./checkpoints", use_tqdm: bool = False, use_tensorboard: bool = False, 
              validation: bool = True, seed: Optional[int] = None, mixed_precision: bool = False,
              device: str = "cuda" if torch.cuda.is_available() else "cpu", return_model: bool = False) -> Optional[VAE]:
    """
    Train the VAE model.

    Args:
        vae: The VAE model to train.
        train_dataloader: DataLoader for the training dataset.
        test_dataloader: DataLoader for the validation dataset.
        optimizer: Optimizer for the VAE model.
        losses: List of dictionaries, each containing:
            - 'loss_name': Name of the loss function (e.g., 'mse_loss', 'huber_noise_loss', etc.)
            - 'callable': The loss function to call
            - 'weight': Weight to apply to the loss
            - 'kwargs': Additional keyword arguments for the loss function.
        epochs: Number of training epochs.
        scheduler: Learning rate scheduler for the optimizer.
        checkpoint_folder: Folder to save the model checkpoints.
        use_tqdm: Whether to use tqdm for progress bars.
        use_tensorboard: Whether to use TensorBoard for logging.
        validation: Whether to perform validation after each epoch on one test batch.
        seed: Random seed for reproducibility.
        mixed_precision: Whether to use mixed precision training.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
        return_model: Whether to return the trained VAE model.
    """
    if not os.path.exists(os.path.join(checkpoint_folder, 'VAE')):
        os.makedirs(os.path.join(checkpoint_folder, 'VAE'))

    # Check for existing checkpoints
    if os.path.exists(os.path.join(checkpoint_folder, "VAE", "vae.pth")):
        checkpoint = torch.load(os.path.join(checkpoint_folder, "VAE", "vae.pth"), map_location=device)
        vae.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        best_loss = checkpoint.get('best_loss', float('inf'))
        seed = checkpoint.get('seed', None)
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"[INFO] Resumed VAE training from epoch {start_epoch} with best loss {best_loss:.4f}.\n")
    else:
        start_epoch = 0
        best_loss = float('inf')
        print("[INFO] Starting VAE training from scratch.\n")

    if start_epoch >= epochs:
        print("[INFO] Training already completed for the specified number of epochs.")
        if return_model:
            return vae
        else:
            return None

    if use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(checkpoint_folder, "VAE"))

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    scaler = GradScaler() if (device.startswith('cuda') and mixed_precision) else None
    for epoch in range(start_epoch, epochs):
        
        #
        # Training loop
        #

        vae.train()

        train_losses = {loss_fn['loss_name']: 0.0 for loss_fn in losses}

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", disable=not use_tqdm)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(scaler is not None), device_type="cuda"):
                # Forward pass
                _loss = step_vae(
                    vae, batch, losses, generator, device
                )

            # Update the training losses
            for k, v in _loss.items():
                train_losses[k] += v.item() / len(train_dataloader)

            total_loss = sum(_loss.values())
            
            # Backward pass and optimization
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        #
        # Testing loop
        # 

        vae.eval()

        test_losses = {loss_fn['loss_name']: 0.0 for loss_fn in losses}

        pbar = tqdm(test_dataloader, desc=f"Testing {epoch + 1}/{epochs}", unit="batch", disable=not use_tqdm)
        with torch.no_grad():
            for batch in pbar:
                # Forward pass
                _loss = step_vae(
                    vae, batch, losses, generator, device
                )

                # Update the testing losses
                for k, v in _loss.items():
                    test_losses[k] += v.item() / len(test_dataloader)
        
        # Update the scheduler if it exists
        tot_test_loss = sum(test_losses.values())
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step(tot_test_loss)
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Summary
        print_epoch_summary(
            epoch, epochs, current_lr, train_losses, test_losses, best_loss
        )
        if use_tensorboard:
            write_tensorboard_summary(
                writer, epoch, current_lr, train_losses, test_losses
            )

        if tot_test_loss < best_loss:
            best_loss = tot_test_loss
            torch.save({
                'model_state_dict': vae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_loss': best_loss,
                'epoch': epoch,
                'seed': seed
            }, os.path.join(checkpoint_folder, "VAE", "vae.pth"))
            print(f"[INFO] New best loss: {best_loss:.4f}. Model saved.\n")
        else:
            print(f"[INFO] No improvement in loss. Current best loss: {best_loss:.4f}.\n")

        #
        # Validation loop
        #

        if not validation:
            continue

        with torch.no_grad():
            batch = next(iter(test_dataloader))
            images = batch['image'].to(device)
            noise = torch.randn(images.size(0), *vae.latent_shape, device=device, generator=generator)
            _, rec_images = vae(
                images, noise, return_stats=False, rescale=False, return_rec=True
            )

            num_samples = min(32, images.size(0))
            cmap = 'gray' if images.size(1) == 1 else None

            fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
            for i in range(num_samples):
                # Original images
                axes[i, 0].imshow(
                    images[i].cpu().numpy().squeeze(),
                    cmap=cmap, vmin=0, vmax=1
                )
                axes[i, 0].set_title(f"Original Image {i+1}")
                axes[i, 0].axis('off')

                # Reconstructed images
                axes[i, 1].imshow(
                    rec_images[i].cpu().numpy().squeeze(), 
                    cmap=cmap, vmin=0, vmax=1
                )
                axes[i, 1].set_title(f"Reconstructed Image {i+1}")
                axes[i, 1].axis('off')

            plt.tight_layout()

        if use_tensorboard:
            writer.add_figure('VAE Reconstruction', fig, global_step=epoch)
            plt.close(fig)

    if return_model:
        return vae
    
def step_vae(vae: VAE, batch: dict, losses: list[dict], 
             generator: torch.Generator, device: str) -> dict[str, torch.Tensor]:
    """
    Perform a single step for the VAE model.

    Args:
        vae: The VAE model to train.
        batch: A dictionary containing the batch data.
            Expected keys are 'image' for the input images.
        losses: List of dictionaries, each containing:
            - 'loss_name': Name of the loss function (e.g., 'mse_loss', 'huber_noise_loss', etc.)
            - 'callable': The loss function to call
            - 'weight': Weight to apply to the loss
            - 'kwargs': Additional keyword arguments for the loss function.
        generator: Random number generator for reproducibility.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
    
    Returns:
        A dictionary containing the computed losses for the batch.
    """
    images = batch['image'].to(device)

    # Forward pass
    noise = torch.randn(images.size(0), *vae.latent_shape, device=device, generator=generator)
    _, stats, rec_images = vae(
        images, noise, return_stats=True, rescale=False, return_rec=True
    )

    # Loss computations
    loss_fn_inputs = {
        'x': images,
        'x_hat': rec_images,
        'stats': stats
    }
    return compute_loss(loss_fn_inputs, losses)

def train_diffusion(diffusion: Diffusion, noise_scheduler: NoiseScheduler, train_dataloader: torch.utils.data.DataLoader, 
                    test_dataloader: torch.utils.data.DataLoader, optimizer: torch.optim.Optimizer, losses: list[dict], 
                    epochs: int, vae: Optional[VAE] = None, scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                    checkpoint_folder: str = "./checkpoints", use_tqdm: bool = False, use_tensorboard: bool = False, 
                    validation: bool = True, seed: Optional[int] = None, mixed_precision: bool = False,
                    device: str = "cuda" if torch.cuda.is_available() else "cpu", return_model: bool = False) -> Optional[Diffusion]:
    """
    Train the diffusion model.

    Args:
        diffusion: The diffusion model to train.
        noise_scheduler: The noise scheduler for the diffusion model.
        train_dataloader: DataLoader for the training dataset.
        test_dataloader: DataLoader for the validation dataset.
        optimizer: Optimizer for the diffusion model.
        losses: List of dictionaries, each containing:
            - 'loss_name': Name of the loss function (e.g., 'mse_loss', 'huber_noise_loss', etc.)
            - 'callable': The loss function to call
            - 'weight': Weight to apply to the loss
            - 'kwargs': Additional keyword arguments for the loss function.
        epochs: Number of training epochs.
        vae: The VAE model used for encoding and decoding images.
            If None, dataloaders are expected to return latent vectors directly.
        scheduler: Learning rate scheduler for the optimizer.
        checkpoint_folder: Folder to save the model checkpoints.
        use_tqdm: Whether to use tqdm for progress bars.
        use_tensorboard: Whether to use TensorBoard for logging.
        validation: Whether to perform validation after each epoch on one test batch.
        seed: Random seed for reproducibility.
        mixed_precision: Whether to use mixed precision training.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
        return_model: Whether to return the trained diffusion model.
    """
    if not os.path.exists(os.path.join(checkpoint_folder, "diffusion")):
        os.makedirs(os.path.join(checkpoint_folder, "diffusion"))

    # Check for existing checkpoints
    if os.path.exists(os.path.join(checkpoint_folder, "diffusion", "diffusion.pth")):
        checkpoint = torch.load(os.path.join(checkpoint_folder, "diffusion", "diffusion.pth"), map_location=device)
        diffusion.load_state_dict(checkpoint['model_state_dict'])
        noise_scheduler.load_state_dict(checkpoint['noise_scheduler_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler and checkpoint['scheduler_state_dict']:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        best_loss = checkpoint.get('best_loss', float('inf'))
        seed = checkpoint.get('seed', None)
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"[INFO] Resumed diffusion training from epoch {start_epoch} with best loss {best_loss:.4f}.\n")
    else:
        start_epoch = 0
        best_loss = float('inf')
        print("[INFO] Starting diffusion training from scratch.\n")

    if start_epoch >= epochs:
        print("[INFO] Training already completed for the specified number of epochs.")
        if return_model:
            return diffusion
        else:
            return None
        
    if use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(checkpoint_folder, "diffusion"))

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)

    scaler = GradScaler() if (device.startswith('cuda') and mixed_precision) else None
    for epoch in range(start_epoch, epochs):

        #
        # Training loop
        #

        diffusion.train()
        vae.eval() if vae is not None else None

        train_losses = {loss_fn['loss_name']: 0.0 for loss_fn in losses}

        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch", disable=not use_tqdm)
        for batch in pbar:
            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=(scaler is not None), device_type="cuda"):
                # Forward pass
                _loss = step_diffusion(
                    diffusion, batch, noise_scheduler, losses, generator, device, vae
                )

            # Update the training losses
            for k, v in _loss.items():
                train_losses[k] += v.item() / len(train_dataloader)

            total_loss = sum(_loss.values())

            # Backward pass and optimization
            if scaler is not None:
                scaler.scale(total_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()
                optimizer.step()

        #
        # Testing loop
        #

        diffusion.eval()
        vae.eval() if vae is not None else None

        test_losses = {loss_fn['loss_name']: 0.0 for loss_fn in losses}

        pbar = tqdm(test_dataloader, desc=f"Testing {epoch + 1}/{epochs}", unit="batch", disable=not use_tqdm)
        with torch.no_grad():
            for batch in pbar:
                # Forward pass
                _loss = step_diffusion(
                    diffusion, batch, noise_scheduler, losses, generator, device, vae
                )

                # Update the testing losses
                for k, v in _loss.items():
                    test_losses[k] += v.item() / len(test_dataloader)

        # Update the scheduler if it exists
        tot_test_loss = sum(test_losses.values())
        if scheduler:
            current_lr = scheduler.get_last_lr()[0]
            scheduler.step(tot_test_loss)
        else:
            current_lr = optimizer.param_groups[0]['lr']

        # Summary
        print_epoch_summary(
            epoch, epochs, current_lr, train_losses, test_losses, best_loss
        )
        if use_tensorboard:
            write_tensorboard_summary(
                writer, epoch, current_lr, train_losses, test_losses
            )

        if tot_test_loss < best_loss:
            best_loss = tot_test_loss
            torch.save({
                'model_state_dict': diffusion.state_dict(),
                'noise_scheduler_state_dict': noise_scheduler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_loss': best_loss,
                'epoch': epoch,
                'seed': seed
            }, os.path.join(checkpoint_folder, "diffusion", "diffusion.pth"))
            print(f"[INFO] New best loss: {best_loss:.4f}. Model saved.\n")
        else:
            print(f"[INFO] No improvement in loss. Current best loss: {best_loss:.4f}.\n")

        #
        # Validation loop
        #

        if not validation:
            continue

        with torch.no_grad():

            # Noise prediction

            batch = next(iter(test_dataloader))
            contexts = batch['context'].to(device)
            if vae is not None:
                images = batch['image'].to(device)
                noise = torch.randn(images.size(0), *vae.latent_shape, device=device, generator=generator)
                latent_images = vae(
                    images, noise, return_stats=False, rescale=True, return_rec=False
                )
            else:
                latent_images = batch['image'].to(device)

            # Sample the time steps and add the noise
            t = torch.randint(
                0, noise_scheduler.steps, (latent_images.size(0),), 
                device=device, generator=generator
            )
            snr = noise_scheduler.compute_snr(t)

            noise = torch.randn(latent_images.size(), device=device, generator=generator)
            noisy_images = noise_scheduler.add_noise_cumulative(
                latent_images, t, noise
            )

            # Forward pass through the diffusion model
            pred_noise = diffusion(
                noisy_images, t, contexts
            )

            num_samples = min(32, images.size(0))
            cmap = 'gray'

            noise = noise.mean(dim=1, keepdim=False) # Mean across channels for visualization
            pred_noise = pred_noise.mean(dim=1, keepdim=False)
            vmax = max(noise.max().item(), pred_noise.max().item())
            vmin = min(noise.min().item(), pred_noise.min().item())

            fig, axes = plt.subplots(num_samples, 2, figsize=(10, 5 * num_samples))
            for i in range(num_samples):
                # Original images
                axes[i, 0].imshow(
                    noise[i].cpu().numpy().squeeze(),
                    cmap=cmap, vmin=vmin, vmax=vmax
                )
                axes[i, 0].set_title(f"Real Noise {i+1}")
                axes[i, 0].text(0., 0., f'{str(t[i].cpu().item())}: {snr[i]}' , color='red', fontsize=14, ha='left', va='top')
                axes[i, 0].axis('off')

                # Reconstructed images
                axes[i, 1].imshow(
                    pred_noise[i].cpu().numpy().squeeze(), 
                    cmap=cmap, vmin=vmin, vmax=vmax
                )
                axes[i, 1].set_title(f"Predicted Noise {i+1}")
                axes[i, 1].axis('off')

            plt.tight_layout()

            if use_tensorboard:
                writer.add_figure('Diffusion Predicted Noise', fig, global_step=epoch)
                plt.close(fig)

            # Sampling with DDPM

            if vae is None:
                continue

            if images.size(1) == 1:
                cmap = 'gray'
            elif images.size(1) == 3:
                cmap = None
            else:
                continue

            sampler = DDPMSampler(noise_scheduler=noise_scheduler, use_tqdm=False).to(device)
            x = torch.randn((1, *vae.latent_shape), device=device)

            sampled_latent = sampler.sample(
                x,
                pred_noise_func=diffusion,
                func_inputs={'context': contexts[i, None]},
                return_intermediates=True,
                return_step=noise_scheduler.steps // 100
            )
            generated_images = vae.decoder(
                torch.cat(sampled_latent, dim=0), rescale=True
            )

            fig, axes = plt.subplots(1, 10, figsize=(10 * 5, 5))
            for i in range(10):
                axes[i].imshow(
                    generated_images[i].cpu().numpy().transpose(1, 2, 0), cmap=cmap
                )
                axes[i].set_title(f"DDPM Generated Image {i+1}")
                axes[i].axis('off')
        
            plt.tight_layout()

            if use_tensorboard:
                writer.add_figure('DDPM Generated Images', fig, global_step=epoch)
                plt.close(fig)

            # Sampling with DDIM

            sampler = DDIMSampler(
                noise_scheduler=noise_scheduler, steps=50, eta=0.05, use_tqdm=False
            ).to(device)
            x = torch.randn((1, *vae.latent_shape), device=device)

            sampled_latent = sampler.sample(
                x,
                pred_noise_func=diffusion,
                func_inputs={'context': contexts[i, None]},
                return_intermediates=True,
                return_step=sampler.steps // 10
            )
            generated_images = vae.decoder(
                torch.cat(sampled_latent, dim=0), rescale=True
            )

            fig, axes = plt.subplots(1, 10, figsize=(10 * 5, 5))
            for i in range(10):
                axes[i].imshow(
                    generated_images[i].cpu().numpy().transpose(1, 2, 0), cmap=cmap
                )
                axes[i].set_title(f"DDIM Generated Image {i+1}")
                axes[i].axis('off')

            plt.tight_layout()

            if use_tensorboard:
                writer.add_figure('DDIM Generated Images', fig, global_step=epoch)
                plt.close(fig)

    if return_model:
        return diffusion
    
def step_diffusion(diffusion: Diffusion, batch: dict, noise_scheduler: NoiseScheduler, losses: list[dict], 
                   generator: torch.Generator, device: str, vae: Optional[VAE] = None) -> dict[str, torch.Tensor]:
    """
    Perform a single step for the diffusion model.

    Args:
        diffusion: The diffusion model to train.
        batch: A dictionary containing the batch data.
            Expected keys are 'image' for the input images and 'context' for additional context.
        noise_scheduler: The noise scheduler for the diffusion model.
        losses: List of dictionaries, each containing:
            - 'loss_name': Name of the loss function (e.g., 'mse_loss', 'huber_noise_loss', etc.)
            - 'callable': The loss function to call
            - 'weight': Weight to apply to the loss
            - 'kwargs': Additional keyword arguments for the loss function.
        generator: Random number generator for reproducibility.
        device: Device to use for training (e.g., 'cuda' or 'cpu').
        vae: The VAE model used for encoding and decoding images.
            If None, dataloaders are expected to return latent vectors directly.
    
    Returns:
        A dictionary containing the computed losses for the batch.
    """
    contexts = batch['context'].to(device)
    if vae is not None:
        images = batch['image'].to(device)
        noise = torch.randn(images.size(0), *vae.latent_shape, device=device, generator=generator)
        with torch.no_grad():
            latent_images = vae(
                images, noise, return_stats=False, rescale=True, return_rec=False
            )
    else:
        latent_images = batch['image'].to(device)

    # Sample the time steps and add the noise
    t = torch.randint(
        0, noise_scheduler.steps, (latent_images.size(0),), 
        device=device, generator=generator
    )
    snr = noise_scheduler.compute_snr(t)

    noise = torch.randn(latent_images.size(), device=device, generator=generator)
    noisy_images = noise_scheduler.add_noise_cumulative(
        latent_images, t, noise
    )

    # Forward pass through the diffusion model
    pred_noise = diffusion(
        noisy_images, t, contexts
    )

    # Loss computations
    loss_fn_inputs = {
        'x': noise,
        'x_hat': pred_noise,
        'snr': snr
    }
    return compute_loss(loss_fn_inputs, losses)