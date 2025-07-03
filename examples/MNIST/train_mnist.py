"""
train_mnist.py

A simple script to train a diffusion model on the MNIST dataset.

Steps are the following:

    1. Load the MNIST dataset and create a data loader.
    2. Instantiate a VAE.
    3. Train the VAE model to encode and decode images.
    4. Instantiate a diffusion model and a NoiseScheduler.
    5. Train the diffusion model for a specified number of epochs.
    6. Save the trained models.
"""


import argparse
import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader

from MNIST import MNIST

from core.models import VAE, Diffusion
from core.schedulers import NoiseScheduler
from trainer.custom_lr_schedulers import CosineLRScheduler
from trainer.losses import VAE_loss, snr_weighted_mse_loss, mse_loss
from trainer.train import train_vae, train_diffusion


if __name__ == "__main__":

    print("\n-------------------------------------------------------\n", 
            "    🌀 The Noise Must Flow — The case of MNIST 🌀      ",
          "\n-------------------------------------------------------\n")
    
    parser = argparse.ArgumentParser(description="Train a diffusion model on MNIST dataset.")

    # Global parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--VAE_config_file", type=str, default="./MNIST_VAE.yaml",
                        help="Path to the VAE configuration file")
    parser.add_argument("--UNet_config_file", type=str, default="./MNIST_UNet.yaml",
                        help="Path to the diffusion model configuration file")
    parser.add_argument("--checkpoint_folder", type=str, default=None,
                        help="Folder to save/load the model checkpoints")
    parser.add_argument("--use_tqdm", type=bool, default=True,
                        help="Flag to indicate whether to use tqdm for progress bars")
    parser.add_argument("--use_tensorboard", type=bool, default=True,
                        help="Flag to indicate whether to use TensorBoard for logging")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    
                        
    # Data loader parameters
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")

    # Noise scheduler parameters
    parser.add_argument("--steps", type=int, default=1000, help="Number of steps for the noise scheduler")
    parser.add_argument("--betas", type=float, nargs=2, default=(1e-4, 0.02), 
                        help="Minimum and maximum beta values for the noise scheduler")
    parser.add_argument("--schedule", type=str, choices=["linear", "cosine", "quadratic", "sigmoid", "geometric"],
                        default="cosine", help="Type of noise schedule to use")

    # Optimizer parameters for VAE
    parser.add_argument("--vae_epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--vae_lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--vae_warmup", type=str, choices=["none", "cosine"], default="cosine", 
                        help="Type of warmup scheduler to use")
    parser.add_argument("--vae_warmup_steps", type=int, default=10, help="Number of warmup steps for the scheduler")

    # Optimizer parameters for diffusion model
    parser.add_argument("--diffusion_epochs", type=int, default=50, help="Number of training epochs for diffusion model")
    parser.add_argument("--diffusion_lr", type=float, default=2e-5, help="Learning rate for diffusion model")
    parser.add_argument("--diffusion_warmup", type=str, choices=["none", "cosine"], default="cosine", 
                        help="Type of warmup scheduler to use for diffusion model")
    parser.add_argument("--diffusion_warmup_steps", type=int, default=20, 
                        help="Number of warmup steps for the diffusion model scheduler")

    args = parser.parse_args()

    if args.checkpoint_folder is None:
        checkpoint_folder = time.strftime("./checkpoints/%Y-%m-%d_%H-%M-%S", time.localtime())
        os.makedirs(checkpoint_folder, exist_ok=False)
    elif not os.path.exists(args.checkpoint_folder):
        raise ValueError(f"Checkpoint folder '{args.checkpoint_folder}' does not exist.")
    else:
        checkpoint_folder = args.checkpoint_folder

    # 1. Load the MNIST dataset and create a data loader
    dataset = MNIST(root='./data', train=True, download=True, unconditional=True, d_context=128)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, 
                                  num_workers=args.num_workers, drop_last=True, pin_memory=True)
    test_dataloader = DataLoader(MNIST(root='./data', train=False, download=True), 
                                 batch_size=args.batch_size, shuffle=False, 
                                 num_workers=args.num_workers, drop_last=True, pin_memory=True)
    
    # 2. Instantiate a VAE
    vae = VAE(
        input_shape=(1, 64, 64),
        config_file=args.VAE_config_file
    ).to(args.device)

    print(f"VAE successfully instantiated with config file: {args.VAE_config_file}")
    print(f"VAE latent shape: {vae.latent_shape} [(latent_dim, height, width)]")
    print(f"{sum(p.numel() for p in vae.parameters() if p.requires_grad)} trainable parameters\n")

    # 3. Train the VAE model to encode and decode images if no checkpoints exist
    # Check for existing checkpoints
    if os.path.exists(os.path.join(checkpoint_folder, "VAE", "vae.pth")):
        checkpoint = torch.load(os.path.join(checkpoint_folder, "VAE", "vae.pth"), map_location=args.device)
        vae.load_state_dict(checkpoint['model_state_dict'])

        best_loss = checkpoint.get('best_loss', float('inf'))
        seed = checkpoint.get('seed', None)
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"[INFO] Resumed VAE training from epoch {start_epoch} with best loss {best_loss:.4f}.\n")
    else:
        optimizer = torch.optim.Adam(vae.parameters(), lr=args.vae_lr)

        if args.vae_warmup == "cosine":
            scheduler = CosineLRScheduler(
                optimizer=optimizer,
                base_lr=args.vae_lr,
                total_epochs=args.vae_epochs,
                warmup_epochs=args.vae_warmup_steps
            )

        losses = [
            {
                "loss_name": "VAE_loss",
                "callable": VAE_loss,
                "weight": 1.0,
                "kwargs": {
                    "beta": 1.0
                }
            }
        ]

        vae = train_vae(
            vae, train_dataloader, test_dataloader, optimizer, losses,
            epochs=args.vae_epochs, scheduler=scheduler, checkpoint_folder=checkpoint_folder, 
            use_tqdm=args.use_tqdm, use_tensorboard=args.use_tensorboard, validation=True, 
            device=args.device, seed=args.seed, return_model=True
        )

    # 4. Instantiate a diffusion model, a NoiseScheduler and a Sampler
    diffusion = Diffusion(
        latent_dim=vae.latent_dim,
        config_file=args.UNet_config_file
    ).to(args.device)

    print(f"Diffusion model successfully instantiated with config file: {args.UNet_config_file}")
    print(f"{sum(p.numel() for p in diffusion.parameters() if p.requires_grad)} trainable parameters\n")

    noise_scheduler = NoiseScheduler(
        steps=args.steps,
        betas=args.betas,
        schedule=args.schedule,
        seed=args.seed
    ).to(args.device)

    # 5. Train the diffusion model for a specified number of epochs
    # Check for existing checkpoints
    if os.path.exists(os.path.join(checkpoint_folder, "diffusion", "diffusion.pth")):
        checkpoint = torch.load(os.path.join(checkpoint_folder, "diffusion", "diffusion.pth"), map_location=args.device)
        diffusion.load_state_dict(checkpoint['model_state_dict'])
        noise_scheduler.load_state_dict(checkpoint['noise_scheduler_state_dict'])

        best_loss = checkpoint.get('best_loss', float('inf'))
        seed = checkpoint.get('seed', None)
        start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"[INFO] Resumed diffusion training from epoch {start_epoch} with best loss {best_loss:.4f}.\n")
    else:
        optimizer = torch.optim.Adam(
            diffusion.parameters(), 
            lr=args.diffusion_lr
        )

        if args.diffusion_warmup == "cosine":
            scheduler = CosineLRScheduler(
                optimizer=optimizer,
                base_lr=args.diffusion_lr,
                total_epochs=args.diffusion_epochs,
                warmup_epochs=args.diffusion_warmup_steps
            )

        losses = [
            {
                "loss_name": "mse_loss", #"snr_weighted_mse_loss",
                "callable": mse_loss, #snr_weighted_mse_loss,
                "weight": 1.0,
                "kwargs": {
                    #"gamma": 5.0,
                    "reduction": "sum"
                }
            }
        ]

        diffusion = train_diffusion(
            diffusion, noise_scheduler, train_dataloader, test_dataloader, optimizer, losses,
            epochs=args.diffusion_epochs, vae=vae, scheduler=scheduler, checkpoint_folder=checkpoint_folder,
            use_tqdm=args.use_tqdm, use_tensorboard=args.use_tensorboard, device=args.device, 
            seed=args.seed, return_model=True
        )