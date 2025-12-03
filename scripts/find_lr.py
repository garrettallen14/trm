#!/usr/bin/env python3
"""
Learning Rate Finder - Find optimal LR before training.

Based on Leslie Smith's LR range test:
1. Start with tiny LR
2. Increase exponentially each batch
3. Track loss
4. Optimal LR = where loss decreases fastest (steepest slope)

Usage:
    python scripts/find_lr.py --d_model 512 --n_layers 4
"""

import argparse
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.train_diffusion import DiscreteDiffusionTRM, DiffusionConfig
from src.data import create_dataloader


def find_lr(
    model,
    train_loader,
    device,
    start_lr=1e-7,
    end_lr=1,
    num_steps=200,
    use_amp=True
):
    """
    Find optimal learning rate using range test.
    
    Returns:
        lrs: list of learning rates tested
        losses: list of corresponding losses
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=start_lr)
    scaler = GradScaler('cuda') if use_amp else None
    
    # Exponential LR schedule
    gamma = (end_lr / start_lr) ** (1 / num_steps)
    
    lrs = []
    losses = []
    smoothed_loss = None
    best_loss = float('inf')
    
    data_iter = iter(train_loader)
    
    for step in range(num_steps):
        # Get batch
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)
        
        # Move to device
        demo_inputs = [d.to(device) for d in batch['demo_inputs']]
        demo_outputs = [d.to(device) for d in batch['demo_outputs']]
        test_input = batch['test_input'].to(device)
        test_output = batch['test_output'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        
        with autocast('cuda', enabled=use_amp):
            loss_dict = model.compute_loss(
                demo_inputs, demo_outputs, test_input, test_output,
                return_metrics=False
            )
            loss = loss_dict['total_loss']
        
        # Check for divergence
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Loss diverged at LR={optimizer.param_groups[0]['lr']:.2e}")
            break
        
        # Smooth the loss
        if smoothed_loss is None:
            smoothed_loss = loss.item()
        else:
            smoothed_loss = 0.9 * smoothed_loss + 0.1 * loss.item()
        
        # Stop if loss explodes
        if smoothed_loss > 4 * best_loss:
            print(f"Loss exploding at LR={optimizer.param_groups[0]['lr']:.2e}")
            break
        
        if smoothed_loss < best_loss:
            best_loss = smoothed_loss
        
        # Record
        current_lr = optimizer.param_groups[0]['lr']
        lrs.append(current_lr)
        losses.append(smoothed_loss)
        
        if step % 20 == 0:
            print(f"Step {step}: LR={current_lr:.2e}, Loss={smoothed_loss:.4f}")
        
        # Backward pass
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        # Update LR
        for param_group in optimizer.param_groups:
            param_group['lr'] *= gamma
    
    return lrs, losses


def plot_lr_finder(lrs, losses, save_path="lr_finder.png"):
    """Plot LR finder results."""
    plt.figure(figsize=(10, 6))
    plt.semilogx(lrs, losses)
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.title('Learning Rate Finder')
    plt.grid(True)
    
    # Find suggested LR (steepest descent)
    if len(losses) > 10:
        # Compute gradient
        gradients = []
        for i in range(5, len(losses) - 5):
            grad = (losses[i+5] - losses[i-5]) / (math.log(lrs[i+5]) - math.log(lrs[i-5]))
            gradients.append((lrs[i], grad))
        
        # Find minimum gradient (steepest descent)
        min_grad_lr = min(gradients, key=lambda x: x[1])[0]
        suggested_lr = min_grad_lr / 10  # Conservative choice
        
        plt.axvline(x=suggested_lr, color='r', linestyle='--', label=f'Suggested: {suggested_lr:.2e}')
        plt.legend()
        print(f"\n✓ Suggested LR: {suggested_lr:.2e}")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--num_timesteps", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_steps", type=int, default=200)
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Create model
    config = DiffusionConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_timesteps=args.num_timesteps
    )
    
    model = DiscreteDiffusionTRM(config).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dataloader
    train_loader = create_dataloader(
        args.data_dir,
        split="training",
        batch_size=args.batch_size,
        augment=True,
        augment_factor=10
    )
    
    # Find LR
    print("\nRunning LR finder...")
    lrs, losses = find_lr(model, train_loader, device, num_steps=args.num_steps)
    
    # Plot results
    plot_lr_finder(lrs, losses)


if __name__ == "__main__":
    main()
