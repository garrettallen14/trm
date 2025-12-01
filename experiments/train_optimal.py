#!/usr/bin/env python3
"""
Full training run with optimal hyperparameters from sweep.

Based on sweep results:
- d_model=512 (required for ARC complexity)
- n_recursions=8 (best accuracy/speed tradeoff)
- lr_trunk=1e-4, lr_embed=5e-3 (good convergence)
- supervision=linear (promising in sweep)

Usage:
    uv run python experiments/train_optimal.py
    uv run python experiments/train_optimal.py --epochs 50
"""

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.model import TinyRecursiveModel
from src.data import create_dataloader


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    # Optimal config from sweeps
    config = {
        "d_model": 512,
        "n_heads": 4,
        "n_layers": 2,
        "n_recursions": 8,
        "lr_trunk": 1e-4,
        "lr_embed": 5e-3,
        "weight_decay": 0.01,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "epochs": args.epochs,
        "augment_factor": args.augment_factor,
        "supervision_weights": "linear",  # Promising from sweep
    }
    
    print("\n" + "="*60)
    print("TRAINING CONFIG")
    print("="*60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = TinyRecursiveModel(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_recursions=config["n_recursions"]
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")
    
    # Optimizer with differential LR
    embed_params = [p for n, p in model.named_parameters() if "embed" in n or "token" in n]
    trunk_params = [p for n, p in model.named_parameters() if "embed" not in n and "token" not in n]
    
    optimizer = optim.AdamW([
        {"params": trunk_params, "lr": config["lr_trunk"]},
        {"params": embed_params, "lr": config["lr_embed"]}
    ], weight_decay=config["weight_decay"])
    
    # Cosine annealing scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=config["epochs"], eta_min=1e-6)
    
    # Data
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        split="training",
        batch_size=config["batch_size"],
        augment=True,
        augment_factor=config["augment_factor"],
        num_workers=2
    )
    
    # Training state
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"experiments/runs/{run_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    history = {
        "train_loss": [],
        "val_accuracy": [],
        "learning_rates": [],
        "epoch_times": []
    }
    
    best_val_acc = 0.0
    
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    for epoch in range(1, config["epochs"] + 1):
        epoch_start = time.time()
        model.train()
        
        epoch_loss = 0.0
        n_batches = 0
        accum_step = 0
        oom_count = 0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        
        for batch in pbar:
            try:
                demo_inputs = [g.to(device) for g in batch["demo_inputs"]]
                demo_outputs = [g.to(device) for g in batch["demo_outputs"]]
                test_input = batch["test_input"].to(device)
                test_output = batch["test_output"].to(device)
                
                loss_dict = model.compute_loss(
                    demo_inputs, demo_outputs, test_input, test_output,
                    n_recursions=config["n_recursions"],
                    supervision_weights=config["supervision_weights"]
                )
                loss = loss_dict["total_loss"] / config["grad_accum"]
                loss.backward()
                
                accum_step += 1
                epoch_loss += loss.item() * config["grad_accum"]
                n_batches += 1
                
                if accum_step >= config["grad_accum"]:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_step = 0
                
                pbar.set_postfix({
                    "loss": f"{loss.item() * config['grad_accum']:.4f}",
                    "oom": oom_count
                })
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_count += 1
                    clear_memory()
                    optimizer.zero_grad()
                    accum_step = 0
                    continue
                raise e
        
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        
        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # Quick validation
        model.eval()
        val_loader = create_dataloader(
            data_dir=args.data_dir,
            split="training",
            batch_size=1,
            augment=False,
            num_workers=0
        )
        
        correct_cells = 0
        total_cells = 0
        
        with torch.no_grad():
            for batch in list(val_loader)[:50]:  # Sample 50 tasks
                try:
                    demo_inputs = [g.to(device) for g in batch["demo_inputs"]]
                    demo_outputs = [g.to(device) for g in batch["demo_outputs"]]
                    test_input = batch["test_input"].to(device)
                    test_output = batch["test_output"].to(device)
                    
                    preds = model.predict(demo_inputs, demo_outputs, test_input)
                    
                    mask = test_output != 10
                    correct_cells += ((preds == test_output) & mask).sum().item()
                    total_cells += mask.sum().item()
                except RuntimeError:
                    clear_memory()
                    continue
        
        val_acc = correct_cells / total_cells if total_cells > 0 else 0
        
        # Log
        history["train_loss"].append(avg_loss)
        history["val_accuracy"].append(val_acc)
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_time)
        
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, val_acc={val_acc:.2%}, "
              f"lr={current_lr:.2e}, time={epoch_time:.1f}s, oom={oom_count}")
        
        # Save checkpoint if best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_acc,
                "config": config
            }, save_dir / "best_model.pt")
            print(f"  -> New best! Saved to {save_dir / 'best_model.pt'}")
        
        # Save history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best validation accuracy: {best_val_acc:.2%}")
    print(f"Results saved to: {save_dir}")
    
    return history, save_dir


def main():
    parser = argparse.ArgumentParser(description="Train TRM with optimal hyperparameters")
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)  # Safe default
    parser.add_argument("--grad_accum", type=int, default=16)
    parser.add_argument("--augment_factor", type=int, default=10)
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
