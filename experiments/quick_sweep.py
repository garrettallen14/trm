#!/usr/bin/env python3
"""
Quick hyperparameter sweep to find optimal settings before full training.

Tests key hyperparameters with short runs (5-10 epochs) to find:
1. Optimal learning rates
2. Optimal recursion depth
3. Optimal supervision weighting
4. Effect of augmentation factor

Usage:
    uv run python experiments/quick_sweep.py --data_dir data/arc-agi-1
    uv run python experiments/quick_sweep.py --sweep lr
    uv run python experiments/quick_sweep.py --sweep all
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import gc

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.model import TinyRecursiveModel
from src.data import create_dataloader


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@dataclass
class SweepConfig:
    """Single experiment configuration."""
    name: str
    d_model: int = 512  # Full model size
    n_heads: int = 4
    n_layers: int = 2
    n_recursions: int = 8  # Full recursions
    batch_size: int = 16  # Should work now with Flash Attention
    grad_accum: int = 2  # Effective batch = 16 * 2 = 32
    lr_trunk: float = 1e-4
    lr_embed: float = 1e-2
    supervision_weights: str = "uniform"
    augment_factor: int = 10
    epochs: int = 5


def run_experiment(config: SweepConfig, data_dir: str, device: torch.device) -> dict:
    """Run a single experiment and return metrics."""
    print(f"\n{'='*60}")
    print(f"Experiment: {config.name}")
    print(f"{'='*60}")
    for k, v in asdict(config).items():
        if k != "name":
            print(f"  {k}: {v}")
    
    # Create model
    model = TinyRecursiveModel(
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        n_recursions=config.n_recursions
    ).to(device)
    
    # Optimizer with differential LR
    embed_params = [p for n, p in model.named_parameters() if "embed" in n or "token" in n]
    trunk_params = [p for n, p in model.named_parameters() if "embed" not in n and "token" not in n]
    
    optimizer = optim.AdamW([
        {"params": trunk_params, "lr": config.lr_trunk},
        {"params": embed_params, "lr": config.lr_embed}
    ], weight_decay=0.01)
    
    # Data
    train_loader = create_dataloader(
        data_dir=data_dir,
        split="training",
        batch_size=config.batch_size,
        augment=True,
        augment_factor=config.augment_factor,
        num_workers=0  # Simpler for experiments
    )
    
    # Training with gradient accumulation
    start_time = time.time()
    train_losses = []
    grad_accum = getattr(config, 'grad_accum', 8)
    
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        accum_step = 0
        
        optimizer.zero_grad()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
        for batch in pbar:
            try:
                demo_inputs = [g.to(device) for g in batch["demo_inputs"]]
                demo_outputs = [g.to(device) for g in batch["demo_outputs"]]
                test_input = batch["test_input"].to(device)
                test_output = batch["test_output"].to(device)
                
                loss_dict = model.compute_loss(
                    demo_inputs, demo_outputs, test_input, test_output,
                    n_recursions=config.n_recursions,
                    supervision_weights=config.supervision_weights
                )
                loss = loss_dict["total_loss"] / grad_accum
                loss.backward()
                
                accum_step += 1
                epoch_loss += loss.item() * grad_accum
                n_batches += 1
                
                if accum_step >= grad_accum:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    accum_step = 0
                
                pbar.set_postfix({"loss": f"{loss.item() * grad_accum:.4f}"})
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    pbar.set_postfix({"status": "OOM-skip"})
                    clear_memory()
                    optimizer.zero_grad()
                    accum_step = 0
                    continue
                raise e
        
        avg_loss = epoch_loss / n_batches
        train_losses.append(avg_loss)
        print(f"  Epoch {epoch}: loss = {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    
    # Quick validation (no augmentation)
    model.eval()
    val_loader = create_dataloader(
        data_dir=data_dir,
        split="training",  # Use training as proxy (we're just comparing relative performance)
        batch_size=config.batch_size,
        augment=False,
        num_workers=0
    )
    
    correct_cells = 0
    total_cells = 0
    
    with torch.no_grad():
        for batch in list(val_loader)[:20]:  # Quick sample
            try:
                demo_inputs = [g.to(device) for g in batch["demo_inputs"]]
                demo_outputs = [g.to(device) for g in batch["demo_outputs"]]
                test_input = batch["test_input"].to(device)
                test_output = batch["test_output"].to(device)
                
                preds = model.predict(demo_inputs, demo_outputs, test_input)
                
                mask = test_output != 10  # Ignore padding
                correct_cells += ((preds == test_output) & mask).sum().item()
                total_cells += mask.sum().item()
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    clear_memory()
                    continue
                raise e
    
    val_acc = correct_cells / total_cells if total_cells > 0 else 0
    
    result = {
        "name": config.name,
        "config": asdict(config),
        "final_loss": train_losses[-1],
        "train_losses": train_losses,
        "val_accuracy": val_acc,
        "elapsed_seconds": elapsed,
        "samples_per_second": (len(train_loader) * config.batch_size * config.epochs) / elapsed
    }
    
    print(f"  Final loss: {result['final_loss']:.4f}")
    print(f"  Val accuracy: {result['val_accuracy']:.2%}")
    print(f"  Time: {elapsed:.1f}s ({result['samples_per_second']:.1f} samples/s)")
    
    return result


def sweep_learning_rates(data_dir: str, device: torch.device) -> list[dict]:
    """Sweep learning rate combinations."""
    print("\n" + "#"*60)
    print("# LEARNING RATE SWEEP")
    print("#"*60)
    
    configs = [
        SweepConfig(name="lr_trunk_1e-5", lr_trunk=1e-5, lr_embed=1e-2),
        SweepConfig(name="lr_trunk_5e-5", lr_trunk=5e-5, lr_embed=1e-2),
        SweepConfig(name="lr_trunk_1e-4", lr_trunk=1e-4, lr_embed=1e-2),
        SweepConfig(name="lr_trunk_3e-4", lr_trunk=3e-4, lr_embed=1e-2),
        SweepConfig(name="lr_embed_1e-3", lr_trunk=1e-4, lr_embed=1e-3),
        SweepConfig(name="lr_embed_5e-3", lr_trunk=1e-4, lr_embed=5e-3),
        SweepConfig(name="lr_embed_1e-2", lr_trunk=1e-4, lr_embed=1e-2),
        SweepConfig(name="lr_embed_5e-2", lr_trunk=1e-4, lr_embed=5e-2),
    ]
    
    return [run_experiment(cfg, data_dir, device) for cfg in configs]


def sweep_recursion_depth(data_dir: str, device: torch.device) -> list[dict]:
    """Sweep recursion depths."""
    print("\n" + "#"*60)
    print("# RECURSION DEPTH SWEEP")
    print("#"*60)
    
    configs = [
        SweepConfig(name="rec_2", n_recursions=2),
        SweepConfig(name="rec_4", n_recursions=4),
        SweepConfig(name="rec_8", n_recursions=8),
        SweepConfig(name="rec_12", n_recursions=12),
        SweepConfig(name="rec_16", n_recursions=16),
    ]
    
    return [run_experiment(cfg, data_dir, device) for cfg in configs]


def sweep_supervision(data_dir: str, device: torch.device) -> list[dict]:
    """Sweep supervision weighting schemes."""
    print("\n" + "#"*60)
    print("# SUPERVISION WEIGHTING SWEEP")
    print("#"*60)
    
    configs = [
        SweepConfig(name="sup_uniform", supervision_weights="uniform"),
        SweepConfig(name="sup_linear", supervision_weights="linear"),
    ]
    
    return [run_experiment(cfg, data_dir, device) for cfg in configs]


def sweep_model_size(data_dir: str, device: torch.device) -> list[dict]:
    """Sweep model sizes."""
    print("\n" + "#"*60)
    print("# MODEL SIZE SWEEP")
    print("#"*60)
    
    configs = [
        SweepConfig(name="tiny_128", d_model=128, n_heads=2, batch_size=64),
        SweepConfig(name="small_256", d_model=256, n_heads=4, batch_size=48),
        SweepConfig(name="base_512", d_model=512, n_heads=4, batch_size=32),
        SweepConfig(name="large_768", d_model=768, n_heads=8, batch_size=16),
    ]
    
    return [run_experiment(cfg, data_dir, device) for cfg in configs]


def main():
    parser = argparse.ArgumentParser(description="Quick hyperparameter sweep")
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1")
    parser.add_argument("--sweep", type=str, default="all", 
                       choices=["lr", "recursion", "supervision", "size", "all"])
    parser.add_argument("--output", type=str, default="experiments/sweep_results.json")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    all_results = {}
    
    if args.sweep in ["lr", "all"]:
        all_results["learning_rates"] = sweep_learning_rates(args.data_dir, device)
    
    if args.sweep in ["recursion", "all"]:
        all_results["recursion_depth"] = sweep_recursion_depth(args.data_dir, device)
    
    if args.sweep in ["supervision", "all"]:
        all_results["supervision"] = sweep_supervision(args.data_dir, device)
    
    if args.sweep in ["size", "all"]:
        all_results["model_size"] = sweep_model_size(args.data_dir, device)
    
    # Summary
    print("\n" + "="*60)
    print("SWEEP SUMMARY")
    print("="*60)
    
    for sweep_name, results in all_results.items():
        print(f"\n{sweep_name}:")
        # Sort by val accuracy
        sorted_results = sorted(results, key=lambda x: x["val_accuracy"], reverse=True)
        for r in sorted_results[:3]:
            print(f"  {r['name']}: loss={r['final_loss']:.4f}, acc={r['val_accuracy']:.2%}")
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
