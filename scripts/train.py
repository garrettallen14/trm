#!/usr/bin/env python3
"""
Training script with YAML config support.

Usage:
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --config configs/local_test.yaml
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import TrainConfig, Trainer


def load_config(config_path: str) -> TrainConfig:
    """Load config from YAML file."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    
    # Determine device
    device = cfg.get("hardware", {}).get("device", "auto")
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    
    return TrainConfig(
        # Model
        d_model=cfg.get("model", {}).get("d_model", 512),
        n_heads=cfg.get("model", {}).get("n_heads", 4),
        n_layers=cfg.get("model", {}).get("n_layers", 2),
        n_recursions=cfg.get("model", {}).get("n_recursions", 16),
        dropout=cfg.get("model", {}).get("dropout", 0.0),
        
        # Training
        batch_size=cfg.get("training", {}).get("batch_size", 64),
        epochs=cfg.get("training", {}).get("epochs", 100),
        lr_trunk=cfg.get("training", {}).get("lr_trunk", 1e-4),
        lr_embed=cfg.get("training", {}).get("lr_embed", 1e-2),
        weight_decay=cfg.get("training", {}).get("weight_decay", 0.01),
        warmup_ratio=cfg.get("training", {}).get("warmup_ratio", 0.1),
        grad_clip=cfg.get("training", {}).get("grad_clip", 1.0),
        supervision_weights=cfg.get("training", {}).get("supervision_weights", "uniform"),
        curriculum_start_recursions=cfg.get("training", {}).get("curriculum_start_recursions", 4),
        curriculum_full_recursions_epoch=cfg.get("training", {}).get("curriculum_full_recursions_epoch", 20),
        
        # Data
        data_dir=cfg.get("data", {}).get("data_dir", "data/arc-agi-1"),
        augment_factor=cfg.get("data", {}).get("augment_factor", 100),
        num_workers=cfg.get("data", {}).get("num_workers", 4),
        
        # Checkpointing
        save_dir=cfg.get("checkpoint", {}).get("save_dir", "checkpoints"),
        save_every=cfg.get("checkpoint", {}).get("save_every", 10),
        
        # Logging
        use_wandb=cfg.get("logging", {}).get("use_wandb", False),
        wandb_project=cfg.get("logging", {}).get("wandb_project", "trm-arc"),
        wandb_run_name=cfg.get("logging", {}).get("wandb_run_name"),
        
        # Hardware
        device=device,
        use_amp=cfg.get("hardware", {}).get("use_amp", True)
    )


def main():
    parser = argparse.ArgumentParser(description="Train TRM on ARC-AGI")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()
    
    print(f"Loading config from {args.config}")
    config = load_config(args.config)
    
    print(f"\nConfiguration:")
    for key, value in vars(config).items():
        print(f"  {key}: {value}")
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
