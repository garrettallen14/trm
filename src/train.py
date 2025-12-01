"""
Training loop for Tiny Recursive Model.

Features:
- Deep supervision at each recursion step
- Differential learning rates (trunk vs embeddings)
- Gradient clipping for stability
- Truncated BPTT for memory efficiency
- Attention entropy monitoring
- Wandb logging
"""

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

from .model import TinyRecursiveModel, count_parameters
from .data import create_dataloader


@dataclass
class TrainConfig:
    """Training configuration."""
    # Model
    d_model: int = 512
    n_heads: int = 4
    n_layers: int = 2
    n_recursions: int = 16
    dropout: float = 0.0
    
    # Training
    batch_size: int = 64
    epochs: int = 100
    lr_trunk: float = 1e-4
    lr_embed: float = 1e-2  # 100x higher for embeddings
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    grad_clip: float = 1.0
    
    # Deep supervision
    supervision_weights: str = "uniform"  # "uniform" or "linear"
    
    # Data
    data_dir: str = "data/arc-agi-1"
    augment_factor: int = 100
    num_workers: int = 4
    
    # Checkpointing
    save_dir: str = "checkpoints"
    save_every: int = 10
    
    # Logging
    use_wandb: bool = True
    wandb_project: str = "trm-arc"
    wandb_run_name: Optional[str] = None
    
    # Hardware
    device: str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    use_amp: bool = True  # Automatic mixed precision
    
    # Curriculum
    curriculum_start_recursions: int = 4
    curriculum_full_recursions_epoch: int = 20


class Trainer:
    """Trainer for TRM."""
    
    def __init__(self, config: TrainConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Create model
        self.model = TinyRecursiveModel(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            n_recursions=config.n_recursions,
            dropout=config.dropout
        ).to(self.device)
        
        # Print model info
        param_counts = count_parameters(self.model)
        print(f"Model parameters: {param_counts['total']:,}")
        
        # Create optimizer with differential learning rates
        self._create_optimizer()
        
        # Create data loaders
        self._create_dataloaders()
        
        # Mixed precision
        self.scaler = GradScaler() if config.use_amp and self.device.type == "cuda" else None
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create save directory
        self.save_dir = Path(config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize wandb
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=vars(config)
            )
            wandb.watch(self.model, log_freq=100)
    
    def _create_optimizer(self):
        """Create optimizer with differential learning rates."""
        config = self.config
        
        # Separate parameters
        embed_params = []
        trunk_params = []
        
        for name, param in self.model.named_parameters():
            if "embed" in name or "token" in name:
                embed_params.append(param)
            else:
                trunk_params.append(param)
        
        self.optimizer = optim.AdamW([
            {"params": trunk_params, "lr": config.lr_trunk},
            {"params": embed_params, "lr": config.lr_embed}
        ], weight_decay=config.weight_decay)
        
        print(f"Trunk params: {sum(p.numel() for p in trunk_params):,} @ lr={config.lr_trunk}")
        print(f"Embed params: {sum(p.numel() for p in embed_params):,} @ lr={config.lr_embed}")
    
    def _create_dataloaders(self):
        """Create training and validation dataloaders."""
        config = self.config
        
        # Training data
        self.train_loader = create_dataloader(
            data_dir=config.data_dir,
            split="training",
            batch_size=config.batch_size,
            augment=True,
            augment_factor=config.augment_factor,
            num_workers=config.num_workers,
            shuffle=True
        )
        
        # Validation data (no augmentation)
        self.val_loader = create_dataloader(
            data_dir=config.data_dir,
            split="evaluation",
            batch_size=config.batch_size,
            augment=False,
            num_workers=config.num_workers,
            shuffle=False
        )
        
        print(f"Training samples: {len(self.train_loader.dataset):,}")
        print(f"Validation samples: {len(self.val_loader.dataset):,}")
    
    def _get_lr_schedule(self, step: int, total_steps: int) -> float:
        """Linear warmup then linear decay."""
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        if step < warmup_steps:
            return step / warmup_steps
        else:
            return 1.0 - (step - warmup_steps) / (total_steps - warmup_steps)
    
    def _get_current_recursions(self, epoch: int) -> int:
        """Get number of recursions for current epoch (curriculum)."""
        config = self.config
        
        if epoch >= config.curriculum_full_recursions_epoch:
            return config.n_recursions
        
        # Linear ramp
        progress = epoch / config.curriculum_full_recursions_epoch
        n_rec = int(config.curriculum_start_recursions + 
                    progress * (config.n_recursions - config.curriculum_start_recursions))
        return max(config.curriculum_start_recursions, n_rec)
    
    def train_epoch(self, epoch: int) -> dict:
        """Train for one epoch."""
        self.model.train()
        config = self.config
        
        total_loss = 0.0
        total_final_loss = 0.0
        total_entropy = 0.0
        n_batches = 0
        
        # Get current recursion depth (curriculum)
        n_recursions = self._get_current_recursions(epoch)
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch in pbar:
            # Move to device
            demo_inputs = [g.to(self.device) for g in batch["demo_inputs"]]
            demo_outputs = [g.to(self.device) for g in batch["demo_outputs"]]
            test_input = batch["test_input"].to(self.device)
            test_output = batch["test_output"].to(self.device)
            
            # Forward pass with optional AMP
            with autocast(enabled=self.scaler is not None):
                loss_dict = self.model.compute_loss(
                    demo_inputs=demo_inputs,
                    demo_outputs=demo_outputs,
                    test_input=test_input,
                    test_output=test_output,
                    n_recursions=n_recursions,
                    supervision_weights=config.supervision_weights
                )
                loss = loss_dict["total_loss"]
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), config.grad_clip)
                self.optimizer.step()
            
            # Update learning rate
            self.global_step += 1
            total_steps = len(self.train_loader) * config.epochs
            lr_scale = self._get_lr_schedule(self.global_step, total_steps)
            
            for param_group in self.optimizer.param_groups:
                if param_group["lr"] == config.lr_trunk:
                    param_group["lr"] = config.lr_trunk * lr_scale
                else:
                    param_group["lr"] = config.lr_embed * lr_scale
            
            # Track metrics
            total_loss += loss.item()
            total_final_loss += loss_dict["final_loss"].item()
            total_entropy += loss_dict["attn_entropy"].item()
            n_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "entropy": f"{loss_dict['attn_entropy'].item():.2f}",
                "rec": n_recursions
            })
            
            # Log to wandb
            if config.use_wandb and WANDB_AVAILABLE and self.global_step % 10 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/final_loss": loss_dict["final_loss"].item(),
                    "train/attn_entropy": loss_dict["attn_entropy"].item(),
                    "train/n_recursions": n_recursions,
                    "train/lr": self.optimizer.param_groups[0]["lr"],
                    "step": self.global_step
                })
        
        return {
            "loss": total_loss / n_batches,
            "final_loss": total_final_loss / n_batches,
            "attn_entropy": total_entropy / n_batches,
            "n_recursions": n_recursions
        }
    
    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation."""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_cells = 0
        n_batches = 0
        
        for batch in tqdm(self.val_loader, desc="Validating"):
            demo_inputs = [g.to(self.device) for g in batch["demo_inputs"]]
            demo_outputs = [g.to(self.device) for g in batch["demo_outputs"]]
            test_input = batch["test_input"].to(self.device)
            test_output = batch["test_output"].to(self.device)
            
            # Compute loss
            loss_dict = self.model.compute_loss(
                demo_inputs=demo_inputs,
                demo_outputs=demo_outputs,
                test_input=test_input,
                test_output=test_output,
                supervision_weights=self.config.supervision_weights
            )
            
            # Compute accuracy
            predictions = self.model.predict(demo_inputs, demo_outputs, test_input)
            
            # Mask out padding (token 10)
            mask = test_output != 10
            correct = ((predictions == test_output) & mask).sum().item()
            total = mask.sum().item()
            
            total_loss += loss_dict["total_loss"].item()
            total_correct += correct
            total_cells += total
            n_batches += 1
        
        return {
            "val_loss": total_loss / n_batches,
            "val_accuracy": total_correct / total_cells if total_cells > 0 else 0
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "config": vars(self.config),
            "metrics": metrics
        }
        
        # Save latest
        path = self.save_dir / "latest.pt"
        torch.save(checkpoint, path)
        
        # Save best
        if metrics.get("val_loss", float('inf')) < self.best_val_loss:
            self.best_val_loss = metrics["val_loss"]
            best_path = self.save_dir / "best.pt"
            torch.save(checkpoint, best_path)
            print(f"New best model saved (val_loss: {self.best_val_loss:.4f})")
        
        # Save periodic
        if epoch % self.config.save_every == 0:
            epoch_path = self.save_dir / f"epoch_{epoch:04d}.pt"
            torch.save(checkpoint, epoch_path)
    
    def load_checkpoint(self, path: str | Path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = checkpoint["global_step"]
        
        return checkpoint["epoch"]
    
    def train(self):
        """Full training loop."""
        config = self.config
        
        print(f"\n{'='*60}")
        print(f"Starting training on {self.device}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        
        for epoch in range(1, config.epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate()
            
            # Combine metrics
            metrics = {**train_metrics, **val_metrics}
            
            # Log
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{config.epochs}")
            print(f"  Train loss: {metrics['loss']:.4f}, Final: {metrics['final_loss']:.4f}")
            print(f"  Val loss: {metrics['val_loss']:.4f}, Accuracy: {metrics['val_accuracy']:.2%}")
            print(f"  Attention entropy: {metrics['attn_entropy']:.2f}")
            print(f"  Recursions: {metrics['n_recursions']}, Time: {epoch_time:.1f}s")
            
            if config.use_wandb and WANDB_AVAILABLE:
                wandb.log({
                    "epoch": epoch,
                    "val/loss": metrics["val_loss"],
                    "val/accuracy": metrics["val_accuracy"],
                    "time/epoch": epoch_time
                })
            
            # Save
            self.save_checkpoint(epoch, metrics)
        
        total_time = time.time() - start_time
        print(f"\nTraining complete! Total time: {total_time/3600:.2f} hours")
        
        if config.use_wandb and WANDB_AVAILABLE:
            wandb.finish()


def main():
    """Main training entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TRM on ARC-AGI")
    
    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_recursions", type=int, default=16)
    
    # Training
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr_trunk", type=float, default=1e-4)
    parser.add_argument("--lr_embed", type=float, default=1e-2)
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1")
    parser.add_argument("--augment_factor", type=int, default=100)
    
    # Checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    
    # Logging
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--wandb_project", type=str, default="trm-arc")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    args = parser.parse_args()
    
    config = TrainConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_recursions=args.n_recursions,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr_trunk=args.lr_trunk,
        lr_embed=args.lr_embed,
        data_dir=args.data_dir,
        augment_factor=args.augment_factor,
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name
    )
    
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
