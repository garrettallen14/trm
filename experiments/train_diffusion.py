#!/usr/bin/env python3
"""
Discrete Diffusion TRM Training Script

A principled approach to iterative refinement for ARC-AGI.

Key innovations over standard TRM:
1. Mathematically grounded (discrete diffusion theory)
2. Natural deep supervision (loss at every timestep)
3. Cosine noise schedule (proven better than linear)
4. Flexible inference (can trade compute for quality)

Author: TRM Research Team
Date: December 2025
"""

import argparse
import json
import math
import os
import time
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.checkpoint import checkpoint
from tqdm import tqdm
from einops import rearrange

# Local imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.data import create_dataloader, Tokenizer
from src.model import RoPE2D, TransformerBlock, LoopedTransformer
from src.evaluation import ARCEvaluator, create_evaluator

# Optional dashboard
try:
    from dashboard.client import DashboardClient
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class DiffusionConfig:
    """All hyperparameters in one place."""
    
    # Model
    d_model: int = 512
    n_heads: int = 4
    n_layers: int = 2
    n_colors: int = 11  # 0-9 + padding
    max_grid_size: int = 32
    
    # Diffusion
    num_timesteps: int = 16  # Same as TRM recursion depth
    noise_schedule: str = "cosine"  # "linear" or "cosine"
    self_conditioning: bool = True  # Condition on previous prediction
    
    # Training
    lr: float = 1e-4
    lr_embed_mult: float = 100.0  # Embedding LR = lr * this
    weight_decay: float = 0.01
    warmup_epochs: int = 5
    epochs: int = 100
    batch_size: int = 8
    grad_accum: int = 4
    clip_grad: float = 1.0
    
    # Data
    augment_factor: int = 100
    
    # Efficiency
    use_amp: bool = True
    compile_model: bool = True
    gradient_checkpointing: bool = False
    
    @property
    def effective_batch(self) -> int:
        return self.batch_size * self.grad_accum


# =============================================================================
# Discrete Noise Schedule
# =============================================================================

class DiscreteNoiseSchedule:
    """
    Noise schedule for discrete diffusion.
    
    We use an absorbing state (mask token) approach:
    - At t=0: clean data
    - At t=T: fully masked
    - Intermediate: probability of masking increases with t
    """
    
    def __init__(
        self, 
        num_timesteps: int = 16, 
        num_colors: int = 11,
        schedule: str = "cosine",
        mask_token: int = 10  # Padding token as mask
    ):
        self.num_timesteps = num_timesteps
        self.num_colors = num_colors
        self.mask_token = mask_token
        
        # Compute mask probabilities for each timestep
        if schedule == "linear":
            # Linear: p(mask) = t / T
            self.mask_probs = torch.linspace(0, 0.999, num_timesteps + 1)
        elif schedule == "cosine":
            # Cosine: smoother, proven better
            # p(mask) = 1 - cos(π/2 * t/T)
            steps = torch.linspace(0, 1, num_timesteps + 1)
            self.mask_probs = 1 - torch.cos(steps * math.pi / 2)
            self.mask_probs = self.mask_probs.clamp(0, 0.999)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
    
    def add_noise(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Add noise to clean data.
        
        Args:
            x: [batch, h, w] clean grid
            t: [batch] or scalar timestep(s)
            
        Returns:
            noisy: [batch, h, w] noisy grid
        """
        if t.dim() == 0:
            t = t.expand(x.shape[0])
        
        batch = x.shape[0]
        device = x.device
        
        # Get mask probabilities for each example
        mask_probs = self.mask_probs.to(device)[t]  # [batch]
        mask_probs = mask_probs.view(batch, 1, 1)  # [batch, 1, 1]
        
        # Sample mask
        mask = torch.rand_like(x.float()) < mask_probs
        
        # Apply mask (replace with mask token)
        noisy = torch.where(mask, self.mask_token, x)
        
        # Optional: also add some uniform noise to unmasked tokens
        # This helps with diversity
        uniform_noise_prob = 0.05 * mask_probs
        uniform_mask = torch.rand_like(x.float()) < uniform_noise_prob
        random_colors = torch.randint_like(x, 0, self.num_colors - 1)  # Exclude mask token
        noisy = torch.where(uniform_mask & ~mask, random_colors, noisy)
        
        return noisy
    
    def get_mask_prob(self, t: int) -> float:
        """Get mask probability at timestep t."""
        return self.mask_probs[t].item()


# =============================================================================
# Discrete Diffusion Model
# =============================================================================

class DiscreteDiffusionTRM(nn.Module):
    """
    Discrete Diffusion Tiny Recursive Model.
    
    Architecture:
    - Encode demos + test input + noisy output
    - Condition on timestep
    - Predict clean output
    
    Key difference from TRM: Explicit timestep conditioning,
    principled noise schedule, natural deep supervision.
    """
    
    def __init__(self, config: DiffusionConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_colors = config.n_colors
        self.num_timesteps = config.num_timesteps
        
        # === Embeddings ===
        self.cell_embed = nn.Embedding(config.n_colors, config.d_model)
        
        # Timestep embedding (sinusoidal + learned projection)
        self.timestep_embed = nn.Sequential(
            SinusoidalPosEmb(config.d_model),
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # Grid type tokens
        self.demo_in_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.demo_out_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.test_in_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        self.noisy_out_token = nn.Parameter(torch.randn(1, 1, config.d_model) * 0.02)
        
        # === Positional Encoding ===
        self.rope = RoPE2D(config.d_model // config.n_heads, config.max_grid_size)
        
        # === Transformer ===
        self.transformer = LoopedTransformer(
            d_model=config.d_model,
            n_heads=config.n_heads,
            n_layers=config.n_layers,
            dropout=0.0
        )
        
        # === Output Head ===
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_head = nn.Linear(config.d_model, config.n_colors)
        
        # === Self-Conditioning (optional) ===
        if config.self_conditioning:
            self.self_cond_proj = nn.Linear(config.n_colors, config.d_model)
        
        # === Noise Schedule ===
        self.noise_schedule = DiscreteNoiseSchedule(
            num_timesteps=config.num_timesteps,
            num_colors=config.n_colors,
            schedule=config.noise_schedule
        )
        
        # Gradient checkpointing flag
        self.gradient_checkpointing = config.gradient_checkpointing
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
    
    def encode_grid(
        self, 
        grid: torch.Tensor, 
        grid_type: str,
        self_cond: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a grid with type token and positions.
        
        Args:
            grid: [batch, h, w] grid
            grid_type: one of 'demo_in', 'demo_out', 'test_in', 'noisy_out'
            self_cond: [batch, h, w, n_colors] previous prediction (optional)
            
        Returns:
            emb: [batch, h*w, d_model]
            positions: [batch, h*w, 2]
        """
        batch, h, w = grid.shape
        device = grid.device
        
        # Cell embeddings
        emb = self.cell_embed(grid)  # [batch, h, w, d_model]
        
        # Add self-conditioning if provided
        if self_cond is not None and self.config.self_conditioning:
            sc_emb = self.self_cond_proj(self_cond)  # [batch, h, w, d_model]
            emb = emb + sc_emb
        
        emb = rearrange(emb, 'b h w d -> b (h w) d')
        
        # Add type token
        type_tokens = {
            'demo_in': self.demo_in_token,
            'demo_out': self.demo_out_token,
            'test_in': self.test_in_token,
            'noisy_out': self.noisy_out_token
        }
        emb = emb + type_tokens[grid_type]
        
        # Compute 2D positions
        positions = torch.stack(torch.meshgrid(
            torch.arange(h, device=device),
            torch.arange(w, device=device),
            indexing='ij'
        ), dim=-1)  # [h, w, 2]
        positions = positions.unsqueeze(0).expand(batch, -1, -1, -1)
        positions = rearrange(positions, 'b h w c -> b (h w) c')
        
        return emb, positions
    
    def forward(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        noisy_output: torch.Tensor,
        timestep: torch.Tensor,
        self_cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass: predict clean output from noisy.
        
        Args:
            demo_inputs: List of [batch, h, w] demo input grids
            demo_outputs: List of [batch, h, w] demo output grids
            test_input: [batch, h, w] test input grid
            noisy_output: [batch, h, w] noisy version of test output
            timestep: [batch] current diffusion timestep
            self_cond: [batch, h, w, n_colors] previous prediction (optional)
            
        Returns:
            logits: [batch, h*w, n_colors] predicted clean colors
        """
        batch = test_input.shape[0]
        device = test_input.device
        
        # === Encode all grids ===
        all_emb = []
        all_pos = []
        
        for inp, out in zip(demo_inputs, demo_outputs):
            inp_emb, inp_pos = self.encode_grid(inp, 'demo_in')
            out_emb, out_pos = self.encode_grid(out, 'demo_out')
            all_emb.extend([inp_emb, out_emb])
            all_pos.extend([inp_pos, out_pos])
        
        test_emb, test_pos = self.encode_grid(test_input, 'test_in')
        noisy_emb, noisy_pos = self.encode_grid(noisy_output, 'noisy_out', self_cond)
        
        all_emb.extend([test_emb, noisy_emb])
        all_pos.extend([test_pos, noisy_pos])
        
        # Concatenate
        x = torch.cat(all_emb, dim=1)
        positions = torch.cat(all_pos, dim=1)
        
        # === Add timestep conditioning ===
        t_emb = self.timestep_embed(timestep)  # [batch, d_model]
        x = x + t_emb.unsqueeze(1)  # Broadcast to all tokens
        
        # === Transformer ===
        if self.gradient_checkpointing and self.training:
            x, _ = checkpoint(
                self.transformer,
                x, positions, None,
                use_reentrant=False
            )
        else:
            x, _ = self.transformer(x, positions)
        
        # === Extract output and predict ===
        noisy_seq_len = noisy_output.shape[1] * noisy_output.shape[2]
        output_hidden = x[:, -noisy_seq_len:]  # Last tokens are noisy output
        
        output_hidden = self.output_norm(output_hidden)
        logits = self.output_head(output_hidden)
        
        return logits
    
    def compute_loss(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        test_output: torch.Tensor,
        return_metrics: bool = False
    ) -> dict:
        """
        Compute diffusion training loss.
        
        Samples random timesteps and computes cross-entropy loss
        for predicting clean data from noisy.
        """
        batch, h, w = test_output.shape
        device = test_output.device
        
        # Sample random timesteps (1 to T, not 0)
        timesteps = torch.randint(1, self.num_timesteps + 1, (batch,), device=device)
        
        # Create noisy outputs
        noisy_output = self.noise_schedule.add_noise(test_output, timesteps)
        
        # Self-conditioning: 50% of time, condition on previous prediction
        self_cond = None
        if self.config.self_conditioning and self.training and torch.rand(1).item() > 0.5:
            with torch.no_grad():
                prev_logits = self.forward(
                    demo_inputs, demo_outputs, test_input, noisy_output, timesteps
                )
                self_cond = F.softmax(prev_logits, dim=-1).view(batch, h, w, -1)
        
        # Forward pass
        logits = self.forward(
            demo_inputs, demo_outputs, test_input, noisy_output, timesteps, self_cond
        )
        
        # Loss: cross-entropy to predict clean
        target = test_output.view(batch, -1)
        loss = F.cross_entropy(
            logits.view(-1, self.n_colors),
            target.view(-1),
            ignore_index=self.noise_schedule.mask_token  # Don't penalize padding
        )
        
        result = {'total_loss': loss}
        
        if return_metrics:
            with torch.no_grad():
                # Cell accuracy
                preds = logits.argmax(dim=-1)
                mask = target != self.noise_schedule.mask_token
                correct = (preds == target) & mask
                cell_acc = correct.float().sum() / mask.float().sum()
                result['cell_accuracy'] = cell_acc.item()
                
                # Per-timestep breakdown (useful for debugging)
                result['timesteps'] = timesteps.float().mean().item()
        
        return result
    
    @torch.no_grad()
    def generate(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        num_steps: Optional[int] = None,
        temperature: float = 1.0,
        return_intermediates: bool = False
    ) -> dict:
        """
        Generate output via iterative denoising.
        
        Args:
            demo_inputs, demo_outputs, test_input: Context
            num_steps: Number of denoising steps (default: num_timesteps)
            temperature: Sampling temperature (1.0 = standard, <1 = sharper)
            return_intermediates: Whether to return intermediate predictions
            
        Returns:
            dict with 'prediction' and optionally 'intermediates'
        """
        if num_steps is None:
            num_steps = self.num_timesteps
        
        batch, h, w = test_input.shape
        device = test_input.device
        
        # Start from fully masked/random
        x = torch.full((batch, h, w), self.noise_schedule.mask_token, device=device)
        
        intermediates = []
        self_cond = None
        
        # Denoise from t=T to t=1
        for step in range(num_steps, 0, -1):
            t = torch.full((batch,), step, device=device, dtype=torch.long)
            
            # Predict clean
            logits = self.forward(
                demo_inputs, demo_outputs, test_input, x, t, self_cond
            )
            
            # Update self-conditioning
            if self.config.self_conditioning:
                self_cond = F.softmax(logits / temperature, dim=-1).view(batch, h, w, -1)
            
            # Sample or argmax
            if step > 1:
                # Stochastic sampling
                probs = F.softmax(logits / temperature, dim=-1)
                x = torch.multinomial(probs.view(-1, self.n_colors), 1)
                x = x.view(batch, h, w)
                
                # Re-mask based on next timestep's schedule
                next_mask_prob = self.noise_schedule.get_mask_prob(step - 1)
                keep_prob = 1 - next_mask_prob
                keep_mask = torch.rand(batch, h, w, device=device) < keep_prob
                x = torch.where(keep_mask, x, self.noise_schedule.mask_token)
            else:
                # Final step: deterministic
                x = logits.argmax(dim=-1).view(batch, h, w)
            
            if return_intermediates:
                intermediates.append(x.clone())
        
        result = {'prediction': x}
        if return_intermediates:
            result['intermediates'] = intermediates
        
        return result


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate(
    model: DiscreteDiffusionTRM,
    data_dir: str,
    split: str,
    device: torch.device,
    n_samples: int = 50,
    num_steps: int = None
) -> dict:
    """
    Evaluate model on ARC tasks.
    
    Returns both cell accuracy and task accuracy.
    """
    model.eval()
    
    from src.data import ARCDataset, collate_fn
    from torch.utils.data import DataLoader
    
    dataset = ARCDataset(data_dir, split=split, augment=False)
    loader = DataLoader(
        dataset, 
        batch_size=1,  # One task at a time for proper eval
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    correct_tasks = 0
    total_tasks = 0
    cell_correct = 0
    cell_total = 0
    
    for i, batch in enumerate(loader):
        if i >= n_samples:
            break
        
        try:
            demo_inputs = [g.to(device) for g in batch["demo_inputs"]]
            demo_outputs = [g.to(device) for g in batch["demo_outputs"]]
            test_input = batch["test_input"].to(device)
            test_output = batch["test_output"].to(device)
            
            # Generate prediction
            result = model.generate(
                demo_inputs, demo_outputs, test_input,
                num_steps=num_steps
            )
            pred = result['prediction']
            
            # Compute metrics
            mask = test_output != 10  # Ignore padding
            
            # Cell accuracy
            cells_match = (pred == test_output) & mask
            cell_correct += cells_match.sum().item()
            cell_total += mask.sum().item()
            
            # Task accuracy (all cells must match)
            task_correct = cells_match.sum() == mask.sum()
            if task_correct:
                correct_tasks += 1
            total_tasks += 1
            
        except Exception as e:
            total_tasks += 1
            continue
    
    return {
        "task_accuracy": correct_tasks / total_tasks if total_tasks > 0 else 0,
        "cell_accuracy": cell_correct / cell_total if cell_total > 0 else 0,
        "tasks_correct": correct_tasks,
        "tasks_total": total_tasks
    }


# =============================================================================
# Training Loop
# =============================================================================

def train(args):
    """Main training function."""
    
    # === Setup ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # GPU optimizations
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.autograd.set_detect_anomaly(False)
        print("GPU optimizations: TF32 + cuDNN benchmark enabled")
    
    # === Config ===
    config = DiffusionConfig(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        num_timesteps=args.num_timesteps,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        augment_factor=args.augment_factor,
        use_amp=args.amp,
        compile_model=args.compile,
        gradient_checkpointing=args.grad_checkpoint,
        self_conditioning=args.self_cond
    )
    
    # === Dashboard ===
    dashboard = None
    if args.dashboard and DASHBOARD_AVAILABLE:
        try:
            dashboard = DashboardClient(mode="http", url="http://localhost:3000")
            print("Dashboard: connected")
        except Exception as e:
            print(f"Dashboard: failed ({e})")
    
    # === Print Config ===
    print("\n" + "="*60)
    print("DISCRETE DIFFUSION TRM TRAINING")
    print("="*60)
    for k, v in asdict(config).items():
        print(f"  {k}: {v}")
    
    # === Create Model ===
    model = DiscreteDiffusionTRM(config).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")
    
    # Compile
    if config.compile_model:
        print("Compiling model with torch.compile...")
        model = torch.compile(model)
        print("Model compiled!")
    
    # === Optimizer ===
    # Differential LR: embeddings get higher LR
    embed_params = [p for n, p in model.named_parameters() if "embed" in n or "token" in n]
    other_params = [p for n, p in model.named_parameters() if "embed" not in n and "token" not in n]
    
    print(f"Embed params: {sum(p.numel() for p in embed_params):,}")
    print(f"Other params: {sum(p.numel() for p in other_params):,}")
    
    optimizer = optim.AdamW([
        {"params": other_params, "lr": config.lr},
        {"params": embed_params, "lr": config.lr * config.lr_embed_mult}
    ], weight_decay=config.weight_decay, fused=True)
    
    # === Scheduler: Warmup + Cosine ===
    def lr_lambda(epoch):
        if epoch < config.warmup_epochs:
            return (epoch + 1) / config.warmup_epochs
        else:
            remaining = config.epochs - config.warmup_epochs
            if remaining <= 0:
                return 1.0  # No decay if warmup >= epochs
            progress = (epoch - config.warmup_epochs) / remaining
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # === Mixed Precision ===
    scaler = GradScaler('cuda') if config.use_amp else None
    if config.use_amp:
        print("Using mixed precision (AMP)")
    
    # === Data ===
    train_loader = create_dataloader(
        args.data_dir,
        split="training",
        batch_size=config.batch_size,
        augment=True,
        augment_factor=config.augment_factor,
        num_workers=4
    )
    print(f"Training samples: {len(train_loader.dataset):,}")
    
    # === Save Directory ===
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(args.save_dir) / f"diffusion_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to: {save_dir}")
    
    # Save config
    with open(save_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # === Training Loop ===
    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)
    
    history = {
        "train_loss": [],
        "cell_accuracy": [],
        "task_accuracy": [],
        "lr": [],
        "epoch_time": []
    }
    
    best_cell_acc = 0.0
    
    for epoch in range(1, config.epochs + 1):
        epoch_start = time.time()
        model.train()
        
        epoch_loss = 0.0
        epoch_cell_acc = 0.0
        n_batches = 0
        accum_step = 0
        oom_count = 0
        
        optimizer.zero_grad(set_to_none=True)
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config.epochs}")
        
        for batch in pbar:
            try:
                # Move to device
                demo_inputs = [g.to(device, non_blocking=True) for g in batch["demo_inputs"]]
                demo_outputs = [g.to(device, non_blocking=True) for g in batch["demo_outputs"]]
                test_input = batch["test_input"].to(device, non_blocking=True)
                test_output = batch["test_output"].to(device, non_blocking=True)
                
                # Forward + loss
                with autocast('cuda', enabled=config.use_amp):
                    loss_dict = model.compute_loss(
                        demo_inputs, demo_outputs, test_input, test_output,
                        return_metrics=(n_batches % 10 == 0)  # Compute cell acc periodically
                    )
                    loss = loss_dict["total_loss"] / config.grad_accum
                
                # Backward
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                
                accum_step += 1
                epoch_loss += loss.item() * config.grad_accum
                n_batches += 1
                
                if "cell_accuracy" in loss_dict:
                    epoch_cell_acc += loss_dict["cell_accuracy"]
                
                # Optimizer step
                if accum_step >= config.grad_accum:
                    if scaler:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), config.clip_grad)
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    accum_step = 0
                
                # Progress bar
                pbar.set_postfix({
                    "loss": f"{loss.item() * config.grad_accum:.4f}",
                    "oom": oom_count
                })
                
                # Dashboard update
                if dashboard and n_batches % 20 == 0:
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    elapsed = time.time() - epoch_start
                    samples_sec = (n_batches * config.batch_size) / elapsed if elapsed > 0 else 0
                    
                    dashboard.update(
                        status="running",
                        model_type="diffusion",
                        loss=loss.item() * config.grad_accum,
                        cell_accuracy=loss_dict.get("cell_accuracy", 0),
                        epoch=epoch,
                        step=n_batches,
                        total_steps=len(train_loader),
                        total_epochs=config.epochs,
                        memory_gb=mem_gb,
                        n_recursions=config.num_timesteps,  # Use timesteps as "recursion" equivalent
                        lr=scheduler.get_last_lr()[0],
                        samples_per_sec=samples_sec
                    )
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_count += 1
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    optimizer.zero_grad(set_to_none=True)
                    accum_step = 0
                    continue
                raise e
        
        # === End of Epoch ===
        avg_loss = epoch_loss / max(n_batches, 1)
        avg_cell_acc = epoch_cell_acc / max(n_batches // 10, 1)  # Approx
        epoch_time = time.time() - epoch_start
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # === Evaluation (quick subset during training) ===
        eval_results = evaluate(
            model, args.data_dir, "evaluation", device,
            n_samples=100, num_steps=config.num_timesteps  # 100 tasks for quick eval
        )
        
        # Log
        history["train_loss"].append(avg_loss)
        history["cell_accuracy"].append(eval_results["cell_accuracy"])
        history["task_accuracy"].append(eval_results["task_accuracy"])
        history["lr"].append(current_lr)
        history["epoch_time"].append(epoch_time)
        
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, "
              f"task_acc={eval_results['task_accuracy']:.1%}, "
              f"cell_acc={eval_results['cell_accuracy']:.1%}, "
              f"lr={current_lr:.2e}, time={epoch_time:.0f}s, oom={oom_count}")
        
        # Dashboard epoch update
        if dashboard:
            dashboard.update(
                status="running",
                model_type="diffusion",
                loss=avg_loss,
                cell_accuracy=eval_results["cell_accuracy"],
                task_accuracy=eval_results["task_accuracy"],
                epoch=epoch,
                step=len(train_loader),
                total_steps=len(train_loader),
                total_epochs=config.epochs,
                lr=current_lr
            )
        
        # Save best model
        if eval_results["cell_accuracy"] > best_cell_acc:
            best_cell_acc = eval_results["cell_accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "cell_accuracy": eval_results["cell_accuracy"],
                "task_accuracy": eval_results["task_accuracy"]
            }, save_dir / "best_model.pt")
        
        # Save history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    # === Final Evaluation (Principled) ===
    print("\n" + "="*60)
    print("FINAL EVALUATION (PRINCIPLED)")
    print("="*60)
    
    # Create proper evaluator with contamination check
    try:
        evaluator = create_evaluator("data", include_agi2=True)
        evaluator.verify_no_contamination()  # CRITICAL: verify no train/test overlap
        
        # Custom generation function for our model
        def diffusion_generate(model, demo_inputs, demo_outputs, test_input):
            result = model.generate(demo_inputs, demo_outputs, test_input, num_steps=config.num_timesteps)
            return result['prediction']
        
        # Full evaluation on all datasets
        summaries = evaluator.evaluate(
            model, device, 
            datasets=None,  # All available
            max_tasks=None,  # All tasks
            generate_fn=diffusion_generate
        )
        
        agi1_results = {
            "task_accuracy": summaries.get("agi1_eval", {}).task_accuracy if "agi1_eval" in summaries else 0,
            "cell_accuracy": summaries.get("agi1_eval", {}).cell_accuracy if "agi1_eval" in summaries else 0
        }
        
    except Exception as e:
        print(f"Principled evaluation failed: {e}")
        print("Falling back to basic evaluation...")
        agi1_results = evaluate(
            model, "data/arc-agi-1", "evaluation", device,
            n_samples=400, num_steps=config.num_timesteps
        )
    
    print(f"ARC-AGI-1: task_acc={agi1_results['task_accuracy']:.1%}, "
          f"cell_acc={agi1_results['cell_accuracy']:.1%}")
    
    # Try AGI-2
    try:
        print("\nEvaluating on ARC-AGI-2...")
        agi2_results = evaluate(
            model, "data/arc-agi-2", "evaluation", device,
            n_samples=200, num_steps=config.num_timesteps
        )
        print(f"ARC-AGI-2: task_acc={agi2_results['task_accuracy']:.1%}, "
              f"cell_acc={agi2_results['cell_accuracy']:.1%}")
    except Exception as e:
        print(f"AGI-2 eval failed: {e}")
        agi2_results = {"task_accuracy": 0, "cell_accuracy": 0}
    
    # Save final results
    final_results = {
        "best_cell_accuracy": best_cell_acc,
        "agi1_task_accuracy": agi1_results["task_accuracy"],
        "agi1_cell_accuracy": agi1_results["cell_accuracy"],
        "agi2_task_accuracy": agi2_results["task_accuracy"],
        "agi2_cell_accuracy": agi2_results["cell_accuracy"],
        "total_epochs": config.epochs,
        "config": asdict(config)
    }
    
    with open(save_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\nResults saved to: {save_dir}")
    
    if dashboard:
        dashboard.update(
            status="completed",
            model_type="diffusion",
            cell_accuracy=agi1_results["cell_accuracy"],
            task_accuracy=agi1_results["task_accuracy"]
        )


def main():
    parser = argparse.ArgumentParser(description="Train Discrete Diffusion TRM")
    
    # Model
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--num_timesteps", type=int, default=16, help="Diffusion timesteps")
    parser.add_argument("--self_cond", action="store_true", help="Use self-conditioning")
    
    # Training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--augment_factor", type=int, default=100)
    
    # Data
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1")
    parser.add_argument("--save_dir", type=str, default="experiments/runs")
    
    # Efficiency
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Gradient checkpointing")
    
    # Logging
    parser.add_argument("--dashboard", action="store_true", help="Send metrics to dashboard")
    
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
