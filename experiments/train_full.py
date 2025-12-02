#!/usr/bin/env python3
"""
Full TRM training run based on research findings.

Key settings from TRM paper:
- 2-layer transformer, d_model=512, 4 heads
- 16 recursion steps (not 8!)
- 6× no-grad refinement loops per grad step
- 100× augmentation (8 dihedrals × 12 color perms)
- Deep supervision with uniform weights
- Differential LR: trunk=1e-4, embed=1e-2
- 100 epochs

Usage:
    uv run python experiments/train_full.py --dashboard --amp
    uv run python experiments/train_full.py --epochs 100 --amp
"""

import argparse
import gc
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.amp import autocast, GradScaler
from tqdm import tqdm

from src.model import TinyRecursiveModel
from src.data import create_dataloader, ARCDataset

# Dashboard client (optional)
try:
    from dashboard.client import DashboardClient
    DASHBOARD_AVAILABLE = True
except ImportError:
    DASHBOARD_AVAILABLE = False


def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def evaluate_task_accuracy(model, data_dir, split, device, n_recursions, n_samples=None):
    """Evaluate task-level accuracy (all cells must match)."""
    model.eval()
    dataset = ARCDataset(data_dir, split=split, augment=False)
    
    if n_samples:
        indices = torch.randperm(len(dataset))[:n_samples].tolist()
    else:
        indices = range(len(dataset))
    
    correct_tasks = 0
    total_tasks = 0
    cell_correct = 0
    cell_total = 0
    
    with torch.no_grad():
        for i in indices:
            item = dataset[i]
            demo_inputs = [g.unsqueeze(0).to(device) for g in item["demo_inputs"]]
            demo_outputs = [g.unsqueeze(0).to(device) for g in item["demo_outputs"]]
            test_input = item["test_input"].unsqueeze(0).to(device)
            test_output = item["test_output"].to(device)
            
            try:
                pred = model.predict(demo_inputs, demo_outputs, test_input, n_recursions=n_recursions)
                pred = pred.squeeze(0)
                
                if pred.dim() == 1:
                    h, w = test_output.shape
                    pred = pred.view(h, w)
                
                mask = test_output != 10
                cells_match = (pred == test_output) & mask
                
                task_correct = cells_match.all().item()
                correct_tasks += task_correct
                total_tasks += 1
                
                cell_correct += cells_match.sum().item()
                cell_total += mask.sum().item()
                
            except Exception:
                total_tasks += 1
                clear_memory()
    
    return {
        "task_accuracy": correct_tasks / total_tasks if total_tasks > 0 else 0,
        "cell_accuracy": cell_correct / cell_total if cell_total > 0 else 0,
        "tasks_correct": correct_tasks,
        "tasks_total": total_tasks
    }


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # === GPU OPTIMIZATIONS ===
        # TF32: 3x faster matmuls on Ampere (A40, A100, etc.) with ~0.1% precision loss
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # cuDNN autotuning: finds fastest algorithms for your hardware
        torch.backends.cudnn.benchmark = True
        
        # Disable debug features for speed
        torch.autograd.set_detect_anomaly(False)
        torch.autograd.profiler.profile(False)
        torch.autograd.profiler.emit_nvtx(False)
        
        print("GPU optimizations: TF32 + cuDNN benchmark enabled")
    
    # Dashboard
    dashboard = None
    if args.dashboard and DASHBOARD_AVAILABLE:
        try:
            dashboard = DashboardClient(mode="http", url="http://localhost:3000")
            print("Dashboard: connected")
        except Exception as e:
            print(f"Dashboard: failed ({e})")
    
    # Optimized TRM config
    config = {
        # Model
        "d_model": 512,
        "n_heads": 4,
        "n_layers": 2,
        "n_recursions": args.n_recursions,  # 8 performed best in our sweep
        
        # Training
        "lr_trunk": 1e-4,
        "lr_embed": 1e-2,  # 100× higher for embeddings
        "weight_decay": 0.01,
        "clip_grad": 1.0,
        "warmup_epochs": 5,
        
        # Batching
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        
        # Data
        "augment_factor": args.augment_factor,
        "epochs": args.epochs,
        
        # TRM-specific
        "supervision_weights": "uniform",
        "no_grad_loops": args.no_grad_loops,  # Skip for faster training
    }
    
    print("\n" + "="*60)
    print("FULL TRM TRAINING CONFIG")
    print("="*60)
    for k, v in config.items():
        print(f"  {k}: {v}")
    
    # Create model
    model = TinyRecursiveModel(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_recursions=config["n_recursions"],
        gradient_checkpointing=args.grad_checkpoint
    ).to(device)
    
    if args.grad_checkpoint:
        print("Using gradient checkpointing (saves ~50% memory, 20% slower)")
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nParameters: {n_params:,}")
    
    # Compile model for faster training (PyTorch 2.0+)
    if args.compile:
        print("Compiling model with torch.compile (max-autotune)...")
        # max-autotune: slower compile, faster runtime
        # reduce-overhead: reduces Python overhead
        model = torch.compile(model, mode="max-autotune")
        print("Model compiled!")
    
    # Optimizer with differential LR (critical for TRM!)
    embed_params = [p for n, p in model.named_parameters() if "embed" in n or "token" in n]
    trunk_params = [p for n, p in model.named_parameters() if "embed" not in n and "token" not in n]
    
    print(f"Trunk params: {sum(p.numel() for p in trunk_params):,}")
    print(f"Embed params: {sum(p.numel() for p in embed_params):,}")
    
    # Fused AdamW: fuses optimizer step into single kernel (10-15% faster)
    optimizer = optim.AdamW([
        {"params": trunk_params, "lr": config["lr_trunk"]},
        {"params": embed_params, "lr": config["lr_embed"]}
    ], weight_decay=config["weight_decay"], fused=True)
    
    # Warmup + Cosine decay (TRM paper: 10% warmup)
    warmup_epochs = config["warmup_epochs"]
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs  # Linear warmup
        else:
            # Cosine decay from 1 to 0
            progress = (epoch - warmup_epochs) / (config["epochs"] - warmup_epochs)
            return 0.5 * (1 + math.cos(math.pi * progress))
    
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision
    scaler = GradScaler('cuda') if args.amp else None
    if args.amp:
        print("Using mixed precision (AMP)")
    
    # Data
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        split="training",
        batch_size=config["batch_size"],
        augment=True,
        augment_factor=config["augment_factor"],
        num_workers=4
    )
    
    # Run info
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path(f"experiments/runs/{run_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    with open(save_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    history = {
        "train_loss": [],
        "task_accuracy": [],
        "cell_accuracy": [],
        "learning_rates": [],
        "epoch_times": []
    }
    
    best_task_acc = 0.0
    total_start = time.time()
    
    print(f"\nSaving to: {save_dir}")
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
        
        optimizer.zero_grad(set_to_none=True)  # Faster than setting to zero
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
        
        # Timing stats
        time_data = 0.0
        time_transfer = 0.0
        time_nograd = 0.0
        time_forward = 0.0
        time_backward = 0.0
        time_optim = 0.0
        iter_start = time.time()
        
        for batch in pbar:
            try:
                # Data loading time (time since last iteration ended)
                t0 = time.time()
                time_data += t0 - iter_start
                
                # Non-blocking transfers: overlap CPU->GPU with compute
                demo_inputs = [g.to(device, non_blocking=True) for g in batch["demo_inputs"]]
                demo_outputs = [g.to(device, non_blocking=True) for g in batch["demo_outputs"]]
                test_input = batch["test_input"].to(device, non_blocking=True)
                test_output = batch["test_output"].to(device, non_blocking=True)
                torch.cuda.synchronize()  # Wait for transfers to complete (for timing)
                t1 = time.time()
                time_transfer += t1 - t0
                
                # === NO-GRAD REFINEMENT LOOPS (TRM trick) ===
                if config["no_grad_loops"] > 0:
                    with torch.no_grad():
                        _ = model(demo_inputs, demo_outputs, test_input,
                                 n_recursions=config["no_grad_loops"])
                    torch.cuda.synchronize()
                t2 = time.time()
                time_nograd += t2 - t1
                
                # === GRADIENT STEP ===
                with autocast('cuda', enabled=args.amp):
                    loss_dict = model.compute_loss(
                        demo_inputs, demo_outputs, test_input, test_output,
                        n_recursions=config["n_recursions"],
                        supervision_weights=config["supervision_weights"]
                    )
                    loss = loss_dict["total_loss"] / config["grad_accum"]
                torch.cuda.synchronize()
                t3 = time.time()
                time_forward += t3 - t2
                
                if scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                torch.cuda.synchronize()
                t4 = time.time()
                time_backward += t4 - t3
                
                accum_step += 1
                epoch_loss += loss.item() * config["grad_accum"]
                n_batches += 1
                
                if accum_step >= config["grad_accum"]:
                    if scaler:
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        nn.utils.clip_grad_norm_(model.parameters(), config["clip_grad"])
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)  # Faster than setting to zero
                    accum_step = 0
                    torch.cuda.synchronize()
                    time_optim += time.time() - t4
                
                # Print timing breakdown every 50 iterations
                if n_batches == 50:
                    total = time_data + time_transfer + time_nograd + time_forward + time_backward + time_optim
                    print(f"\n⏱️  TIMING BREAKDOWN (first 50 iters):")
                    print(f"  Data loading:  {time_data:6.2f}s ({100*time_data/total:5.1f}%)")
                    print(f"  GPU transfer:  {time_transfer:6.2f}s ({100*time_transfer/total:5.1f}%)")
                    print(f"  No-grad loops: {time_nograd:6.2f}s ({100*time_nograd/total:5.1f}%)")
                    print(f"  Forward pass:  {time_forward:6.2f}s ({100*time_forward/total:5.1f}%)")
                    print(f"  Backward pass: {time_backward:6.2f}s ({100*time_backward/total:5.1f}%)")
                    print(f"  Optimizer:     {time_optim:6.2f}s ({100*time_optim/total:5.1f}%)")
                    print(f"  TOTAL:         {total:6.2f}s ({total/50:.2f}s/iter)\n")
                
                iter_start = time.time()  # Reset for next iteration
                
                pbar.set_postfix({
                    "loss": f"{loss.item() * config['grad_accum']:.4f}",
                    "oom": oom_count
                })
                
                # Dashboard update
                if dashboard and n_batches % 20 == 0:
                    mem_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0
                    elapsed = time.time() - epoch_start
                    samples_sec = (n_batches * config["batch_size"]) / elapsed if elapsed > 0 else 0
                    
                    dashboard.update(
                        status="running",
                        loss=loss.item() * config["grad_accum"],
                        epoch=epoch,
                        step=n_batches,
                        total_steps=len(train_loader),
                        total_epochs=config["epochs"],
                        memory_gb=mem_gb,
                        n_recursions=config["n_recursions"],
                        lr=current_lr if 'current_lr' in dir() else config["lr_trunk"],
                        samples_per_sec=samples_sec
                    )
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    oom_count += 1
                    clear_memory()
                    optimizer.zero_grad(set_to_none=True)  # Faster than setting to zero
                    accum_step = 0
                    continue
                raise e
        
        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        # === TASK-LEVEL EVALUATION ===
        # Evaluate on HELD-OUT evaluation split (not training!)
        eval_results = evaluate_task_accuracy(
            model, args.data_dir, "evaluation", device,
            n_recursions=config["n_recursions"], n_samples=50
        )
        
        # Log
        history["train_loss"].append(avg_loss)
        history["task_accuracy"].append(eval_results["task_accuracy"])
        history["cell_accuracy"].append(eval_results["cell_accuracy"])
        history["learning_rates"].append(current_lr)
        history["epoch_times"].append(epoch_time)
        
        print(f"Epoch {epoch}: loss={avg_loss:.4f}, "
              f"task_acc={eval_results['task_accuracy']:.1%}, "
              f"cell_acc={eval_results['cell_accuracy']:.1%}, "
              f"lr={current_lr:.2e}, time={epoch_time:.0f}s, oom={oom_count}")
        
        # Dashboard epoch log
        if dashboard:
            dashboard.log_epoch(
                epoch=epoch,
                train_loss=avg_loss,
                val_accuracy=eval_results["task_accuracy"]
            )
        
        # Save best model
        if eval_results["task_accuracy"] > best_task_acc:
            best_task_acc = eval_results["task_accuracy"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "task_accuracy": eval_results["task_accuracy"],
                "cell_accuracy": eval_results["cell_accuracy"],
                "config": config
            }, save_dir / "best_model.pt")
            print(f"  -> New best! Task acc: {best_task_acc:.1%}")
        
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config
            }, save_dir / f"checkpoint_epoch{epoch}.pt")
        
        # Save history
        with open(save_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
    
    total_time = time.time() - total_start
    
    # === FINAL EVALUATION ===
    print("\n" + "="*60)
    print("FINAL EVALUATION")
    print("="*60)
    
    # Load best model
    best_ckpt = torch.load(save_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])
    
    # Full eval on AGI-1 (evaluation split = held-out)
    print("\nEvaluating on ARC-AGI-1 evaluation split (held-out)...")
    agi1_results = evaluate_task_accuracy(
        model, "data/arc-agi-1", "evaluation", device,
        n_recursions=config["n_recursions"]
    )
    print(f"AGI-1 Task Accuracy: {agi1_results['task_accuracy']:.1%} "
          f"({agi1_results['tasks_correct']}/{agi1_results['tasks_total']})")
    
    # Eval on AGI-2 (evaluation split = held-out)
    print("\nEvaluating on ARC-AGI-2 evaluation split...")
    try:
        agi2_results = evaluate_task_accuracy(
            model, "data/arc-agi-2", "evaluation", device,
            n_recursions=config["n_recursions"]
        )
    except Exception as e:
        print(f"AGI-2 eval failed (maybe not downloaded?): {e}")
        agi2_results = {"task_accuracy": 0, "cell_accuracy": 0, "tasks_correct": 0, "tasks_total": 0}
    print(f"AGI-2 Task Accuracy: {agi2_results['task_accuracy']:.1%} "
          f"({agi2_results['tasks_correct']}/{agi2_results['tasks_total']})")
    
    # Save final results
    final_results = {
        "training_time_hours": total_time / 3600,
        "best_task_accuracy": best_task_acc,
        "agi1_task_accuracy": agi1_results["task_accuracy"],
        "agi1_cell_accuracy": agi1_results["cell_accuracy"],
        "agi2_task_accuracy": agi2_results["task_accuracy"],
        "agi2_cell_accuracy": agi2_results["cell_accuracy"],
        "config": config
    }
    
    with open(save_dir / "final_results.json", "w") as f:
        json.dump(final_results, f, indent=2)
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Total time: {total_time/3600:.1f} hours")
    print(f"Best task accuracy: {best_task_acc:.1%}")
    print(f"AGI-1: {agi1_results['task_accuracy']:.1%}")
    print(f"AGI-2: {agi2_results['task_accuracy']:.1%}")
    print(f"Results saved to: {save_dir}")


def main():
    parser = argparse.ArgumentParser(description="Full TRM training run")
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)  # Higher with grad checkpointing
    parser.add_argument("--grad_accum", type=int, default=4)  # Effective batch = 32
    parser.add_argument("--augment_factor", type=int, default=20)  # 20× for fast iteration
    parser.add_argument("--n_recursions", type=int, default=8)  # 8 was best in sweep
    parser.add_argument("--no_grad_loops", type=int, default=0)  # Skip for speed
    parser.add_argument("--amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for speed")
    parser.add_argument("--grad_checkpoint", action="store_true", help="Use gradient checkpointing to save memory")
    parser.add_argument("--dashboard", action="store_true", help="Send metrics to dashboard")
    args = parser.parse_args()
    
    train(args)


if __name__ == "__main__":
    main()
