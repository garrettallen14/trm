#!/usr/bin/env python3
"""
Profile ARC data to understand grid sizes and loading throughput.

Usage:
    uv run python experiments/profile_data.py
"""

import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.data import ARCDataset, create_dataloader


def profile_grid_sizes(data_dir: str):
    """Analyze grid size distribution in ARC tasks."""
    print("\n" + "="*60)
    print("GRID SIZE ANALYSIS")
    print("="*60)
    
    dataset = ARCDataset(data_dir, split="training", augment=False)
    
    input_sizes = []
    output_sizes = []
    demo_counts = []
    total_tokens_per_task = []
    
    for i in range(len(dataset)):
        item = dataset[i]
        demo_counts.append(len(item["demo_inputs"]))
        
        task_tokens = 0
        for inp in item["demo_inputs"]:
            h, w = inp.shape
            input_sizes.append((h, w))
            task_tokens += h * w
        
        for out in item["demo_outputs"]:
            h, w = out.shape
            output_sizes.append((h, w))
            task_tokens += h * w
        
        # Test
        h, w = item["test_input"].shape
        input_sizes.append((h, w))
        task_tokens += h * w
        
        h, w = item["test_output"].shape
        output_sizes.append((h, w))
        task_tokens += h * w
        
        total_tokens_per_task.append(task_tokens)
    
    # Analyze
    print(f"\nTotal tasks: {len(dataset)}")
    
    # Demo counts
    print(f"\nDemo pairs per task:")
    demo_counter = Counter(demo_counts)
    for n, count in sorted(demo_counter.items()):
        print(f"  {n} demos: {count} tasks ({100*count/len(dataset):.1f}%)")
    
    # Grid sizes
    heights = [s[0] for s in input_sizes + output_sizes]
    widths = [s[1] for s in input_sizes + output_sizes]
    
    print(f"\nGrid dimensions:")
    print(f"  Height: min={min(heights)}, max={max(heights)}, mean={sum(heights)/len(heights):.1f}")
    print(f"  Width:  min={min(widths)}, max={max(widths)}, mean={sum(widths)/len(widths):.1f}")
    
    # Size buckets
    print(f"\nGrid size distribution:")
    size_buckets = {
        "tiny (≤5×5)": 0,
        "small (6-10)": 0,
        "medium (11-20)": 0,
        "large (21-30)": 0
    }
    for h, w in input_sizes + output_sizes:
        max_dim = max(h, w)
        if max_dim <= 5:
            size_buckets["tiny (≤5×5)"] += 1
        elif max_dim <= 10:
            size_buckets["small (6-10)"] += 1
        elif max_dim <= 20:
            size_buckets["medium (11-20)"] += 1
        else:
            size_buckets["large (21-30)"] += 1
    
    total = len(input_sizes + output_sizes)
    for name, count in size_buckets.items():
        print(f"  {name}: {count} ({100*count/total:.1f}%)")
    
    # Tokens per task (affects attention memory)
    print(f"\nTokens per task (affects memory):")
    print(f"  Min: {min(total_tokens_per_task)}")
    print(f"  Max: {max(total_tokens_per_task)}")
    print(f"  Mean: {sum(total_tokens_per_task)/len(total_tokens_per_task):.0f}")
    print(f"  P90: {sorted(total_tokens_per_task)[int(0.9*len(total_tokens_per_task))]}")
    print(f"  P99: {sorted(total_tokens_per_task)[int(0.99*len(total_tokens_per_task))]}")
    
    # Memory estimation
    d_model = 512
    n_heads = 4
    n_recursions = 8
    
    print(f"\nMemory estimation (d_model={d_model}, heads={n_heads}, recursions={n_recursions}):")
    for percentile, name in [(0.5, "median"), (0.9, "P90"), (0.99, "P99"), (1.0, "max")]:
        idx = min(int(percentile * len(total_tokens_per_task)), len(total_tokens_per_task) - 1)
        tokens = sorted(total_tokens_per_task)[idx]
        # Rough memory: attention is O(n²), embeddings are O(n*d)
        attn_mem = tokens * tokens * n_heads * 4 / 1e9  # float32
        embed_mem = tokens * d_model * 4 / 1e9
        print(f"  {name} task ({tokens} tokens): ~{attn_mem + embed_mem:.2f} GB per sample")


def profile_dataloader(data_dir: str, device: torch.device):
    """Profile data loading throughput."""
    print("\n" + "="*60)
    print("DATALOADER THROUGHPUT")
    print("="*60)
    
    configs = [
        {"batch_size": 1, "num_workers": 0, "augment": False},
        {"batch_size": 4, "num_workers": 0, "augment": False},
        {"batch_size": 4, "num_workers": 0, "augment": True, "augment_factor": 10},
        {"batch_size": 4, "num_workers": 2, "augment": True, "augment_factor": 10},
        {"batch_size": 4, "num_workers": 4, "augment": True, "augment_factor": 10},
    ]
    
    for cfg in configs:
        loader = create_dataloader(
            data_dir=data_dir,
            split="training",
            **cfg
        )
        
        # Warmup
        batch = next(iter(loader))
        
        # Benchmark loading
        start = time.perf_counter()
        n_batches = 0
        n_samples = 0
        
        for batch in loader:
            n_batches += 1
            n_samples += batch["test_input"].shape[0]
            if n_batches >= 100:
                break
        
        elapsed = time.perf_counter() - start
        
        aug_str = f"aug={cfg.get('augment_factor', 'off')}" if cfg.get('augment') else "no_aug"
        print(f"  batch={cfg['batch_size']}, workers={cfg['num_workers']}, {aug_str}: "
              f"{n_samples/elapsed:.1f} samples/s, {1000*elapsed/n_batches:.1f} ms/batch")


def profile_transfer(data_dir: str, device: torch.device):
    """Profile CPU->GPU transfer time."""
    print("\n" + "="*60)
    print("CPU->GPU TRANSFER")
    print("="*60)
    
    loader = create_dataloader(
        data_dir=data_dir,
        split="training",
        batch_size=4,
        augment=False,
        num_workers=0
    )
    
    # Get some batches
    batches = [next(iter(loader)) for _ in range(10)]
    
    # Benchmark transfer
    start = time.perf_counter()
    for batch in batches:
        demo_inputs = [g.to(device) for g in batch["demo_inputs"]]
        demo_outputs = [g.to(device) for g in batch["demo_outputs"]]
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    print(f"  Transfer time: {1000*elapsed/len(batches):.2f} ms/batch")
    
    # Check if pin_memory helps
    if device.type == "cuda":
        loader_pinned = create_dataloader(
            data_dir=data_dir,
            split="training",
            batch_size=4,
            augment=False,
            num_workers=0
        )
        # pin_memory is already True by default in create_dataloader
        print(f"  (pin_memory=True by default)")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
    
    profile_grid_sizes(args.data_dir)
    profile_dataloader(args.data_dir, device)
    
    if device.type == "cuda":
        profile_transfer(args.data_dir, device)
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print("  - Use batch_size=4 with grad_accum=8 for effective batch 32")
    print("  - Use num_workers=2-4 for faster data loading")
    print("  - Large grids (P99) may still OOM - skip gracefully")
    print("="*60)


if __name__ == "__main__":
    main()
