#!/usr/bin/env python3
"""
Profile model to find optimal batch size and identify bottlenecks.

This script:
1. Finds maximum batch size that fits in memory
2. Measures throughput at different batch sizes
3. Profiles forward/backward pass to identify bottlenecks
4. Tests different recursion depths

Usage:
    uv run python experiments/profile_model.py
    uv run python experiments/profile_model.py --d_model 256 --n_recursions 8
"""

import argparse
import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from tqdm import tqdm

from src.model import TinyRecursiveModel, count_parameters


def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_memory_stats(device):
    """Get current memory usage."""
    if device.type == "cuda":
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9,
        }
    elif device.type == "mps":
        return {
            "allocated_gb": torch.mps.current_allocated_memory() / 1e9,
        }
    return {"allocated_gb": 0}


def clear_memory(device):
    """Clear GPU memory."""
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    elif device.type == "mps":
        torch.mps.empty_cache()


def create_dummy_batch(batch_size: int, n_demos: int = 2, grid_size: int = 10, device: torch.device = None):
    """Create dummy batch for profiling."""
    demo_inputs = [torch.randint(0, 10, (batch_size, grid_size, grid_size), device=device) for _ in range(n_demos)]
    demo_outputs = [torch.randint(0, 10, (batch_size, grid_size, grid_size), device=device) for _ in range(n_demos)]
    test_input = torch.randint(0, 10, (batch_size, grid_size, grid_size), device=device)
    test_output = torch.randint(0, 10, (batch_size, grid_size, grid_size), device=device)
    return demo_inputs, demo_outputs, test_input, test_output


def find_max_batch_size(model: nn.Module, device: torch.device, n_recursions: int = 8, grid_size: int = 15):
    """Binary search for maximum batch size that fits in memory."""
    print("\n" + "="*60)
    print("Finding Maximum Batch Size")
    print("="*60)
    
    model.to(device)
    model.train()
    
    low, high = 1, 512
    max_working = 1
    
    while low <= high:
        mid = (low + high) // 2
        clear_memory(device)
        
        try:
            demo_inputs, demo_outputs, test_input, test_output = create_dummy_batch(mid, grid_size=grid_size, device=device)
            
            # Forward + backward
            loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=n_recursions)
            loss_dict["total_loss"].backward()
            
            # Success
            max_working = mid
            mem = get_memory_stats(device)
            print(f"  Batch {mid}: ✓ (mem: {mem.get('allocated_gb', 0):.2f} GB)")
            low = mid + 1
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "mps" in str(e).lower():
                print(f"  Batch {mid}: ✗ OOM")
                high = mid - 1
            else:
                raise e
        
        finally:
            clear_memory(device)
    
    print(f"\nMax batch size: {max_working}")
    return max_working


def benchmark_throughput(model: nn.Module, device: torch.device, batch_sizes: list[int], n_recursions: int = 8, n_iters: int = 10):
    """Measure throughput at different batch sizes."""
    print("\n" + "="*60)
    print("Benchmarking Throughput")
    print("="*60)
    
    model.to(device)
    results = []
    
    for batch_size in batch_sizes:
        clear_memory(device)
        
        try:
            demo_inputs, demo_outputs, test_input, test_output = create_dummy_batch(batch_size, device=device)
            
            # Warmup
            model.train()
            for _ in range(3):
                loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=n_recursions)
                loss_dict["total_loss"].backward()
                model.zero_grad()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            # Benchmark
            start = time.perf_counter()
            for _ in range(n_iters):
                loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=n_recursions)
                loss_dict["total_loss"].backward()
                model.zero_grad()
            
            if device.type == "cuda":
                torch.cuda.synchronize()
            
            elapsed = time.perf_counter() - start
            samples_per_sec = (batch_size * n_iters) / elapsed
            ms_per_batch = (elapsed / n_iters) * 1000
            
            mem = get_memory_stats(device)
            
            results.append({
                "batch_size": batch_size,
                "samples_per_sec": samples_per_sec,
                "ms_per_batch": ms_per_batch,
                "memory_gb": mem.get("allocated_gb", 0)
            })
            
            print(f"  Batch {batch_size:3d}: {samples_per_sec:6.1f} samples/s, {ms_per_batch:6.1f} ms/batch, {mem.get('allocated_gb', 0):.2f} GB")
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  Batch {batch_size:3d}: OOM")
            else:
                raise e
        
        clear_memory(device)
    
    return results


def profile_recursion_depths(model: nn.Module, device: torch.device, batch_size: int, depths: list[int]):
    """Profile different recursion depths."""
    print("\n" + "="*60)
    print("Profiling Recursion Depths")
    print("="*60)
    
    model.to(device)
    model.train()
    results = []
    
    demo_inputs, demo_outputs, test_input, test_output = create_dummy_batch(batch_size, device=device)
    
    for depth in depths:
        clear_memory(device)
        
        # Warmup
        for _ in range(2):
            loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=depth)
            loss_dict["total_loss"].backward()
            model.zero_grad()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        # Benchmark
        n_iters = 10
        start = time.perf_counter()
        for _ in range(n_iters):
            loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=depth)
            loss_dict["total_loss"].backward()
            model.zero_grad()
        
        if device.type == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        ms_per_iter = (elapsed / n_iters) * 1000
        mem = get_memory_stats(device)
        
        results.append({
            "depth": depth,
            "ms_per_iter": ms_per_iter,
            "memory_gb": mem.get("allocated_gb", 0)
        })
        
        print(f"  Depth {depth:2d}: {ms_per_iter:6.1f} ms/iter, {mem.get('allocated_gb', 0):.2f} GB")
    
    return results


def profile_grid_sizes(model: nn.Module, device: torch.device, n_recursions: int = 8):
    """Profile max batch size at different grid sizes."""
    print("\n" + "="*60)
    print("Max Batch Size by Grid Size")
    print("="*60)
    print("  (Grid sizes based on ARC data distribution)")
    
    model.to(device)
    model.train()
    
    # Test realistic grid sizes from ARC distribution
    # Tiny: 3-5, Small: 6-10, Medium: 11-15, Large: 16-25, XL: 26-30
    grid_sizes = [5, 10, 15, 20, 25, 30]
    results = {}
    
    for grid_size in grid_sizes:
        clear_memory(device)
        
        # Binary search for max batch
        low, high = 1, 256
        max_working = 1
        
        while low <= high:
            mid = (low + high) // 2
            clear_memory(device)
            
            try:
                demo_inputs, demo_outputs, test_input, test_output = create_dummy_batch(
                    mid, n_demos=3, grid_size=grid_size, device=device
                )
                
                loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=n_recursions)
                loss_dict["total_loss"].backward()
                model.zero_grad()
                
                max_working = mid
                low = mid + 1
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    high = mid - 1
                else:
                    raise e
            
            clear_memory(device)
        
        results[grid_size] = max_working
        print(f"  Grid {grid_size:2d}x{grid_size:2d}: max batch = {max_working:3d}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Profile TRM model")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=4)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_recursions", type=int, default=8)
    parser.add_argument("--quick", action="store_true", help="Quick mode - skip some tests")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Device: {device}")
    
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"Total memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Create model
    model = TinyRecursiveModel(
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        n_recursions=args.n_recursions
    )
    
    params = count_parameters(model)
    print(f"\nModel: d={args.d_model}, h={args.n_heads}, L={args.n_layers}, K={args.n_recursions}")
    print(f"Parameters: {params['total']:,}")
    
    # Profile by grid size (most useful info)
    grid_batch_limits = profile_grid_sizes(model, device, n_recursions=args.n_recursions)
    
    if not args.quick:
        # Find max batch size with typical grid (12x12 is average)
        max_batch = find_max_batch_size(model, device, n_recursions=args.n_recursions, grid_size=12)
        
        # Benchmark throughput
        batch_sizes = [b for b in [4, 8, 16, 32, 64] if b <= max_batch]
        if batch_sizes:
            benchmark_throughput(model, device, batch_sizes, n_recursions=args.n_recursions)
        
        # Profile recursion depths
        optimal_batch = min(16, max_batch)
        profile_recursion_depths(model, device, optimal_batch, depths=[2, 4, 8, 12, 16])
    
    print("\n" + "="*60)
    print("RECOMMENDATIONS")
    print("="*60)
    print(f"  For small grids (≤10): batch_size=32-64 works")
    print(f"  For medium grids (11-20): batch_size=8-16 works")
    print(f"  For large grids (21-30): batch_size=2-4 works")
    print(f"")
    print(f"  Safe default: batch_size=4 with grad_accum=8")
    print(f"  This handles all grid sizes with OOM fallback")
    print("="*60)


if __name__ == "__main__":
    main()
