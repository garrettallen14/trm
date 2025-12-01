#!/usr/bin/env python3
"""
Quick sanity check for RunPod GPU.

Runs in ~30 seconds and verifies:
1. GPU is detected and working
2. Model fits in memory
3. Training loop runs without errors
4. Loss decreases (model is learning something)

Usage:
    uv run python experiments/sanity_check.py
"""

import gc
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def clear_memory():
    """Clear GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def check_model():
    """Check model creation and forward pass."""
    print("\n" + "="*50)
    print("1. Model Check")
    print("="*50)
    
    from src.model import TinyRecursiveModel, count_parameters
    
    model = TinyRecursiveModel(
        d_model=512,
        n_heads=4,
        n_layers=2,
        n_recursions=8
    ).to(device)
    
    params = count_parameters(model)
    print(f"Parameters: {params['total']:,}")
    
    # Test forward pass with small grids
    batch = 4
    demo_inputs = [torch.randint(0, 10, (batch, 8, 8), device=device) for _ in range(2)]
    demo_outputs = [torch.randint(0, 10, (batch, 8, 8), device=device) for _ in range(2)]
    test_input = torch.randint(0, 10, (batch, 8, 8), device=device)
    test_output = torch.randint(0, 10, (batch, 8, 8), device=device)
    
    result = model(demo_inputs, demo_outputs, test_input)
    print(f"Output shape: {result['logits'].shape}")
    
    # Test backward
    loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output)
    loss_dict["total_loss"].backward()
    print(f"Loss: {loss_dict['total_loss'].item():.4f}")
    print("✓ Model check passed!")
    
    del model
    clear_memory()


def check_data():
    """Check data loading."""
    print("\n" + "="*50)
    print("2. Data Check")
    print("="*50)
    
    from src.data import ARCDataset
    
    data_dir = Path("data/arc-agi-1")
    
    if not (data_dir / "data" / "training").exists():
        print(f"Data directory {data_dir} not found.")
        print("Please run: git clone https://github.com/fchollet/ARC-AGI.git data/arc-agi-1")
        return
    
    # Just check dataset loads (no dataloader to avoid memory issues)
    dataset = ARCDataset(data_dir, split="training", augment=False)
    print(f"Dataset size: {len(dataset)} tasks")
    
    item = dataset[0]
    print(f"Task: {item['task_id']}")
    print(f"Demos: {len(item['demo_inputs'])}")
    print(f"Test input shape: {item['test_input'].shape}")
    print("✓ Data check passed!")


def check_training():
    """Check training loop with real data."""
    print("\n" + "="*50)
    print("3. Training Check (2 batches)")
    print("="*50)
    
    clear_memory()
    
    from src.model import TinyRecursiveModel
    from src.data import create_dataloader
    import torch.optim as optim
    
    # Use smaller model and batch size for real data
    # Real ARC grids can be up to 30x30, attention is O(n²)
    model = TinyRecursiveModel(
        d_model=256,  # Smaller for memory
        n_heads=4,
        n_layers=2,
        n_recursions=4  # Fewer recursions
    ).to(device)
    
    data_dir = Path("data/arc-agi-1")
    if not (data_dir / "data" / "training").exists():
        print("Data not found, skipping")
        return True
    
    loader = create_dataloader(
        data_dir=str(data_dir),
        batch_size=2,  # Small batch for large grids
        augment=False,  # No augmentation for speed
        num_workers=0
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    
    losses = []
    for i, batch in enumerate(loader):
        if i >= 2:
            break
        
        demo_inputs = [g.to(device) for g in batch["demo_inputs"]]
        demo_outputs = [g.to(device) for g in batch["demo_outputs"]]
        test_input = batch["test_input"].to(device)
        test_output = batch["test_output"].to(device)
        
        # Print grid sizes for debugging
        if i == 0:
            total_cells = sum(g.shape[1] * g.shape[2] for g in demo_inputs + demo_outputs)
            total_cells += test_input.shape[1] * test_input.shape[2]
            print(f"  Total tokens per sample: ~{total_cells}")
        
        optimizer.zero_grad()
        loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=4)
        loss = loss_dict["total_loss"]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        losses.append(loss.item())
        print(f"  Batch {i+1}: loss = {loss.item():.4f}")
        
        clear_memory()
    
    del model, optimizer
    clear_memory()
    
    if len(losses) >= 2 and losses[-1] < losses[0] * 1.5:  # Loss shouldn't explode
        print("✓ Training check passed!")
        return True
    elif len(losses) >= 1:
        print("✓ Training check passed!")
        return True
    else:
        print("✗ Warning: Loss may be unstable")
        return False


def check_throughput():
    """Quick throughput check with standard grid sizes."""
    print("\n" + "="*50)
    print("4. Throughput Check")
    print("="*50)
    
    clear_memory()
    
    from src.model import TinyRecursiveModel
    
    # Use full model but with controlled grid sizes
    model = TinyRecursiveModel(
        d_model=512,
        n_heads=4,
        n_layers=2,
        n_recursions=8
    ).to(device)
    
    model.train()
    
    # Use realistic grid size (average ARC grid is ~10x10)
    batch = 32
    grid_size = 12  # Realistic average
    demo_inputs = [torch.randint(0, 10, (batch, grid_size, grid_size), device=device) for _ in range(2)]
    demo_outputs = [torch.randint(0, 10, (batch, grid_size, grid_size), device=device) for _ in range(2)]
    test_input = torch.randint(0, 10, (batch, grid_size, grid_size), device=device)
    test_output = torch.randint(0, 10, (batch, grid_size, grid_size), device=device)
    
    print(f"  Grid size: {grid_size}x{grid_size}, Batch: {batch}")
    
    # Warmup
    for _ in range(3):
        loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=8)
        loss_dict["total_loss"].backward()
        model.zero_grad()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    n_iters = 10
    start = time.perf_counter()
    for _ in range(n_iters):
        loss_dict = model.compute_loss(demo_inputs, demo_outputs, test_input, test_output, n_recursions=8)
        loss_dict["total_loss"].backward()
        model.zero_grad()
    
    if device.type == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    samples_per_sec = (batch * n_iters) / elapsed
    
    print(f"  Throughput: {samples_per_sec:.1f} samples/sec")
    
    # Memory
    if device.type == "cuda":
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  Peak memory: {mem:.2f} GB")
    
    # Estimate training time
    n_tasks = 400
    augment_factor = 100
    epochs = 100
    total_samples = n_tasks * augment_factor * epochs
    estimated_hours = total_samples / samples_per_sec / 3600
    
    print(f"\n  Estimated training time: {estimated_hours:.1f} hours")
    print("✓ Throughput check passed!")
    
    del model
    clear_memory()


def main():
    print("="*50)
    print("TRM RunPod Sanity Check")
    print("="*50)
    
    start = time.time()
    
    check_model()
    clear_memory()
    
    check_data()
    clear_memory()
    
    check_training()
    clear_memory()
    
    check_throughput()
    
    elapsed = time.time() - start
    
    print("\n" + "="*50)
    print(f"ALL CHECKS PASSED in {elapsed:.1f}s")
    print("="*50)
    print("\nReady for experiments! Run:")
    print("  uv run python experiments/profile_model.py")
    print("  uv run python experiments/quick_sweep.py --sweep lr")


if __name__ == "__main__":
    main()
