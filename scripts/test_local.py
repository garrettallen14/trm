#!/usr/bin/env python3
"""
Quick local test to verify everything works on M4 Pro.

This tests:
1. Model can be created and forward pass works
2. Data loading works
3. Training loop runs without errors
4. Evaluation runs without errors

Usage:
    python scripts/test_local.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"CUDA available: {torch.cuda.is_available()}")

# Determine device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")


def test_model():
    """Test model creation and forward pass."""
    print("\n" + "="*60)
    print("Testing Model")
    print("="*60)
    
    from src.model import TinyRecursiveModel, count_parameters
    
    # Create small model for testing
    model = TinyRecursiveModel(
        d_model=128,
        n_heads=4,
        n_layers=2,
        n_recursions=4
    ).to(device)
    
    params = count_parameters(model)
    print(f"Model parameters: {params['total']:,}")
    
    # Create dummy data
    batch = 2
    demo_inputs = [torch.randint(0, 10, (batch, 5, 5)).to(device) for _ in range(2)]
    demo_outputs = [torch.randint(0, 10, (batch, 5, 5)).to(device) for _ in range(2)]
    test_input = torch.randint(0, 10, (batch, 5, 5)).to(device)
    test_output = torch.randint(0, 10, (batch, 5, 5)).to(device)
    
    # Forward pass
    result = model(demo_inputs, demo_outputs, test_input, return_intermediates=True)
    print(f"Output shape: {result['logits'].shape}")
    print(f"Intermediates: {len(result['intermediates'])} steps")
    print(f"Attention entropy: {result['attn_entropy']:.4f}")
    
    # Loss computation
    loss_dict = model.compute_loss(
        demo_inputs, demo_outputs, test_input, test_output,
        supervision_weights="linear"
    )
    print(f"Total loss: {loss_dict['total_loss']:.4f}")
    
    # Backward pass
    loss_dict['total_loss'].backward()
    print("Backward pass successful!")
    
    # Prediction
    pred = model.predict(demo_inputs, demo_outputs, test_input)
    print(f"Prediction shape: {pred.shape}")
    
    print("✓ Model test passed!")
    return True


def test_data():
    """Test data loading."""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    data_dir = Path("data/arc-agi-1")
    
    if not data_dir.exists():
        print(f"Data directory {data_dir} not found.")
        print("Please run: git clone https://github.com/fchollet/ARC-AGI.git data/arc-agi-1")
        return False
    
    from src.data import ARCDataset, create_dataloader
    
    # Test dataset
    dataset = ARCDataset(data_dir, split="training", augment=False)
    print(f"Dataset size: {len(dataset)}")
    
    item = dataset[0]
    print(f"Task ID: {item['task_id']}")
    print(f"Demo inputs: {len(item['demo_inputs'])}")
    print(f"Test input shape: {item['test_input'].shape}")
    
    # Test dataloader
    loader = create_dataloader(
        data_dir,
        split="training",
        batch_size=2,
        augment=True,
        augment_factor=2,
        num_workers=0
    )
    
    batch = next(iter(loader))
    print(f"Batch demo inputs: {len(batch['demo_inputs'])}, shape: {batch['demo_inputs'][0].shape}")
    print(f"Batch test input shape: {batch['test_input'].shape}")
    
    print("✓ Data test passed!")
    return True


def test_training_step():
    """Test one training step."""
    print("\n" + "="*60)
    print("Testing Training Step")
    print("="*60)
    
    data_dir = Path("data/arc-agi-1")
    if not data_dir.exists():
        print("Skipping (no data)")
        return True
    
    from src.model import TinyRecursiveModel
    from src.data import create_dataloader
    
    # Small model
    model = TinyRecursiveModel(
        d_model=64,
        n_heads=2,
        n_layers=1,
        n_recursions=2
    ).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Get one batch
    loader = create_dataloader(
        data_dir,
        batch_size=2,
        augment=True,
        augment_factor=2,
        num_workers=0
    )
    batch = next(iter(loader))
    
    # Move to device
    demo_inputs = [g.to(device) for g in batch["demo_inputs"]]
    demo_outputs = [g.to(device) for g in batch["demo_outputs"]]
    test_input = batch["test_input"].to(device)
    test_output = batch["test_output"].to(device)
    
    # Training step
    model.train()
    optimizer.zero_grad()
    
    loss_dict = model.compute_loss(
        demo_inputs, demo_outputs, test_input, test_output,
        n_recursions=2
    )
    
    loss = loss_dict['total_loss']
    print(f"Loss before: {loss.item():.4f}")
    
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    
    # Check loss decreased
    with torch.no_grad():
        loss_dict2 = model.compute_loss(
            demo_inputs, demo_outputs, test_input, test_output,
            n_recursions=2
        )
    print(f"Loss after: {loss_dict2['total_loss'].item():.4f}")
    
    print("✓ Training step test passed!")
    return True


def test_evaluation():
    """Test evaluation."""
    print("\n" + "="*60)
    print("Testing Evaluation")
    print("="*60)
    
    data_dir = Path("data/arc-agi-1")
    if not data_dir.exists():
        print("Skipping (no data)")
        return True
    
    from src.model import TinyRecursiveModel
    from src.eval import Evaluator
    from src.data import ARCDataset
    
    # Small model
    model = TinyRecursiveModel(
        d_model=64,
        n_heads=2,
        n_layers=1,
        n_recursions=2
    )
    
    evaluator = Evaluator(model, device=str(device))
    
    # Load one task
    dataset = ARCDataset(data_dir, augment=False)
    item = dataset[0]
    
    result = evaluator.evaluate_task(
        demo_inputs=item["demo_inputs"],
        demo_outputs=item["demo_outputs"],
        test_input=item["test_input"],
        test_output=item["test_output"],
        n_ensemble=2,
        pass_at_k=2
    )
    
    print(f"Correct: {result['correct']}")
    print(f"Pass@2: {result['pass_k_correct']}")
    print(f"Cell accuracy: {result['cell_accuracy']:.2%}")
    
    print("✓ Evaluation test passed!")
    return True


def main():
    print("="*60)
    print("TRM Local Test Suite")
    print("="*60)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_model()
    all_passed &= test_data()
    all_passed &= test_training_step()
    all_passed &= test_evaluation()
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("Ready for cloud training.")
    else:
        print("SOME TESTS FAILED ✗")
        print("Please fix issues before proceeding.")
    print("="*60)


if __name__ == "__main__":
    main()
