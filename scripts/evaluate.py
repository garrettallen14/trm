#!/usr/bin/env python3
"""
Evaluation script for TRM on ARC-AGI.

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data_dir data/arc-agi-1
    python scripts/evaluate.py --checkpoint checkpoints/best.pt --data_dir data/arc-agi-2 --n_ensemble 100
"""

import argparse
import sys
from pathlib import Path

import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import TinyRecursiveModel
from src.eval import Evaluator


def main():
    parser = argparse.ArgumentParser(description="Evaluate TRM on ARC-AGI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1", help="Path to ARC data")
    parser.add_argument("--split", type=str, default="evaluation", help="Dataset split")
    parser.add_argument("--n_ensemble", type=int, default=100, help="Ensemble size")
    parser.add_argument("--pass_at_k", type=int, default=2, help="Pass@k attempts")
    parser.add_argument("--n_recursions", type=int, default=None, help="Override recursion depth")
    parser.add_argument("--output", type=str, default=None, help="Output JSON path")
    parser.add_argument("--max_tasks", type=int, default=None, help="Limit tasks (for debugging)")
    parser.add_argument("--device", type=str, default="auto", help="Device (cuda/mps/cpu/auto)")
    
    args = parser.parse_args()
    
    # Determine device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    
    # Create model
    model = TinyRecursiveModel(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_recursions=config["n_recursions"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    print(f"Model loaded (epoch {checkpoint['epoch']})")
    
    # Evaluate
    evaluator = Evaluator(model, device=device)
    
    print(f"\nEvaluating on {args.data_dir} ({args.split})")
    print(f"Ensemble: {args.n_ensemble}, Pass@{args.pass_at_k}")
    
    results = evaluator.evaluate_dataset(
        data_dir=args.data_dir,
        split=args.split,
        n_ensemble=args.n_ensemble,
        pass_at_k=args.pass_at_k,
        n_recursions=args.n_recursions,
        max_tasks=args.max_tasks
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Tasks:            {results['n_tasks']}")
    print(f"Correct:          {results['n_correct']}")
    print(f"Accuracy:         {results['accuracy']:.2%}")
    print(f"Pass@{args.pass_at_k}:          {results['pass_k_accuracy']:.2%}")
    print(f"Avg Cell Acc:     {results['avg_cell_accuracy']:.2%}")
    print(f"{'='*60}")
    
    # Save results
    if args.output:
        output_path = args.output
    else:
        # Auto-generate output path
        data_name = Path(args.data_dir).name
        output_path = f"results_{data_name}_{args.split}_e{args.n_ensemble}.json"
    
    evaluator.save_results(results, output_path)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
