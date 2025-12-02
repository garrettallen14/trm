#!/usr/bin/env python3
"""
Proper task-level evaluation for ARC.

A task is correct only if ALL cells match the expected output.
This is much harder than cell-level accuracy.

Usage:
    uv run python experiments/eval_tasks.py --checkpoint experiments/runs/<run_id>/best_model.pt
    uv run python experiments/eval_tasks.py --checkpoint experiments/runs/<run_id>/best_model.pt --dataset arc-agi-2
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from tqdm import tqdm

from src.model import TinyRecursiveModel
from src.data import ARCDataset, Tokenizer


def evaluate_task_accuracy(model, dataset, device, n_recursions=8):
    """Evaluate task-level accuracy (all cells must match)."""
    model.eval()
    
    correct_tasks = 0
    total_tasks = 0
    
    cell_correct = 0
    cell_total = 0
    
    results = []
    
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Evaluating"):
            item = dataset[i]
            
            # Prepare inputs
            demo_inputs = [g.unsqueeze(0).to(device) for g in item["demo_inputs"]]
            demo_outputs = [g.unsqueeze(0).to(device) for g in item["demo_outputs"]]
            test_input = item["test_input"].unsqueeze(0).to(device)
            test_output = item["test_output"].to(device)
            
            try:
                # Get prediction
                pred = model.predict(demo_inputs, demo_outputs, test_input, n_recursions=n_recursions)
                pred = pred.squeeze(0)  # Remove batch dim
                
                # Reshape pred to match test_output if needed
                if pred.dim() == 1:
                    h, w = test_output.shape
                    pred = pred.view(h, w)
                
                # Check if all cells match
                mask = test_output != 10  # Ignore padding
                cells_match = (pred == test_output) & mask
                
                task_correct = cells_match.all().item()
                correct_tasks += task_correct
                total_tasks += 1
                
                # Also track cell-level
                cell_correct += cells_match.sum().item()
                cell_total += mask.sum().item()
                
                results.append({
                    "task_id": item.get("task_id", f"task_{i}"),
                    "correct": task_correct,
                    "cell_accuracy": cells_match.sum().item() / mask.sum().item(),
                    "grid_size": f"{test_output.shape[0]}x{test_output.shape[1]}"
                })
                
            except Exception as e:
                print(f"Error on task {i}: {e}")
                total_tasks += 1
                results.append({
                    "task_id": item.get("task_id", f"task_{i}"),
                    "correct": False,
                    "cell_accuracy": 0,
                    "error": str(e)
                })
    
    task_accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
    cell_accuracy = cell_correct / cell_total if cell_total > 0 else 0
    
    return {
        "task_accuracy": task_accuracy,
        "tasks_correct": correct_tasks,
        "tasks_total": total_tasks,
        "cell_accuracy": cell_accuracy,
        "results": results
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate task-level accuracy")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="arc-agi-1", choices=["arc-agi-1", "arc-agi-2"])
    parser.add_argument("--split", type=str, default="training")
    parser.add_argument("--n_recursions", type=int, default=8)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint["config"]
    
    # Create model
    model = TinyRecursiveModel(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_recursions=config["n_recursions"]
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Load dataset
    data_dir = f"data/{args.dataset}"
    print(f"Loading dataset: {data_dir}/{args.split}")
    dataset = ARCDataset(data_dir, split=args.split, augment=False)
    print(f"Tasks: {len(dataset)}")
    
    # Evaluate
    results = evaluate_task_accuracy(model, dataset, device, n_recursions=args.n_recursions)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {args.dataset} ({args.split})")
    print(f"Tasks: {results['tasks_total']}")
    print(f"")
    print(f"Task Accuracy:  {results['task_accuracy']:.2%} ({results['tasks_correct']}/{results['tasks_total']})")
    print(f"Cell Accuracy:  {results['cell_accuracy']:.2%}")
    print("="*60)
    
    # Show some examples
    correct = [r for r in results["results"] if r["correct"]]
    wrong = [r for r in results["results"] if not r["correct"]]
    
    if correct:
        print(f"\nCorrect tasks ({len(correct)}):")
        for r in correct[:5]:
            print(f"  {r['task_id']}: {r['grid_size']}")
    
    if wrong:
        print(f"\nWrong tasks (showing 5 of {len(wrong)}):")
        for r in wrong[:5]:
            print(f"  {r['task_id']}: cell_acc={r['cell_accuracy']:.1%}")


if __name__ == "__main__":
    main()
