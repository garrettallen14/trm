"""
Evaluation for Tiny Recursive Model on ARC-AGI.

Features:
- Exact grid match evaluation
- Pass@k scoring (multiple attempts)
- Ensemble over augmentations
- Per-category breakdown
- Visualization of predictions
"""

import json
from collections import defaultdict
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

from .model import TinyRecursiveModel
from .data import ARCDataset, DihedralAugmentation, ColorPermutation, Tokenizer


class Evaluator:
    """Evaluate TRM on ARC-AGI benchmarks."""
    
    def __init__(
        self,
        model: TinyRecursiveModel,
        device: str = "cuda"
    ):
        self.model = model
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self.tokenizer = Tokenizer()
    
    @torch.no_grad()
    def evaluate_task(
        self,
        demo_inputs: list[torch.Tensor],
        demo_outputs: list[torch.Tensor],
        test_input: torch.Tensor,
        test_output: torch.Tensor,
        n_ensemble: int = 1,
        pass_at_k: int = 2,
        n_recursions: int | None = None
    ) -> dict:
        """
        Evaluate a single task.
        
        Args:
            demo_inputs: Demonstration input grids
            demo_outputs: Demonstration output grids
            test_input: Test input grid
            test_output: Ground truth test output
            n_ensemble: Number of augmentation ensembles
            pass_at_k: Number of attempts (pass if any correct)
            n_recursions: Override number of recursion steps
        
        Returns:
            Dictionary with evaluation results
        """
        original_h, original_w = test_output.shape
        
        # Collect predictions from multiple augmentations
        all_predictions = []
        
        for aug_idx in range(n_ensemble):
            # Generate augmentation
            if aug_idx == 0:
                # First attempt: no augmentation
                dihedral = 0
                color_perm = {i: i for i in range(10)}
            else:
                dihedral = DihedralAugmentation.random_transform()
                color_perm = ColorPermutation.generate_permutation(preserve_zero=True)
            
            # Inverse color permutation for de-augmenting predictions
            inv_color_perm = {v: k for k, v in color_perm.items()}
            
            # Apply augmentation
            aug_demo_in = [DihedralAugmentation.apply(ColorPermutation.apply(g, color_perm), dihedral) for g in demo_inputs]
            aug_demo_out = [DihedralAugmentation.apply(ColorPermutation.apply(g, color_perm), dihedral) for g in demo_outputs]
            aug_test_in = DihedralAugmentation.apply(ColorPermutation.apply(test_input, color_perm), dihedral)
            
            # Move to device and add batch dimension
            aug_demo_in = [g.unsqueeze(0).to(self.device) for g in aug_demo_in]
            aug_demo_out = [g.unsqueeze(0).to(self.device) for g in aug_demo_out]
            aug_test_in = aug_test_in.unsqueeze(0).to(self.device)
            
            # Get prediction
            pred = self.model.predict(
                aug_demo_in, aug_demo_out, aug_test_in,
                n_recursions=n_recursions
            )[0].cpu()  # [h, w]
            
            # De-augment prediction
            # Inverse dihedral (dihedral transforms are their own inverses for some, need to handle properly)
            inv_dihedral = [0, 3, 2, 1, 4, 5, 6, 7][dihedral]  # Inverse mapping
            pred = DihedralAugmentation.apply(pred, inv_dihedral)
            pred = ColorPermutation.apply(pred, inv_color_perm)
            
            # Crop to original size
            pred = pred[:original_h, :original_w]
            
            all_predictions.append(pred)
        
        # Majority voting for ensemble
        if n_ensemble > 1:
            stacked = torch.stack(all_predictions)  # [n_ensemble, h, w]
            # Mode voting
            final_pred, _ = torch.mode(stacked, dim=0)
        else:
            final_pred = all_predictions[0]
        
        # Check correctness
        correct = torch.equal(final_pred, test_output)
        
        # Cell-level accuracy
        mask = test_output != self.tokenizer.PAD_TOKEN
        cell_correct = ((final_pred == test_output) & mask).sum().item()
        cell_total = mask.sum().item()
        cell_accuracy = cell_correct / cell_total if cell_total > 0 else 0
        
        # Pass@k: try multiple times
        pass_k_correct = correct
        if not correct and pass_at_k > 1:
            for attempt in range(1, pass_at_k):
                # New prediction with randomness
                dihedral = DihedralAugmentation.random_transform()
                color_perm = ColorPermutation.generate_permutation()
                inv_color_perm = {v: k for k, v in color_perm.items()}
                
                aug_demo_in = [DihedralAugmentation.apply(ColorPermutation.apply(g, color_perm), dihedral) for g in demo_inputs]
                aug_demo_out = [DihedralAugmentation.apply(ColorPermutation.apply(g, color_perm), dihedral) for g in demo_outputs]
                aug_test_in = DihedralAugmentation.apply(ColorPermutation.apply(test_input, color_perm), dihedral)
                
                aug_demo_in = [g.unsqueeze(0).to(self.device) for g in aug_demo_in]
                aug_demo_out = [g.unsqueeze(0).to(self.device) for g in aug_demo_out]
                aug_test_in = aug_test_in.unsqueeze(0).to(self.device)
                
                pred = self.model.predict(aug_demo_in, aug_demo_out, aug_test_in, n_recursions=n_recursions)[0].cpu()
                
                inv_dihedral = [0, 3, 2, 1, 4, 5, 6, 7][dihedral]
                pred = DihedralAugmentation.apply(pred, inv_dihedral)
                pred = ColorPermutation.apply(pred, inv_color_perm)
                pred = pred[:original_h, :original_w]
                
                if torch.equal(pred, test_output):
                    pass_k_correct = True
                    break
        
        return {
            "correct": correct,
            "pass_k_correct": pass_k_correct,
            "cell_accuracy": cell_accuracy,
            "prediction": final_pred,
            "n_predictions": len(all_predictions)
        }
    
    def evaluate_dataset(
        self,
        data_dir: str | Path,
        split: str = "evaluation",
        n_ensemble: int = 100,
        pass_at_k: int = 2,
        n_recursions: int | None = None,
        max_tasks: int | None = None
    ) -> dict:
        """
        Evaluate on full dataset.
        
        Args:
            data_dir: Path to ARC data
            split: "training" or "evaluation"
            n_ensemble: Ensemble size
            pass_at_k: Pass@k attempts
            n_recursions: Override recursion depth
            max_tasks: Limit number of tasks (for debugging)
        
        Returns:
            Dictionary with aggregate metrics
        """
        dataset = ARCDataset(data_dir, split=split, augment=False)
        
        results = []
        task_ids = []
        
        tasks = list(range(len(dataset)))
        if max_tasks:
            tasks = tasks[:max_tasks]
        
        for idx in tqdm(tasks, desc=f"Evaluating {split}"):
            item = dataset[idx]
            
            result = self.evaluate_task(
                demo_inputs=item["demo_inputs"],
                demo_outputs=item["demo_outputs"],
                test_input=item["test_input"],
                test_output=item["test_output"],
                n_ensemble=n_ensemble,
                pass_at_k=pass_at_k,
                n_recursions=n_recursions
            )
            
            results.append(result)
            task_ids.append(item["task_id"])
        
        # Aggregate
        n_tasks = len(results)
        n_correct = sum(r["correct"] for r in results)
        n_pass_k = sum(r["pass_k_correct"] for r in results)
        avg_cell_acc = np.mean([r["cell_accuracy"] for r in results])
        
        return {
            "n_tasks": n_tasks,
            "n_correct": n_correct,
            "accuracy": n_correct / n_tasks,
            "pass_k_accuracy": n_pass_k / n_tasks,
            "avg_cell_accuracy": avg_cell_acc,
            "results": results,
            "task_ids": task_ids
        }
    
    def save_results(self, results: dict, path: str | Path):
        """Save evaluation results to JSON."""
        # Convert tensors to lists
        serializable = {
            "n_tasks": results["n_tasks"],
            "n_correct": results["n_correct"],
            "accuracy": results["accuracy"],
            "pass_k_accuracy": results["pass_k_accuracy"],
            "avg_cell_accuracy": results["avg_cell_accuracy"],
            "per_task": [
                {
                    "task_id": tid,
                    "correct": r["correct"],
                    "pass_k_correct": r["pass_k_correct"],
                    "cell_accuracy": r["cell_accuracy"]
                }
                for tid, r in zip(results["task_ids"], results["results"])
            ]
        }
        
        with open(path, "w") as f:
            json.dump(serializable, f, indent=2)


def visualize_prediction(
    demo_inputs: list[torch.Tensor],
    demo_outputs: list[torch.Tensor],
    test_input: torch.Tensor,
    test_output: torch.Tensor,
    prediction: torch.Tensor,
    save_path: str | Path | None = None
):
    """Visualize a task with prediction."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    
    # ARC color palette
    arc_colors = [
        "#000000",  # 0: black
        "#0074D9",  # 1: blue
        "#FF4136",  # 2: red
        "#2ECC40",  # 3: green
        "#FFDC00",  # 4: yellow
        "#AAAAAA",  # 5: gray
        "#F012BE",  # 6: magenta
        "#FF851B",  # 7: orange
        "#7FDBFF",  # 8: cyan
        "#870C25",  # 9: maroon
    ]
    cmap = mcolors.ListedColormap(arc_colors)
    
    # Figure layout
    n_demos = len(demo_inputs)
    fig, axes = plt.subplots(2, n_demos + 2, figsize=(3*(n_demos + 2), 6))
    
    # Demo pairs
    for i, (inp, out) in enumerate(zip(demo_inputs, demo_outputs)):
        axes[0, i].imshow(inp.numpy(), cmap=cmap, vmin=0, vmax=9)
        axes[0, i].set_title(f"Demo {i+1} Input")
        axes[0, i].axis("off")
        
        axes[1, i].imshow(out.numpy(), cmap=cmap, vmin=0, vmax=9)
        axes[1, i].set_title(f"Demo {i+1} Output")
        axes[1, i].axis("off")
    
    # Test input
    axes[0, n_demos].imshow(test_input.numpy(), cmap=cmap, vmin=0, vmax=9)
    axes[0, n_demos].set_title("Test Input")
    axes[0, n_demos].axis("off")
    axes[1, n_demos].axis("off")
    
    # Ground truth
    axes[0, n_demos + 1].imshow(test_output.numpy(), cmap=cmap, vmin=0, vmax=9)
    axes[0, n_demos + 1].set_title("Ground Truth")
    axes[0, n_demos + 1].axis("off")
    
    # Prediction
    correct = torch.equal(prediction, test_output)
    color = "green" if correct else "red"
    axes[1, n_demos + 1].imshow(prediction.numpy(), cmap=cmap, vmin=0, vmax=9)
    axes[1, n_demos + 1].set_title(f"Prediction ({'✓' if correct else '✗'})", color=color)
    axes[1, n_demos + 1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate TRM on ARC-AGI")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/arc-agi-1")
    parser.add_argument("--split", type=str, default="evaluation")
    parser.add_argument("--n_ensemble", type=int, default=100)
    parser.add_argument("--pass_at_k", type=int, default=2)
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument("--max_tasks", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    
    args = parser.parse_args()
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint["config"]
    
    model = TinyRecursiveModel(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        n_recursions=config["n_recursions"]
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    # Evaluate
    evaluator = Evaluator(model, device=args.device)
    results = evaluator.evaluate_dataset(
        data_dir=args.data_dir,
        split=args.split,
        n_ensemble=args.n_ensemble,
        pass_at_k=args.pass_at_k,
        max_tasks=args.max_tasks
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Evaluation Results on {args.data_dir} ({args.split})")
    print(f"{'='*60}")
    print(f"Tasks: {results['n_tasks']}")
    print(f"Correct: {results['n_correct']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    print(f"Pass@{args.pass_at_k}: {results['pass_k_accuracy']:.2%}")
    print(f"Avg Cell Accuracy: {results['avg_cell_accuracy']:.2%}")
    
    # Save
    evaluator.save_results(results, args.output)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
