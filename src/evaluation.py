"""
Principled Evaluation for ARC-AGI.

Key principles:
1. STRICT train/test separation with verification
2. Evaluate ALL test pairs per task (not just first)
3. Support both ARC-AGI-1 and ARC-AGI-2
4. Deterministic (no randomness in eval)
5. Log task IDs for audit trail

Usage:
    from src.evaluation import ARCEvaluator
    
    evaluator = ARCEvaluator(
        train_dir="data/arc-agi-1/data/training",
        eval_dirs={
            "agi1": "data/arc-agi-1/data/evaluation",
            "agi2": "data/arc-agi-2"
        }
    )
    
    # Verify no contamination
    evaluator.verify_no_contamination()
    
    # Run evaluation
    results = evaluator.evaluate(model, device)
"""

import json
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from tqdm import tqdm


@dataclass
class EvalResult:
    """Result for a single task evaluation."""
    task_id: str
    test_idx: int
    predicted: torch.Tensor
    target: torch.Tensor
    cell_correct: int
    cell_total: int
    task_correct: bool
    
    @property
    def cell_accuracy(self) -> float:
        return self.cell_correct / self.cell_total if self.cell_total > 0 else 0


@dataclass
class EvalSummary:
    """Summary of evaluation results."""
    dataset_name: str
    task_accuracy: float
    cell_accuracy: float
    tasks_correct: int
    tasks_total: int
    cells_correct: int
    cells_total: int
    task_ids_evaluated: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"{self.dataset_name}: "
            f"task_acc={self.task_accuracy:.1%} ({self.tasks_correct}/{self.tasks_total}), "
            f"cell_acc={self.cell_accuracy:.1%}"
        )


class ARCEvaluator:
    """
    Principled evaluator for ARC-AGI tasks.
    
    Ensures strict train/test separation and comprehensive evaluation.
    """
    
    def __init__(
        self,
        train_dir: str | Path,
        eval_dirs: dict[str, str | Path],
        pad_token: int = 10
    ):
        """
        Args:
            train_dir: Directory containing training tasks
            eval_dirs: Dict mapping dataset names to evaluation directories
            pad_token: Padding token value (default 10)
        """
        self.train_dir = Path(train_dir)
        self.eval_dirs = {k: Path(v) for k, v in eval_dirs.items()}
        self.pad_token = pad_token
        
        # Load task IDs
        self.train_task_ids = self._get_task_ids(self.train_dir)
        self.eval_task_ids = {
            name: self._get_task_ids(path) 
            for name, path in self.eval_dirs.items()
        }
        
        print(f"[Evaluator] Training tasks: {len(self.train_task_ids)}")
        for name, ids in self.eval_task_ids.items():
            print(f"[Evaluator] {name} eval tasks: {len(ids)}")
    
    def _get_task_ids(self, directory: Path) -> set[str]:
        """Get all task IDs from a directory."""
        task_ids = set()
        
        # Try different structures
        for pattern in ["*.json", "data/*/*.json", "*/*.json", "challenges/*.json"]:
            for f in directory.glob(pattern):
                task_ids.add(f.stem)
        
        return task_ids
    
    def verify_no_contamination(self) -> bool:
        """
        Verify that training and evaluation sets are disjoint.
        
        Raises AssertionError if contamination is detected.
        """
        print("\n" + "="*60)
        print("CONTAMINATION CHECK")
        print("="*60)
        
        all_clean = True
        
        for name, eval_ids in self.eval_task_ids.items():
            overlap = self.train_task_ids & eval_ids
            
            if overlap:
                print(f"❌ CONTAMINATION DETECTED in {name}!")
                print(f"   Overlapping task IDs: {sorted(overlap)[:10]}...")
                all_clean = False
            else:
                print(f"✓ {name}: No overlap with training ({len(eval_ids)} tasks)")
        
        print("="*60 + "\n")
        
        if not all_clean:
            raise AssertionError("Train/test contamination detected!")
        
        return True
    
    def _load_task(self, path: Path) -> dict:
        """Load a single task from JSON."""
        with open(path) as f:
            return json.load(f)
    
    def _find_task_file(self, directory: Path, task_id: str) -> Optional[Path]:
        """Find the JSON file for a task ID."""
        # Try different locations
        candidates = [
            directory / f"{task_id}.json",
            directory / "data" / "training" / f"{task_id}.json",
            directory / "data" / "evaluation" / f"{task_id}.json",
            directory / "challenges" / f"{task_id}.json",
        ]
        
        for candidate in candidates:
            if candidate.exists():
                return candidate
        
        # Glob search as fallback
        for f in directory.rglob(f"{task_id}.json"):
            return f
        
        return None
    
    def _load_solutions(self, directory: Path, task_id: str) -> Optional[dict]:
        """Load solutions for ARC-AGI-2 format."""
        solutions_file = directory / "solutions" / f"{task_id}.json"
        if solutions_file.exists():
            with open(solutions_file) as f:
                return json.load(f)
        return None
    
    @torch.no_grad()
    def evaluate_single(
        self,
        model,
        device: torch.device,
        task_data: dict,
        task_id: str,
        solutions: Optional[dict] = None,
        generate_fn: Optional[Callable] = None
    ) -> list[EvalResult]:
        """
        Evaluate a single task (all test pairs).
        
        Args:
            model: The model to evaluate
            device: Device to run on
            task_data: Task dictionary with 'train' and 'test'
            task_id: Task identifier
            solutions: Optional solutions dict (for AGI-2)
            generate_fn: Optional custom generation function
            
        Returns:
            List of EvalResult for each test pair
        """
        results = []
        
        # Extract demos
        demo_inputs = [
            torch.tensor(p["input"], dtype=torch.long, device=device).unsqueeze(0)
            for p in task_data["train"]
        ]
        demo_outputs = [
            torch.tensor(p["output"], dtype=torch.long, device=device).unsqueeze(0)
            for p in task_data["train"]
        ]
        
        # Evaluate EACH test pair
        for test_idx, test_pair in enumerate(task_data["test"]):
            test_input = torch.tensor(
                test_pair["input"], dtype=torch.long, device=device
            ).unsqueeze(0)
            
            # Get ground truth
            if "output" in test_pair:
                test_output = test_pair["output"]
            elif solutions and str(test_idx) in solutions:
                test_output = solutions[str(test_idx)]
            else:
                continue  # Skip if no ground truth
            
            test_output = torch.tensor(
                test_output, dtype=torch.long, device=device
            ).unsqueeze(0)
            
            # Generate prediction
            try:
                if generate_fn:
                    pred = generate_fn(model, demo_inputs, demo_outputs, test_input)
                else:
                    # Default: use model.generate if available
                    if hasattr(model, 'generate'):
                        result = model.generate(demo_inputs, demo_outputs, test_input)
                        pred = result['prediction'] if isinstance(result, dict) else result
                    else:
                        # Fallback: forward pass with argmax
                        output = model(demo_inputs, demo_outputs, test_input)
                        logits = output['logits'] if isinstance(output, dict) else output
                        pred = logits.argmax(dim=-1)
                        
                        # Reshape if needed
                        if pred.dim() == 2:  # [batch, seq]
                            h, w = test_output.shape[1], test_output.shape[2]
                            pred = pred.view(1, h, w)
            except Exception as e:
                # On error, count as wrong
                print(f"Warning: Generation failed for {task_id}: {e}")
                pred = torch.zeros_like(test_output)
            
            # Ensure shapes match
            if pred.shape != test_output.shape:
                # Reshape or pad as needed
                h, w = test_output.shape[1], test_output.shape[2]
                if pred.numel() == h * w:
                    pred = pred.view(1, h, w)
                else:
                    # Shape mismatch - count as wrong
                    pred = torch.zeros_like(test_output)
            
            # Compute metrics
            mask = test_output != self.pad_token
            cells_match = (pred == test_output) & mask
            cell_correct = cells_match.sum().item()
            cell_total = mask.sum().item()
            task_correct = (cell_correct == cell_total)
            
            results.append(EvalResult(
                task_id=task_id,
                test_idx=test_idx,
                predicted=pred.cpu(),
                target=test_output.cpu(),
                cell_correct=cell_correct,
                cell_total=cell_total,
                task_correct=task_correct
            ))
        
        return results
    
    @torch.no_grad()
    def evaluate(
        self,
        model,
        device: torch.device,
        datasets: Optional[list[str]] = None,
        max_tasks: Optional[int] = None,
        show_progress: bool = True,
        generate_fn: Optional[Callable] = None
    ) -> dict[str, EvalSummary]:
        """
        Run full evaluation on specified datasets.
        
        Args:
            model: Model to evaluate
            device: Device to run on
            datasets: List of dataset names to evaluate (default: all)
            max_tasks: Max tasks per dataset (default: all)
            show_progress: Show progress bar
            generate_fn: Custom generation function
            
        Returns:
            Dict mapping dataset names to EvalSummary
        """
        model.eval()
        
        if datasets is None:
            datasets = list(self.eval_dirs.keys())
        
        summaries = {}
        
        for dataset_name in datasets:
            if dataset_name not in self.eval_dirs:
                print(f"Warning: Unknown dataset {dataset_name}")
                continue
            
            eval_dir = self.eval_dirs[dataset_name]
            task_ids = sorted(self.eval_task_ids[dataset_name])
            
            if max_tasks:
                task_ids = task_ids[:max_tasks]
            
            all_results = []
            
            iterator = tqdm(task_ids, desc=f"Eval {dataset_name}") if show_progress else task_ids
            
            for task_id in iterator:
                task_file = self._find_task_file(eval_dir, task_id)
                if task_file is None:
                    continue
                
                task_data = self._load_task(task_file)
                solutions = self._load_solutions(eval_dir, task_id)
                
                results = self.evaluate_single(
                    model, device, task_data, task_id, solutions, generate_fn
                )
                all_results.extend(results)
            
            # Aggregate results
            tasks_correct = sum(1 for r in all_results if r.task_correct)
            tasks_total = len(all_results)
            cells_correct = sum(r.cell_correct for r in all_results)
            cells_total = sum(r.cell_total for r in all_results)
            
            summaries[dataset_name] = EvalSummary(
                dataset_name=dataset_name,
                task_accuracy=tasks_correct / tasks_total if tasks_total > 0 else 0,
                cell_accuracy=cells_correct / cells_total if cells_total > 0 else 0,
                tasks_correct=tasks_correct,
                tasks_total=tasks_total,
                cells_correct=cells_correct,
                cells_total=cells_total,
                task_ids_evaluated=task_ids
            )
            
            print(summaries[dataset_name])
        
        return summaries


def create_evaluator(
    data_root: str = "data",
    include_agi2: bool = True
) -> ARCEvaluator:
    """
    Create a standard evaluator for ARC-AGI.
    
    Args:
        data_root: Root directory containing data
        include_agi2: Whether to include ARC-AGI-2
        
    Returns:
        Configured ARCEvaluator
    """
    data_root = Path(data_root)
    
    eval_dirs = {
        "agi1_eval": data_root / "arc-agi-1" / "data" / "evaluation"
    }
    
    if include_agi2:
        agi2_path = data_root / "arc-agi-2"
        if agi2_path.exists():
            eval_dirs["agi2"] = agi2_path
    
    return ARCEvaluator(
        train_dir=data_root / "arc-agi-1" / "data" / "training",
        eval_dirs=eval_dirs
    )


if __name__ == "__main__":
    """Test the evaluator setup."""
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data")
    args = parser.parse_args()
    
    print("Creating evaluator...")
    evaluator = create_evaluator(args.data_root, include_agi2=True)
    
    print("\nVerifying no contamination...")
    evaluator.verify_no_contamination()
    
    print("\n✓ Evaluator ready!")
    print(f"  Training tasks: {len(evaluator.train_task_ids)}")
    for name, ids in evaluator.eval_task_ids.items():
        print(f"  {name}: {len(ids)} tasks")
