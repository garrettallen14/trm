#!/usr/bin/env python3
"""
Ablation Study Runner - Systematic hyperparameter search.

Principle: Change ONE thing at a time, measure impact.

Usage:
    python scripts/ablation_study.py --quick  # Fast sanity check
    python scripts/ablation_study.py --full   # Full ablation
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path


# Baseline configuration
BASELINE = {
    "d_model": 256,
    "n_layers": 2,
    "n_heads": 4,
    "num_timesteps": 16,
    "max_demos": 2,
    "batch_size": 64,
    "grad_accum": 2,
    "lr": 1e-4,
    "augment_factor": 50,
    "epochs": 10,  # Short runs for ablation
    "self_cond": False,
}


# Ablation experiments: what to vary
ABLATIONS = {
    # Model capacity
    "d_model": [128, 256, 512, 768],
    "n_layers": [2, 4, 6, 8],
    
    # Diffusion
    "num_timesteps": [8, 16, 32, 64],
    "self_cond": [False, True],
    
    # Context
    "max_demos": [1, 2, 3],
    
    # Training
    "lr": [1e-5, 5e-5, 1e-4, 2e-4, 5e-4],
    "augment_factor": [10, 25, 50, 100],
}


def run_experiment(config: dict, name: str) -> dict:
    """Run a single training experiment."""
    
    cmd = [
        "uv", "run", "python", "experiments/train_diffusion.py",
        "--d_model", str(config["d_model"]),
        "--n_layers", str(config["n_layers"]),
        "--n_heads", str(config["n_heads"]),
        "--num_timesteps", str(config["num_timesteps"]),
        "--max_demos", str(config["max_demos"]),
        "--batch_size", str(config["batch_size"]),
        "--grad_accum", str(config["grad_accum"]),
        "--lr", str(config["lr"]),
        "--augment_factor", str(config["augment_factor"]),
        "--epochs", str(config["epochs"]),
        "--amp",
        "--include_agi2",
    ]
    
    if config.get("self_cond"):
        cmd.append("--self_cond")
    
    print(f"\n{'='*60}")
    print(f"Running: {name}")
    print(f"Config: {config}")
    print(f"{'='*60}\n")
    
    # Run and capture output
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    # Parse final metrics from output
    metrics = {"name": name, "config": config}
    
    for line in result.stdout.split("\n"):
        if "cell_acc=" in line and "task_acc=" in line:
            # Parse: "Epoch X: loss=Y, task_acc=Z%, cell_acc=W%"
            try:
                parts = line.split(",")
                for part in parts:
                    if "task_acc=" in part:
                        metrics["task_acc"] = float(part.split("=")[1].strip().rstrip("%"))
                    if "cell_acc=" in part:
                        metrics["cell_acc"] = float(part.split("=")[1].strip().rstrip("%"))
                    if "loss=" in part:
                        metrics["loss"] = float(part.split("=")[1].strip())
            except:
                pass
    
    return metrics


def run_ablation_study(quick: bool = False):
    """Run systematic ablation study."""
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = Path(f"experiments/ablation_{timestamp}.json")
    
    if quick:
        # Quick mode: just test a few key things
        experiments = [
            ("baseline", BASELINE),
            ("d_model_512", {**BASELINE, "d_model": 512}),
            ("n_layers_4", {**BASELINE, "n_layers": 4}),
            ("self_cond", {**BASELINE, "self_cond": True}),
            ("timesteps_32", {**BASELINE, "num_timesteps": 32}),
        ]
    else:
        # Full ablation: test each dimension
        experiments = [("baseline", BASELINE)]
        
        for param, values in ABLATIONS.items():
            for value in values:
                if value != BASELINE.get(param):
                    config = {**BASELINE, param: value}
                    name = f"{param}_{value}"
                    experiments.append((name, config))
    
    print(f"Running {len(experiments)} experiments")
    print(f"Results will be saved to: {results_file}")
    
    for name, config in experiments:
        try:
            result = run_experiment(config, name)
            results.append(result)
            
            # Save incrementally
            with open(results_file, "w") as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Experiment {name} failed: {e}")
            results.append({"name": name, "error": str(e)})
    
    # Print summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    
    # Sort by task accuracy
    sorted_results = sorted(
        [r for r in results if "task_acc" in r],
        key=lambda x: x.get("task_acc", 0),
        reverse=True
    )
    
    print(f"\n{'Experiment':<30} {'Task Acc':>10} {'Cell Acc':>10} {'Loss':>10}")
    print("-"*60)
    for r in sorted_results[:10]:
        print(f"{r['name']:<30} {r.get('task_acc', 0):>9.1f}% {r.get('cell_acc', 0):>9.1f}% {r.get('loss', 0):>10.4f}")
    
    print(f"\nFull results saved to: {results_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Quick ablation (5 experiments)")
    parser.add_argument("--full", action="store_true", help="Full ablation (all combinations)")
    args = parser.parse_args()
    
    run_ablation_study(quick=args.quick or not args.full)


if __name__ == "__main__":
    main()
