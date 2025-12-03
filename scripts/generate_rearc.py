#!/usr/bin/env python3
"""
Generate synthetic ARC tasks using RE-ARC.

RE-ARC provides procedural generators for each of the 400 ARC training tasks,
allowing unlimited synthetic data generation.

Usage:
    python scripts/generate_rearc.py --num_per_task 50 --output data/re-arc-generated
"""

import argparse
import json
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Generate RE-ARC synthetic data")
    parser.add_argument("--num_per_task", type=int, default=50, 
                        help="Number of synthetic examples per task")
    parser.add_argument("--output", type=str, default="data/re-arc-generated",
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    
    # Check if RE-ARC is available
    rearc_path = Path("data/re-arc")
    if not rearc_path.exists():
        print("ERROR: RE-ARC not found. Run:")
        print("  git clone https://github.com/michaelhodel/re-arc.git data/re-arc")
        print("  cd data/re-arc && pip install -e .")
        sys.exit(1)
    
    # Add RE-ARC to path
    sys.path.insert(0, str(rearc_path))
    
    try:
        from arc import generate_dataset
    except ImportError:
        print("ERROR: RE-ARC not installed. Run:")
        print("  cd data/re-arc && pip install -e .")
        sys.exit(1)
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {args.num_per_task} examples per task...")
    print(f"Output: {output_dir}")
    
    # Generate dataset
    # RE-ARC's generate_dataset returns a dict: task_id -> list of examples
    dataset = generate_dataset(
        n=args.num_per_task,
        seed=args.seed
    )
    
    # Save in ARC-compatible format
    training_dir = output_dir / "training"
    training_dir.mkdir(exist_ok=True)
    
    total_examples = 0
    for task_id, examples in dataset.items():
        # Each example has 'train' and 'test' pairs
        task_data = {
            "train": [],
            "test": []
        }
        
        for ex in examples:
            # RE-ARC format: each example has input/output pairs
            if "train" in ex:
                task_data["train"].extend(ex["train"])
            if "test" in ex:
                task_data["test"].extend(ex["test"])
        
        # Save task
        task_file = training_dir / f"{task_id}.json"
        with open(task_file, "w") as f:
            json.dump(task_data, f)
        
        total_examples += len(examples)
    
    print(f"\nGenerated {total_examples} total examples across {len(dataset)} tasks")
    print(f"Saved to: {training_dir}")
    
    # Also save a simple flat format for easier loading
    flat_dir = output_dir / "flat"
    flat_dir.mkdir(exist_ok=True)
    
    example_id = 0
    for task_id, examples in dataset.items():
        for ex in examples:
            flat_file = flat_dir / f"{example_id:06d}_{task_id}.json"
            with open(flat_file, "w") as f:
                json.dump(ex, f)
            example_id += 1
    
    print(f"Also saved flat format: {flat_dir} ({example_id} files)")


if __name__ == "__main__":
    main()
