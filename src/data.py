"""
Data loading and augmentation for ARC-AGI.

Handles:
- Loading ARC-AGI-1 and ARC-AGI-2 tasks
- Cell-based tokenization
- Dihedral augmentations (8 symmetries)
- Color permutations
- Synthetic data generation via RE-ARC
"""

import json
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class Tokenizer:
    """
    Tokenizer for ARC grids.
    
    Colors 0-9 map to tokens 0-9.
    Token 10 is padding.
    """
    
    PAD_TOKEN = 10
    N_COLORS = 11  # 0-9 + padding
    
    def encode_grid(self, grid: list[list[int]]) -> torch.Tensor:
        """Convert grid to tensor."""
        return torch.tensor(grid, dtype=torch.long)
    
    def decode_grid(self, tensor: torch.Tensor) -> list[list[int]]:
        """Convert tensor back to grid."""
        return tensor.tolist()
    
    def pad_grid(
        self, 
        grid: torch.Tensor, 
        max_h: int, 
        max_w: int
    ) -> torch.Tensor:
        """Pad grid to max size."""
        h, w = grid.shape
        padded = torch.full((max_h, max_w), self.PAD_TOKEN, dtype=torch.long)
        padded[:h, :w] = grid
        return padded


class DihedralAugmentation:
    """
    Dihedral group D4 augmentations (8 symmetries).
    
    Includes:
    - Identity
    - 90° rotation
    - 180° rotation
    - 270° rotation
    - Horizontal flip
    - Vertical flip
    - Diagonal flip (transpose)
    - Anti-diagonal flip
    """
    
    @staticmethod
    def apply(grid: torch.Tensor, transform_id: int) -> torch.Tensor:
        """Apply one of 8 dihedral transforms."""
        if transform_id == 0:
            return grid  # Identity
        elif transform_id == 1:
            return torch.rot90(grid, 1, [0, 1])  # 90°
        elif transform_id == 2:
            return torch.rot90(grid, 2, [0, 1])  # 180°
        elif transform_id == 3:
            return torch.rot90(grid, 3, [0, 1])  # 270°
        elif transform_id == 4:
            return torch.flip(grid, [1])  # Horizontal flip
        elif transform_id == 5:
            return torch.flip(grid, [0])  # Vertical flip
        elif transform_id == 6:
            return grid.T  # Transpose
        elif transform_id == 7:
            return torch.flip(grid.T, [0, 1])  # Anti-diagonal
        else:
            raise ValueError(f"Invalid transform_id: {transform_id}")
    
    @staticmethod
    def random_transform() -> int:
        """Get random transform id."""
        return random.randint(0, 7)


class ColorPermutation:
    """
    Permute colors while preserving structure.
    
    Color 0 (background) is typically preserved.
    Colors 1-9 are permuted randomly.
    """
    
    @staticmethod
    def generate_permutation(preserve_zero: bool = True) -> dict[int, int]:
        """Generate a random color permutation."""
        if preserve_zero:
            colors = list(range(1, 10))
            random.shuffle(colors)
            perm = {0: 0}
            perm.update({i + 1: colors[i] for i in range(9)})
        else:
            colors = list(range(10))
            random.shuffle(colors)
            perm = {i: colors[i] for i in range(10)}
        return perm
    
    @staticmethod
    def apply(grid: torch.Tensor, permutation: dict[int, int]) -> torch.Tensor:
        """Apply color permutation to grid."""
        result = grid.clone()
        for old_color, new_color in permutation.items():
            result[grid == old_color] = new_color
        return result


class ARCTask:
    """Represents a single ARC task."""
    
    def __init__(
        self,
        task_id: str,
        train_pairs: list[dict],
        test_pairs: list[dict]
    ):
        self.task_id = task_id
        self.train_pairs = train_pairs  # [{"input": grid, "output": grid}, ...]
        self.test_pairs = test_pairs
    
    @classmethod
    def from_json(cls, path: Path) -> "ARCTask":
        """Load task from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        task_id = path.stem
        return cls(
            task_id=task_id,
            train_pairs=data["train"],
            test_pairs=data["test"]
        )
    
    def get_demo_grids(self) -> tuple[list, list]:
        """Get demonstration input/output pairs."""
        inputs = [torch.tensor(p["input"], dtype=torch.long) for p in self.train_pairs]
        outputs = [torch.tensor(p["output"], dtype=torch.long) for p in self.train_pairs]
        return inputs, outputs
    
    def get_test_grids(self, idx: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        """Get test input/output pair."""
        pair = self.test_pairs[idx]
        return (
            torch.tensor(pair["input"], dtype=torch.long),
            torch.tensor(pair["output"], dtype=torch.long)
        )


class ARCDataset(Dataset):
    """
    PyTorch Dataset for ARC tasks.
    
    Each item returns:
    - demo_inputs: List of demonstration input grids
    - demo_outputs: List of demonstration output grids
    - test_input: Test input grid
    - test_output: Test output grid (ground truth)
    - task_id: Task identifier
    """
    
    def __init__(
        self,
        data_dir: str | Path,
        split: str = "training",  # "training" or "evaluation"
        augment: bool = True,
        augment_factor: int = 100,  # Number of augmented versions per task
        max_grid_size: int = 30,
        seed: int = 42
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.augment = augment
        self.augment_factor = augment_factor
        self.max_grid_size = max_grid_size
        self.tokenizer = Tokenizer()
        
        random.seed(seed)
        np.random.seed(seed)
        
        # Load tasks
        self.tasks = self._load_tasks()
        
        # Pre-generate augmentation configs for reproducibility
        if augment:
            self._generate_augmentation_configs()
    
    def _load_tasks(self) -> list[ARCTask]:
        """Load all tasks from directory."""
        tasks = []
        
        # Handle different directory structures
        if (self.data_dir / self.split).exists():
            task_dir = self.data_dir / self.split
        elif (self.data_dir / "data" / self.split).exists():
            task_dir = self.data_dir / "data" / self.split
        else:
            # Try challenges/solutions structure (ARC-AGI-2)
            challenges_dir = self.data_dir / "challenges"
            if challenges_dir.exists():
                return self._load_arc_agi_2_tasks(challenges_dir)
            raise ValueError(f"Cannot find tasks in {self.data_dir}")
        
        for task_file in sorted(task_dir.glob("*.json")):
            try:
                task = ARCTask.from_json(task_file)
                tasks.append(task)
            except Exception as e:
                print(f"Warning: Failed to load {task_file}: {e}")
        
        print(f"Loaded {len(tasks)} tasks from {task_dir}")
        return tasks
    
    def _load_arc_agi_2_tasks(self, challenges_dir: Path) -> list[ARCTask]:
        """Load ARC-AGI-2 format tasks."""
        tasks = []
        solutions_dir = challenges_dir.parent / "solutions"
        
        for challenge_file in sorted(challenges_dir.glob("*.json")):
            task_id = challenge_file.stem
            solution_file = solutions_dir / f"{task_id}.json"
            
            try:
                with open(challenge_file) as f:
                    challenge = json.load(f)
                
                # Load solutions if available
                if solution_file.exists():
                    with open(solution_file) as f:
                        solutions = json.load(f)
                else:
                    solutions = None
                
                # Construct task
                train_pairs = challenge.get("train", [])
                test_pairs = []
                
                for i, test in enumerate(challenge.get("test", [])):
                    test_pair = {"input": test["input"]}
                    if solutions and str(i) in solutions:
                        test_pair["output"] = solutions[str(i)]
                    elif "output" in test:
                        test_pair["output"] = test["output"]
                    else:
                        continue  # Skip tests without solutions
                    test_pairs.append(test_pair)
                
                if test_pairs:
                    tasks.append(ARCTask(task_id, train_pairs, test_pairs))
            
            except Exception as e:
                print(f"Warning: Failed to load {challenge_file}: {e}")
        
        print(f"Loaded {len(tasks)} tasks from {challenges_dir}")
        return tasks
    
    def _generate_augmentation_configs(self):
        """Pre-generate augmentation configurations."""
        self.aug_configs = []
        
        for task_idx in range(len(self.tasks)):
            for aug_idx in range(self.augment_factor):
                self.aug_configs.append({
                    "task_idx": task_idx,
                    "dihedral": DihedralAugmentation.random_transform(),
                    "color_perm": ColorPermutation.generate_permutation(preserve_zero=True)
                })
    
    def __len__(self) -> int:
        if self.augment:
            return len(self.aug_configs)
        return len(self.tasks)
    
    def __getitem__(self, idx: int) -> dict:
        if self.augment:
            config = self.aug_configs[idx]
            task = self.tasks[config["task_idx"]]
            dihedral = config["dihedral"]
            color_perm = config["color_perm"]
        else:
            task = self.tasks[idx]
            dihedral = 0
            color_perm = {i: i for i in range(10)}
        
        # Get grids
        demo_inputs, demo_outputs = task.get_demo_grids()
        test_input, test_output = task.get_test_grids()
        
        # Apply augmentations
        def augment_grid(grid):
            grid = DihedralAugmentation.apply(grid, dihedral)
            grid = ColorPermutation.apply(grid, color_perm)
            return grid
        
        demo_inputs = [augment_grid(g) for g in demo_inputs]
        demo_outputs = [augment_grid(g) for g in demo_outputs]
        test_input = augment_grid(test_input)
        test_output = augment_grid(test_output)
        
        return {
            "demo_inputs": demo_inputs,
            "demo_outputs": demo_outputs,
            "test_input": test_input,
            "test_output": test_output,
            "task_id": task.task_id
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate batch of variable-size grids.
    
    Pads all grids to the maximum size in the batch.
    """
    tokenizer = Tokenizer()
    
    # Find max sizes
    max_h, max_w = 0, 0
    for item in batch:
        for grid in item["demo_inputs"] + item["demo_outputs"]:
            max_h = max(max_h, grid.shape[0])
            max_w = max(max_w, grid.shape[1])
        max_h = max(max_h, item["test_input"].shape[0], item["test_output"].shape[0])
        max_w = max(max_w, item["test_input"].shape[1], item["test_output"].shape[1])
    
    # Pad all grids
    batch_demo_inputs = []
    batch_demo_outputs = []
    batch_test_inputs = []
    batch_test_outputs = []
    task_ids = []
    
    for item in batch:
        # Pad demos (assume same number of demos per task)
        padded_demo_in = [tokenizer.pad_grid(g, max_h, max_w) for g in item["demo_inputs"]]
        padded_demo_out = [tokenizer.pad_grid(g, max_h, max_w) for g in item["demo_outputs"]]
        
        batch_demo_inputs.append(padded_demo_in)
        batch_demo_outputs.append(padded_demo_out)
        batch_test_inputs.append(tokenizer.pad_grid(item["test_input"], max_h, max_w))
        batch_test_outputs.append(tokenizer.pad_grid(item["test_output"], max_h, max_w))
        task_ids.append(item["task_id"])
    
    # Stack into tensors
    # Handle variable number of demos per task by finding max and padding
    max_demos = max(len(demos) for demos in batch_demo_inputs)
    
    # Pad tasks with fewer demos using padding token grids
    pad_grid = torch.full((max_h, max_w), tokenizer.pad_token, dtype=torch.long)
    
    for b in range(len(batch)):
        while len(batch_demo_inputs[b]) < max_demos:
            batch_demo_inputs[b].append(pad_grid.clone())
            batch_demo_outputs[b].append(pad_grid.clone())
    
    # Now stack - demo_inputs: list of [batch, h, w] for each demo
    demo_inputs = [
        torch.stack([batch_demo_inputs[b][d] for b in range(len(batch))])
        for d in range(max_demos)
    ]
    demo_outputs = [
        torch.stack([batch_demo_outputs[b][d] for b in range(len(batch))])
        for d in range(max_demos)
    ]
    
    return {
        "demo_inputs": demo_inputs,  # List of [batch, h, w]
        "demo_outputs": demo_outputs,
        "test_input": torch.stack(batch_test_inputs),
        "test_output": torch.stack(batch_test_outputs),
        "task_ids": task_ids,
        "n_demos": [len(item["demo_inputs"]) for item in batch]  # Track actual demo counts
    }


def create_dataloader(
    data_dir: str | Path,
    split: str = "training",
    batch_size: int = 16,
    augment: bool = True,
    augment_factor: int = 100,
    num_workers: int = 4,
    shuffle: bool = True,
    seed: int = 42
) -> DataLoader:
    """Create a DataLoader for ARC tasks."""
    dataset = ARCDataset(
        data_dir=data_dir,
        split=split,
        augment=augment,
        augment_factor=augment_factor,
        seed=seed
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )


if __name__ == "__main__":
    # Test data loading
    import sys
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = "data/arc-agi-1"
    
    print(f"Testing data loading from {data_dir}")
    
    # Load without augmentation
    dataset = ARCDataset(data_dir, augment=False)
    print(f"Dataset size (no aug): {len(dataset)}")
    
    # Get one item
    item = dataset[0]
    print(f"Task: {item['task_id']}")
    print(f"Demo inputs: {len(item['demo_inputs'])} grids")
    print(f"Demo input shapes: {[g.shape for g in item['demo_inputs']]}")
    print(f"Test input shape: {item['test_input'].shape}")
    print(f"Test output shape: {item['test_output'].shape}")
    
    # Load with augmentation
    dataset_aug = ARCDataset(data_dir, augment=True, augment_factor=10)
    print(f"\nDataset size (10x aug): {len(dataset_aug)}")
    
    # Test dataloader
    dataloader = create_dataloader(
        data_dir, 
        batch_size=4, 
        augment=True, 
        augment_factor=10,
        num_workers=0
    )
    
    batch = next(iter(dataloader))
    print(f"\nBatch:")
    print(f"  Demo inputs: {len(batch['demo_inputs'])} demos, shape {batch['demo_inputs'][0].shape}")
    print(f"  Test input shape: {batch['test_input'].shape}")
    print(f"  Test output shape: {batch['test_output'].shape}")
