"""
Utility functions for TRM.
"""

import json
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(preference)


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable}


def format_number(n: int) -> str:
    """Format large numbers with K/M/B suffixes."""
    if n >= 1e9:
        return f"{n/1e9:.1f}B"
    elif n >= 1e6:
        return f"{n/1e6:.1f}M"
    elif n >= 1e3:
        return f"{n/1e3:.1f}K"
    return str(n)


def load_arc_task(path: Path) -> dict:
    """Load a single ARC task from JSON."""
    with open(path) as f:
        return json.load(f)


def save_json(data: dict, path: Path):
    """Save data to JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> dict:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


def visualize_grid(
    grid: torch.Tensor | np.ndarray,
    title: str = "",
    ax=None,
    show: bool = True
):
    """Visualize a single ARC grid."""
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
    
    if isinstance(grid, torch.Tensor):
        grid = grid.cpu().numpy()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=9)
    ax.set_title(title)
    ax.axis("off")
    
    # Add grid lines
    h, w = grid.shape
    for i in range(h + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.5)
    for j in range(w + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.5)
    
    if show and ax is None:
        plt.show()


def visualize_task(
    demo_inputs: list,
    demo_outputs: list,
    test_input: torch.Tensor,
    test_output: Optional[torch.Tensor] = None,
    prediction: Optional[torch.Tensor] = None,
    save_path: Optional[Path] = None
):
    """Visualize a complete ARC task."""
    import matplotlib.pyplot as plt
    
    n_demos = len(demo_inputs)
    n_cols = n_demos + 2  # demos + test + output/pred
    
    fig, axes = plt.subplots(2, n_cols, figsize=(3 * n_cols, 6))
    
    # Demo pairs
    for i, (inp, out) in enumerate(zip(demo_inputs, demo_outputs)):
        visualize_grid(inp, f"Demo {i+1} In", ax=axes[0, i], show=False)
        visualize_grid(out, f"Demo {i+1} Out", ax=axes[1, i], show=False)
    
    # Test input
    visualize_grid(test_input, "Test Input", ax=axes[0, n_demos], show=False)
    axes[1, n_demos].axis("off")
    
    # Test output / prediction
    if test_output is not None:
        visualize_grid(test_output, "Ground Truth", ax=axes[0, n_demos + 1], show=False)
    else:
        axes[0, n_demos + 1].axis("off")
    
    if prediction is not None:
        correct = torch.equal(prediction, test_output) if test_output is not None else False
        color = "green" if correct else "red"
        visualize_grid(prediction, f"Prediction {'✓' if correct else '✗'}", 
                      ax=axes[1, n_demos + 1], show=False)
        axes[1, n_demos + 1].set_title(f"Prediction {'✓' if correct else '✗'}", color=color)
    else:
        axes[1, n_demos + 1].axis("off")
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
