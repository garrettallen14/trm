# Tiny Recursive Model (TRM) for ARC-AGI

A minimal implementation of a Tiny Recursive Model for the Abstraction and Reasoning Corpus (ARC-AGI) benchmark.

## Overview

This implementation achieves competitive results on ARC-AGI using a small (~7M parameter) recursive transformer with:

- **2-layer looped transformer** with shared weights
- **2D Rotary Position Embeddings** for grid-aware attention
- **Deep supervision** at each recursion step
- **Cell-based tokenization** for precise grid manipulation
- **Test-time augmentation** for robust predictions

## Quick Start

### 1. Local Setup (M4 Pro)

```bash
cd ~/github/trm

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv venv
uv pip install torch numpy einops tqdm pyyaml matplotlib seaborn wandb

# Clone ARC data
git clone https://github.com/fchollet/ARC-AGI.git data/arc-agi-1
```

### 2. RunPod Setup (GPU)

```bash
# One-liner setup
bash scripts/setup_runpod.sh
```

### 3. Experiments (Before Full Training)

```bash
# Sanity check (~30 sec)
uv run python experiments/sanity_check.py

# Profile to find optimal batch size (~2 min)
uv run python experiments/profile_model.py

# Quick hyperparameter sweep (~30 min)
uv run python experiments/quick_sweep.py --sweep all
```

### 4. Full Training

```bash
# After finding optimal hyperparams
uv run python scripts/train.py --config configs/a40_full.yaml
```

### 5. Evaluate

```bash
uv run python scripts/evaluate.py \
    --checkpoint checkpoints/best.pt \
    --data_dir data/arc-agi-1 \
    --n_ensemble 100
```

## Project Structure

```
trm/
├── configs/
│   ├── base.yaml          # Base configuration
│   ├── local_test.yaml    # For local testing
│   └── a40_full.yaml      # Full training on A40
├── data/
│   └── arc-agi-1/         # ARC-AGI-1 dataset
├── experiments/           # Hyperparameter tuning
│   ├── sanity_check.py    # Quick GPU verification
│   ├── profile_model.py   # Find optimal batch size
│   └── quick_sweep.py     # HP sweep before full training
├── src/
│   ├── model.py           # TRM architecture
│   ├── data.py            # Dataset & augmentation
│   ├── train.py           # Training loop
│   └── eval.py            # Evaluation
├── scripts/
│   ├── train.py           # Training entry point
│   ├── evaluate.py        # Evaluation entry point
│   └── setup_runpod.sh    # RunPod setup script
├── checkpoints/           # Saved models
└── docs/                  # Research notes
```

## Architecture

```
Input: Demo pairs (input→output) + test input
  ↓
Cell Embedding (11 tokens: 0-9 colors + pad)
  ↓
2D RoPE Positional Encoding
  ↓
┌─────────────────────────────────┐
│  Looped Transformer (2 layers)  │ ← Repeat K times
│  - Multi-head attention         │   with input injection
│  - Feed-forward network         │
│  - Pre-LayerNorm               │
└─────────────────────────────────┘
  ↓
Output Head (predict cell colors)
  ↓
Deep Supervision Loss at each step
```

## Key Hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| d_model | 512 | Embedding dimension |
| n_heads | 4 | Attention heads |
| n_layers | 2 | Transformer layers (shared) |
| n_recursions | 16 | Recursion steps |
| lr_trunk | 1e-4 | Learning rate for transformer |
| lr_embed | 1e-2 | Learning rate for embeddings (100x) |
| batch_size | 128 | Per-GPU batch size |
| augment_factor | 100 | Augmentations per task |

## Expected Results

| Benchmark | Ensemble=1 | Ensemble=100 | Pass@2 |
|-----------|------------|--------------|--------|
| ARC-AGI-1 | ~25% | ~40% | ~45% |
| ARC-AGI-2 | ~5% | ~10% | ~12% |

## Training Cost

| Hardware | Time | Cost |
|----------|------|------|
| 1× A40 (48GB) | ~48 hours | ~$20 |
| 4× A40 | ~12 hours | ~$20 |
| 4× H100 | ~6 hours | ~$70 |

## References

- [ARC-AGI Benchmark](https://arcprize.org/)
- [TRM Paper (arXiv:2510.04871)](https://arxiv.org/abs/2510.04871)
- [Original ARC Paper](https://arxiv.org/abs/1911.01547)

## License

MIT
