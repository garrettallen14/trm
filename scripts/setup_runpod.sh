#!/bin/bash
# Setup script for RunPod GPU instance
# Uses uv for fast package management

set -e

echo "=========================================="
echo "TRM RunPod Setup"
echo "=========================================="

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env 2>/dev/null || true
    export PATH="$HOME/.local/bin:$PATH"
fi

echo "uv version: $(uv --version)"

# Create venv and install deps
echo "Creating virtual environment..."
uv venv

echo "Installing dependencies..."
uv pip install torch numpy einops tqdm pyyaml matplotlib seaborn wandb fastapi uvicorn

# Clone ARC data if not present
if [ ! -d "data/arc-agi-1" ]; then
    echo "Cloning ARC-AGI-1..."
    mkdir -p data
    git clone --depth 1 https://github.com/fchollet/ARC-AGI.git data/arc-agi-1
else
    echo "ARC-AGI-1 already exists"
fi

if [ ! -d "data/arc-agi-2" ]; then
    echo "Cloning ARC-AGI-2..."
    mkdir -p data
    git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git data/arc-agi-2
else
    echo "ARC-AGI-2 already exists"
fi


# Clone RE-ARC (synthetic data generation)
if [ ! -d "data/re-arc" ]; then
    echo "Cloning RE-ARC (synthetic data generator)..."
    git clone https://github.com/michaelhodel/re-arc.git data/re-arc
else
    echo "RE-ARC already exists"
fi


# Generate RE-ARC synthetic data (if RE-ARC exists and data not generated)
if [ -d "data/re-arc" ] && [ ! -d "data/re-arc-generated" ]; then
    echo "Generating RE-ARC synthetic data (this may take a few minutes)..."
    cd data/re-arc
    pip install -e . 2>/dev/null || pip3 install -e . 2>/dev/null || true
    cd ../..
    python scripts/generate_rearc.py --num_per_task 50 --output data/re-arc-generated 2>/dev/null || echo "RE-ARC generation skipped (run manually: python scripts/generate_rearc.py)"
else
    echo "RE-ARC data already generated or RE-ARC not installed"
fi

# Verify GPU
echo ""
echo "=========================================="
echo "Verifying GPU..."
echo "=========================================="

uv run python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Sanity check:  uv run python experiments/sanity_check.py"
echo "  2. Profile:       uv run python experiments/profile_model.py"
echo "  3. Quick sweep:   uv run python experiments/quick_sweep.py --sweep lr"
echo "  4. Full train:    uv run python scripts/train.py --config configs/a40_full.yaml"
echo "=========================================="
