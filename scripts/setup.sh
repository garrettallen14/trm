#!/bin/bash
# Setup script for TRM project

set -e

echo "=========================================="
echo "TRM Setup Script"
echo "=========================================="

# Create directories
echo "Creating directories..."
mkdir -p data checkpoints

# Clone ARC-AGI-1
if [ ! -d "data/arc-agi-1" ]; then
    echo "Cloning ARC-AGI-1..."
    git clone https://github.com/fchollet/ARC-AGI.git data/arc-agi-1
else
    echo "ARC-AGI-1 already exists"
fi

# Clone ARC-AGI-2
if [ ! -d "data/arc-agi-2" ]; then
    echo "Cloning ARC-AGI-2..."
    git clone https://github.com/arcprize/ARC-AGI-2.git data/arc-agi-2
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

# Install dependencies
echo "Installing Python dependencies..."
echo "(If using macOS, you may need to create a venv first:"
echo "  python3 -m venv venv && source venv/bin/activate)"
pip3 install -r requirements.txt 2>/dev/null || pip install -r requirements.txt 2>/dev/null || echo "Please install dependencies manually: pip install -r requirements.txt"

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

# Verify setup
echo ""
echo "=========================================="
echo "Verifying setup..."
echo "=========================================="

python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'MPS available: {torch.backends.mps.is_available()}')

from pathlib import Path

agi1 = Path('data/arc-agi-1/data/training')
agi2 = Path('data/arc-agi-2/challenges')

if agi1.exists():
    n_tasks = len(list(agi1.glob('*.json')))
    print(f'ARC-AGI-1 training tasks: {n_tasks}')
else:
    print('WARNING: ARC-AGI-1 not found')

if agi2.exists():
    n_tasks = len(list(agi2.glob('*.json')))
    print(f'ARC-AGI-2 challenges: {n_tasks}')
else:
    print('WARNING: ARC-AGI-2 not found')
"

echo ""
echo "=========================================="
echo "Setup complete!"
echo ""
echo "Next steps:"
echo "  1. Test locally:    python scripts/test_local.py"
echo "  2. Train locally:   python scripts/train.py --config configs/local_test.yaml"
echo "  3. Train on GPU:    python scripts/train.py --config configs/a40_full.yaml"
echo "=========================================="
