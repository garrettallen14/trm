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

# Install dependencies
echo "Installing Python dependencies..."
echo "(If using macOS, you may need to create a venv first:"
echo "  python3 -m venv venv && source venv/bin/activate)"
pip3 install -r requirements.txt 2>/dev/null || pip install -r requirements.txt 2>/dev/null || echo "Please install dependencies manually: pip install -r requirements.txt"

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
