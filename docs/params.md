# TRM Training Configuration

## Full Training Run (Based on Research)

```bash
# The real deal - full TRM training
uv run python experiments/train_full.py --amp --dashboard

# Expected: ~12-24 hours, 35-45% task accuracy on AGI-1
```

### Key Settings from TRM Paper

| Setting | Value | Why |
|---------|-------|-----|
| **n_recursions** | 16 | Paper uses 16, not 8 |
| **augment_factor** | 100 | 8 dihedrals × 12 color perms |
| **no_grad_loops** | 6 | Free test-time simulation |
| **lr_embed** | 1e-2 | 100× higher than trunk |
| **epochs** | 100 | Full training |
| **supervision** | uniform | Deep supervision at all steps |

---

# Sweep Results Analysis

## What We Learned

### 1. Learning Rates
| Config | Loss | Val Acc | Notes |
|--------|------|---------|-------|
| lr_trunk=3e-4 | **1.94** | 57% | Best loss convergence |
| lr_embed=5e-3 | 2.02 | **66%** | Good accuracy |
| lr_embed=1e-3 | 1.98 | 40% | Too slow |
| lr_embed=5e-2 | 2.02 | 37% | Too fast |

**Winner**: `lr_trunk=1e-4`, `lr_embed=5e-3 to 1e-2`

### 2. Recursion Depth
| Depth | Loss | Val Acc | Speed |
|-------|------|---------|-------|
| 2 | 1.99 | 57% | 17.6/s |
| 4 | 2.01 | 44% | 11.1/s |
| **8** | 2.02 | **67%** | 6.2/s |
| 12 | 2.02 | 48% | 4.2/s |
| 16 | 2.02 | 49% | 3.2/s |

**Winner**: `n_recursions=8` - best accuracy, good speed tradeoff

### 3. Model Size
| Size | Loss | Val Acc | Notes |
|------|------|---------|-------|
| 128 | 2.41 | 3% | Way too small |
| 256 | 2.41 | 5% | Still too small |
| **512** | 1.94 | 57%+ | Required for ARC |

**Winner**: `d_model=512` is necessary

### 4. Supervision (partial)
- Linear showed promising mid-training loss (1.85)
- Worth trying in full training

---

## Optimal Training Config

---

# Summary

## Key Findings

| Parameter | Optimal Value | Why |
|-----------|--------------|-----|
| **d_model** | 512 | Smaller fails completely (3-5% acc) |
| **n_recursions** | 8 | Best accuracy (67%), good speed |
| **lr_trunk** | 1e-4 | Good convergence |
| **lr_embed** | 5e-3 | Best accuracy/loss balance |
| **supervision** | linear | Promising intermediate loss |

## Optimal Training Config

```
d_model=512, n_heads=4, n_layers=2
n_recursions=8
lr_trunk=1e-4, lr_embed=5e-3
batch_size=1, grad_accum=16 (effective=16)
augment_factor=10
supervision=linear
epochs=20
```

## Run Full Training

```bash
# Default (20 epochs, ~20 min)
uv run python experiments/train_optimal.py

# Longer run (50 epochs, ~50 min)
uv run python experiments/train_optimal.py --epochs 50
```

**Features:**
- Saves to `experiments/runs/<timestamp>/`
- Saves best model checkpoint
- Saves history (loss, accuracy per epoch)
- OOM handling built in
- Cosine LR schedule

**Expected time:** ~1 min/epoch = ~20 min for 20 epochs