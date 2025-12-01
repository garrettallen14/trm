# Experiments

Scripts for hyperparameter tuning and optimization before full training runs.

## Order of Operations

### 1. Sanity Check (~30 sec)
Verify GPU works and model runs:
```bash
uv run python experiments/sanity_check.py
```

### 2. Profile Model (~2 min)
Find max batch size and throughput:
```bash
uv run python experiments/profile_model.py
```

### 3. Quick Hyperparameter Sweep (~30 min)
Test different settings with short runs:
```bash
# All sweeps
uv run python experiments/quick_sweep.py --sweep all

# Or individual sweeps
uv run python experiments/quick_sweep.py --sweep lr         # Learning rates
uv run python experiments/quick_sweep.py --sweep recursion  # Recursion depths
uv run python experiments/quick_sweep.py --sweep supervision # Loss weighting
uv run python experiments/quick_sweep.py --sweep size       # Model sizes
```

### 4. Full Training
Once you have optimal settings:
```bash
uv run python scripts/train.py --config configs/a40_full.yaml
```

## Expected Results

### Profile (A40 48GB)
- Max batch size: ~128-256
- Throughput: ~200-400 samples/sec
- Estimated training time: 6-12 hours

### Quick Sweep Findings
The sweep will find optimal:
- **Learning rates**: Typically lr_trunk=1e-4, lr_embed=1e-2 works best
- **Recursion depth**: 8-16 steps, more helps but diminishing returns after 12
- **Supervision**: "linear" often beats "uniform"
- **Model size**: 512 is sweet spot (256 too small, 768 marginal gains)

## Output

Results are saved to `experiments/sweep_results.json` with format:
```json
{
  "learning_rates": [
    {"name": "lr_trunk_1e-4", "final_loss": 2.1, "val_accuracy": 0.35, ...},
    ...
  ],
  ...
}
```
