# First Principles Training Methodology

## The Scientific Approach

Instead of ad-hoc tweaking, follow this systematic process:

## Phase 1: Validation (✅ Done)

**Goal:** Prove the pipeline works end-to-end.

```bash
# Quick test: Does it train? Does loss go down?
python experiments/train_diffusion.py --epochs 5 --batch_size 4
```

Checklist:
- [x] Loss decreases
- [x] No crashes/OOM
- [x] Evaluation runs
- [x] Dashboard works

## Phase 2: Learning Rate Finding

**Goal:** Find optimal LR for your model size.

```bash
python scripts/find_lr.py --d_model 512 --n_layers 4
```

**Why it matters:**
- Too low: Training too slow, gets stuck
- Too high: Training unstable, diverges
- Optimal: Fastest stable convergence

**Rule of thumb:**
- Smaller models: LR = 1e-3 to 5e-4
- Larger models: LR = 1e-4 to 5e-5
- With AdamW + warmup: Usually 1e-4 is safe

## Phase 3: Ablation Study

**Goal:** Understand what actually matters.

```bash
# Quick ablation (5 key experiments)
python scripts/ablation_study.py --quick

# Full ablation (all parameters)
python scripts/ablation_study.py --full
```

**Principle:** Change ONE thing at a time.

### Key Dimensions to Test:

| Dimension | Values to Try | Expected Impact |
|-----------|---------------|-----------------|
| `d_model` | 256, 512, 768 | Higher = better reasoning, slower |
| `n_layers` | 2, 4, 6 | Deeper = more complex patterns |
| `num_timesteps` | 16, 32, 64 | More = finer refinement |
| `self_cond` | False, True | Usually helps 1-3% |
| `max_demos` | 1, 2, 3 | More context = better generalization |
| `lr` | 5e-5, 1e-4, 2e-4 | Model-dependent |
| `augment_factor` | 25, 50, 100 | More = less overfitting |

## Phase 4: Scaling Laws

**Goal:** Understand compute/quality tradeoffs.

### Chinchilla-style Analysis:

```
Quality ∝ (Model Size)^a × (Data Size)^b × (Compute)^c
```

For ARC specifically:
- **Model size** matters a lot (reasoning capacity)
- **Data diversity** > data quantity (400 tasks × augmentation)
- **Diffusion steps** = free compute scaling at inference

### Practical Scaling:

| Config | Params | Train Time | Expected Task Acc |
|--------|--------|------------|-------------------|
| d=256, L=2 | 1.7M | 2 min/epoch | 0-1% |
| d=512, L=4 | 7M | 8 min/epoch | 2-5% |
| d=768, L=6 | 15M | 15 min/epoch | 5-10% |
| d=1024, L=8 | 30M | 30 min/epoch | 10-15%? |

## Phase 5: Final Training Run

**Goal:** Train best config to convergence.

Once you know what works:

```bash
python experiments/train_diffusion.py \
  --d_model <best> \
  --n_layers <best> \
  --num_timesteps <best> \
  --self_cond \
  --epochs 200 \
  --include_agi2 \
  --dashboard
```

### Early Stopping Criteria:
- Task accuracy plateaus for 20+ epochs
- Validation loss starts increasing (overfitting)
- AGI-2 accuracy dropping while AGI-1 improves (memorization)

## Phase 6: Inference Optimization

**Goal:** Maximize test-time performance.

For diffusion models, you can trade compute for quality at inference:

```python
# During evaluation, try different step counts:
for steps in [16, 32, 64, 128]:
    results = evaluate(model, num_steps=steps)
    print(f"Steps={steps}: task_acc={results['task_acc']}")
```

Also try:
- Temperature scaling (lower = more confident)
- Multiple samples per task (majority vote)
- Test-time augmentation

---

## Summary: The Process

1. **Validate** → Does it work at all?
2. **Find LR** → What's the optimal learning rate?
3. **Ablate** → What hyperparameters matter?
4. **Scale** → How big should the model be?
5. **Train** → Full run with best config
6. **Optimize** → Maximize inference quality

**Time investment:**
- Phase 1-2: 1 hour
- Phase 3: 2-4 hours (can run overnight)
- Phase 4-5: 1-2 days
- Phase 6: A few hours

---

## Anti-Patterns to Avoid

❌ **Changing multiple things at once**
- You won't know what helped

❌ **Training to convergence before ablating**
- Waste of compute

❌ **Ignoring the validation set**
- You'll overfit without knowing

❌ **Optimizing for cell accuracy only**
- Task accuracy is what matters for ARC

❌ **Using test set for hyperparameter tuning**
- Creates data leakage
