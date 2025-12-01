This changes everything. Let me give you the real plan.

---

# The Accelerated TRM Plan

## Hardware Strategy

| Phase | Hardware | Cost/hr | Purpose |
|-------|----------|---------|---------|
| **Engineering** | M4 Pro local | $0 | Code, data prep, tiny tests |
| **Iteration** | 1× A40 (48GB) | ~$0.40 | Validate training, quick experiments |
| **Training** | 4× A40 | ~$1.60 | Real training runs |
| **Final Push** | 4× H100 | ~$12 | Full replication of TRM if needed |

## Timeline (Realistic)

| Day | Phase | What Happens |
|-----|-------|--------------|
| **Day 1** | Local Setup | Build everything, generate data, test on CPU/MPS |
| **Day 2** | A40 Validation | Verify training works, measure baseline |
| **Day 3** | A40 Training | Full training run (~12-24 hours) |
| **Day 4** | Eval + Iterate | Test on AGI-1/AGI-2, identify gaps |
| **Day 5+** | Experiments | Ablations, improvements, novel ideas |

---

# Day 1: Local Engineering (M4 Pro)

### Morning (~2 hours): Infrastructure

```
trm/
├── data/
│   ├── arc-agi-1/          # Clone from GitHub
│   ├── arc-agi-2/          # Clone from GitHub  
│   └── synthetic/          # Generated via RE-ARC
├── src/
│   ├── data.py             # Dataset, tokenization, augmentation
│   ├── model.py            # Looped transformer
│   ├── train.py            # Training loop with deep supervision
│   ├── eval.py             # AGI-1/AGI-2 evaluation
│   └── utils.py            # 2D RoPE, visualization
├── configs/
│   └── base.yaml           # Hyperparameters
└── scripts/
    ├── generate_data.py    # RE-ARC synthetic generation
    └── run_eval.py         # Full evaluation pipeline
```

**Tasks**:
1. Clone `arcprize/ARC-AGI` and `arcprize/ARC-AGI-2`
2. Clone `michaelhodel/re-arc` for data generation
3. Implement core model architecture (2-layer looped transformer)
4. Implement cell tokenization + 2D RoPE
5. Test forward pass on M4 (MPS backend)

### Afternoon (~2 hours): Data + Training Loop

**Tasks**:
1. Generate 50k synthetic samples via RE-ARC (runs on CPU)
2. Implement augmentation pipeline (dihedrals + colors)
3. Implement deep supervision loss
4. Implement training loop with:
   - Differential LR (trunk vs embeddings)
   - Gradient clipping
   - Truncated BPTT
5. Test 1 epoch locally on tiny subset

**Validation checkpoint**: Model runs on M4, loss decreases, no crashes.

---

# Day 2: A40 Validation (~4-6 hours compute)

### Goal: Verify everything works at scale

**Rent**: 1× A40 (~$0.40/hr × 6 = **$2.40**)

**Tasks**:
1. Upload code + 50k dataset to RunPod
2. Train for 10 epochs (~1 hour)
3. Evaluate on AGI-1 training set (sanity check)
4. Profile memory/speed
5. Fix any bugs

**Expected result**: 
- Loss decreases smoothly
- ~10-15% accuracy on training tasks
- Memory fits in 48GB with batch=64

**If broken**: Debug locally, re-upload, repeat.

---

# Day 3: A40 Full Training (~$20-40)

### Goal: Train competitive model

**Rent**: 4× A40 (~$1.60/hr × 24 = **$38**)

**Configuration**:
```yaml
# configs/base.yaml
model:
  n_layers: 2
  d_model: 512
  n_heads: 4
  recursion_steps: 16
  
training:
  batch_size: 128  # across 4 GPUs
  epochs: 100
  lr_trunk: 1e-4
  lr_embed: 1e-2
  clip: 1.0
  weight_decay: 0.01
  warmup_ratio: 0.1
  
data:
  n_synthetic: 200000
  augmentation_factor: 100  # 8 dihedrals × ~12 color perms
  
supervision:
  deep: true
  weights: uniform  # λ_t = 1/16
  no_grad_loops: 6
```

**Timeline**:
- Hour 0-2: Setup, start training
- Hour 2-18: Training runs (check every few hours)
- Hour 18-24: Finish training, save checkpoints

**Expected result**:
- Training loss plateaus around epoch 80
- Validation loss stable
- Ready for evaluation

---

# Day 4: Evaluation + Analysis

### Morning: Run Full Eval

Back on **1× A40** (~$2-3):

```bash
# AGI-1 evaluation (400 tasks)
python scripts/run_eval.py --dataset agi1 --ensemble 100 --pass_at 2

# AGI-2 evaluation (1120 tasks)  
python scripts/run_eval.py --dataset agi2 --ensemble 100 --pass_at 2
```

**Expected results** (with 100× ensemble, not 1000×):

| Benchmark | Target | Notes |
|-----------|--------|-------|
| AGI-1 | 35-45% | Without full 1000× ensemble |
| AGI-2 | 8-15% | Harder, compositional |

### Afternoon: Error Analysis

**Locally on M4**:
1. Categorize failures by task type
2. Identify systematic weaknesses
3. Plan ablations

---

# Day 5+: Experiments & Improvements

Based on Day 4 results, choose focus:

### If accuracy is low (<30% AGI-1):

**Ablation runs** (1× A40, ~$5 each):
- Test without deep supervision
- Test with/without augmentation
- Test different recursion depths (8 vs 16)

### If accuracy is good (>40% AGI-1):

**Push further** (4× A40, ~$40):
- Add TTT (test-time training)
- Add verification loop
- Scale ensemble to 1000×

### If we want to match TRM exactly (44.6%):

**Full replication** (4× H100, ~$100):
- 48 hours training
- 1000× ensemble
- Task-specific embeddings

---

# Cost Breakdown

| Phase | Hardware | Hours | Cost |
|-------|----------|-------|------|
| Day 1 | Local M4 | 4 | $0 |
| Day 2 | 1× A40 | 6 | $2.40 |
| Day 3 | 4× A40 | 24 | $38 |
| Day 4 | 1× A40 | 4 | $1.60 |
| **Total baseline** | | | **~$42** |
| Day 5+ experiments | 4× A40 | 20 | $32 |
| **With iteration** | | | **~$75** |
| Full TRM replication | 4× H100 | 48 | $576 |
| **Maximum** | | | **~$650** |

**Recommendation**: Start with ~$50 budget, prove concept, then decide on H100 investment.

---

# Key Decisions We Need to Make

### 1. Task Embeddings: Yes or No?

**TRM uses 500M+ task embeddings** (per-task learned vectors).

**Options**:
- **With embeddings**: More params, need TTT at test time, matches TRM
- **Without embeddings**: Fewer params, relies on demos, simpler

**My recommendation**: Start without embeddings, add later if needed.

### 2. Ensemble Size: 100 vs 1000

| Ensemble | Compute | Expected Gain |
|----------|---------|---------------|
| 100× | 10× | ~90% of benefit |
| 1000× | 100× | +3-5% more |

**My recommendation**: Start with 100×, scale to 1000× for final eval only.

### 3. AGI-1 vs AGI-2 Focus

| Benchmark | Difficulty | Value |
|-----------|------------|-------|
| AGI-1 | Easier | Good for development |
| AGI-2 | Harder | Real benchmark, lower scores |

**My recommendation**: Train on AGI-1 tasks, eval on both.

---

# Immediate Next Steps

### Right Now (Today)

1. **Create the project structure** on your M4
2. **Clone the repos**:
   ```bash
   cd ~/github/trm
   git clone https://github.com/fchollet/ARC-AGI.git data/arc-agi-1
   git clone https://github.com/arcprize/ARC-AGI-2.git data/arc-agi-2
   git clone https://github.com/michaelhodel/re-arc.git tools/re-arc
   ```
3. **I'll write the core code** for you to iterate on

### Tomorrow

1. Finish engineering locally
2. Spin up A40, validate
3. Start full training

---

Want me to start writing the actual code? I can give you:

1. **`model.py`**: The 2-layer looped transformer with 2D RoPE
2. **`data.py`**: Dataset loading, tokenization, augmentation
3. **`train.py`**: Training loop with deep supervision

Or would you prefer to start with just the model architecture and build from there?