# TRM Research Roadmap

## Current Status (Dec 2, 2025)

**Baseline Training Running:**
- Config: 8 recursions, 100× augmentation, 4 no-grad loops
- Progress: Epoch 8, ~73% cell accuracy, 0% task accuracy
- ETA: ~7-8 more days for 100 epochs

**Key Insight**: Cell accuracy is improving steadily. Task accuracy will emerge once cell acc > 90%.

---

## What We Know (From Research Deep Dive)

### Performance Attribution (TRM Paper Ablations)

| Factor | Contribution | Our Status |
|--------|--------------|------------|
| Deep supervision | ~40% | ✅ Implemented |
| Test-time ensembling (1000×) | ~25% | ❌ Not yet |
| Task embeddings (500M params!) | ~20% | ❌ Not using |
| Recursion depth | ~10% | ✅ 8 steps |
| Architecture | ~5% | ✅ Baseline |

### The Controversial Truth

> "The architecture matters less than supervision and inference tricks."

TRM's 44.6% comes from:
1. **Deep supervision** at every recursion step
2. **Massive ensembling** (1000 augmented versions per task)
3. **Task-specific embeddings** (makes each task easier)
4. **TTT** (fine-tune on each task at test time)

The looping/recursion is ~5% of the gains!

---

## Novel Architecture Experiments

### 1. Discrete Diffusion TRM ⭐ (Highest Priority)

**File:** `experiments/diffusion_trm.py`

**Hypothesis:** Replace ad-hoc recursion with principled discrete diffusion.

| Aspect | Standard TRM | Diffusion TRM |
|--------|--------------|---------------|
| Init | Zero/copy grid | Random noise |
| Iteration | Residual update | Denoise step |
| Supervision | Per-step loss | Per-timestep loss (free!) |
| Theory | Ad-hoc | Score matching |

**Why promising:**
- Mathematically principled (proven theory)
- Natural curriculum (coarse → fine)
- Can use diffusion tricks (CFG, scheduling)
- Same compute pattern

**Experiment Plan:**
```bash
# Day 1: Prototype on tiny subset
python experiments/diffusion_trm.py  # Already created

# Day 2: Train on full data (small model)
python experiments/train_diffusion.py --d_model 256 --epochs 10

# Day 3: Compare to baseline TRM
# Measure: loss curves, cell acc, task acc
```

---

### 2. Adaptive Halting (PonderNet-style)

**File:** `experiments/adaptive_trm.py`

**Hypothesis:** Don't waste compute on easy tasks.

| Task Type | Fixed TRM | Adaptive TRM |
|-----------|-----------|--------------|
| Simple rotation | 8 steps | 2-3 steps |
| Complex composition | 8 steps (underfits) | 16-32 steps |
| Average | 8 steps | ~5 steps (2× faster) |

**Why promising:**
- 2× faster inference on average
- Better accuracy on hard tasks (more iterations)
- Learned, not heuristic

**Experiment Plan:**
```bash
# Day 1: Test SimpleAdaptiveTRM
python experiments/adaptive_trm.py

# Day 2: Train with ponder cost
python experiments/train_adaptive.py --lambda_p 0.01

# Day 3: Analyze step distribution
# Q: Which tasks need more steps? Correlate with complexity.
```

---

### 3. Object-Centric Tokenization

**Hypothesis:** ARC is about objects, not pixels.

Current: `tokens = grid.flatten()` → 900 tokens for 30×30
Proposed: `tokens = detect_objects(grid)` → 10-50 tokens

**Why promising:**
- +15% on relational tasks (research finding)
- 10-50× fewer tokens
- Better spatial understanding

**Implementation Sketch:**
```python
class ObjectTokenizer:
    def __call__(self, grid):
        # 1. Connected component detection
        objects = find_connected_components(grid)
        
        # 2. Extract object features
        features = []
        for obj in objects:
            features.append({
                'color': obj.color,
                'bbox': obj.bounding_box,
                'shape': obj.shape_embedding,
                'position': obj.center
            })
        
        # 3. Encode as tokens
        return self.embed_objects(features)
```

**Experiment Plan:**
```bash
# Day 1: Implement object detection
# Day 2: Compare token counts (cell vs object)
# Day 3: Train small model with object tokens
```

---

### 4. Hybrid Latent/Token Switching

**From research:** COCONUT, SwiReasoning, TaH models

**Hypothesis:** Think in latent space, output in token space.

- **Latent mode:** Fast, parallel, for spatial reasoning
- **Token mode:** Explicit, verifiable, for output

**Why promising:**
- 2-7× efficiency gains (research finding)
- Best of both worlds

**Experiment Plan:**
```bash
# Week 2: After other experiments
# Requires more engineering
```

---

## Experiment Priority Queue

### This Week (While Baseline Runs)

| Priority | Experiment | Est. Time | Expected Gain |
|----------|------------|-----------|---------------|
| 1 | Discrete Diffusion | 2 days | Novel architecture |
| 2 | Adaptive Halting | 1 day | 2× inference speed |
| 3 | TTT Implementation | 1 day | +5-10% accuracy |

### After Baseline Completes

| Priority | Experiment | Est. Time | Expected Gain |
|----------|------------|-----------|---------------|
| 1 | Ensemble (100×) | 4 hours | +10-15% accuracy |
| 2 | Pass@2 voting | 1 hour | +3-5% accuracy |
| 3 | Object tokenization | 3 days | +10% on relational |

---

## Key Metrics to Track

### Training Metrics
- Loss (should decrease)
- Cell accuracy (should increase)
- Task accuracy (will stay 0% until cell acc > 90%)
- Steps per second
- Memory usage

### Evaluation Metrics (After Training)
- Task accuracy on ARC-AGI-1 eval (400 tasks)
- Task accuracy on ARC-AGI-2 (1120 tasks)
- Pass@1 vs Pass@2
- Accuracy by task category (rotation, fill, composition, etc.)

### Novel Architecture Metrics
- Compute efficiency (FLOPs per task)
- Adaptive: Steps distribution
- Diffusion: Denoising quality per timestep

---

## Hardware Considerations

### Current Setup (RunPod A40)
- 47GB VRAM
- Good for: Training baseline, small experiments
- Limit: ~batch_size 12 without OOMs

### For Full TRM Replication
- Would need: 4× H100 (~$12/hr)
- 48 hours training
- 1000× ensemble at inference

### For Novel Experiments
- A40 is sufficient
- Focus on architectural innovation, not scale

---

## Success Criteria

### Short-term (This Week)
- [ ] Baseline reaches 80%+ cell accuracy
- [ ] Discrete diffusion prototype works
- [ ] Adaptive halting shows variable steps

### Medium-term (2 Weeks)
- [ ] Baseline reaches 30%+ task accuracy
- [ ] One novel architecture matches baseline
- [ ] TTT adds +5% accuracy

### Long-term (1 Month)
- [ ] Beat TRM's 44.6% with novel approach
- [ ] Publishable insight about what matters
- [ ] Working system for ARC-AGI-2

---

## Open Questions

1. **Does diffusion actually help, or is it just recursion with extra steps?**
   - Experiment: Compare same compute budget

2. **How much do task embeddings matter?**
   - TRM: 500M params in embeddings
   - Without: Can we match with TTT?

3. **What's the minimum ensemble size?**
   - TRM uses 1000×
   - Can we get 90% of gains with 100×?

4. **Why does cell accuracy plateau before task accuracy emerges?**
   - Is it distribution shift?
   - Or just statistical (need >99% per-cell for 100% task)?

---

## References

- TRM Paper: Samsung AI (2025)
- PonderNet: Banino et al. (2021)
- Discrete Diffusion: Austin et al. (2021)
- ARC-AGI: Chollet (2019)
- Deep Supervision: Lee et al. (2015)

---

*Last updated: Dec 2, 2025*
