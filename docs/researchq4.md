# Deep Reflection on TRM Research: Round 3

These three queries directly address the gaps I identified. Let me analyze each deeply, then synthesize everything into a final action plan.

---

## Query 13: ARC-AGI-2 Task Categories

### What I Learned

ARC-AGI-2 is fundamentally different from AGI-1:

| Metric | AGI-1 | AGI-2 |
|--------|-------|-------|
| Tasks | ~400 | ~1,120 |
| Depth ≥3 | ~20% | ~50% |
| AI SOTA | ~55% | ~4–29% |
| Human avg | ~85% | ~60% |

The four categories with their AI failure rates:

| Category | % of Tasks | AI Failure Rate |
|----------|------------|-----------------|
| **Symbolic Interpretation** | 25% | 80–90% |
| **Multi-Rule Compositional** | 30% | 85–95% |
| **Contextual Rule Application** | 25% | 75–85% |
| **In-Context Symbol Definition** | 20% | 70–80% |

The **Multi-Rule Compositional** category is where everything breaks: 85–95% failure rate, depth 3–6+, requires `compose_chain` and `interact_rule` primitives that current models can't handle.

Key primitives breakdown:
- **Spatial**: ~50 ops (rotate, translate, mirror)
- **Object**: ~40 ops (detect, extend, connect)
- **Conditional**: ~30 ops (gate, modulate, apply_if)
- **Color**: ~40 ops (remap, fill, decode)

### What Intrigues Me

The **"gravity + color sort"** example (task cbebaa4b) is exactly the kind of multi-step composition that breaks models. It requires:
1. Simulate falling (physics-like)
2. Then sort by color (abstract rule)
3. Interaction between steps (sorting depends on final positions)

This is 3 primitives with **dependencies between them**—not just a chain.

The **~25% task overlap** between categories suggests many tasks are "hybrid challenges." A model that excels at one category might still fail on hybrid tasks.

**70%+ of eval tasks require depth ≥3** confirms that AGI-2 is specifically designed to break shallow models. Our fixed 8-iteration baseline might not be enough—we may need 16+ for the hard subset.

### What I'm Skeptical About

- **"90% solvable with 50–100 primitives"** sounds like post-hoc analysis. The DSL was designed by looking at tasks. True generalization would require primitives that emerge from learning, not hand-coding.
- **Human calibration at 60%** seems low. If average humans solve 60%, but "experts" solve 100%, there's huge variance. The benchmark might be testing puzzle-solving skill as much as intelligence.
- **o3 at 29% with high-compute** vs. 4% standard is a 7× gap from just more search. This suggests the tasks are brute-forceable if you throw enough compute—concerning for "fluid intelligence" claims.

### Underappreciated Insight

**In-Context Symbol Definition** (20% of tasks, 70–80% AI failure) is actually the most interesting category for tiny models. It tests whether the model can **learn a new symbol mapping from 2–3 examples**, which is exactly what TTT should enable. A model that masters this category would have genuine few-shot abstraction.

The primitive `infer_definition` (from I/O pairs) is essentially what a neural network does during training. If we can make this work at inference time (via TTT or latent adaptation), we solve the category.

---

## Query 14: Deep Supervision Implementation

### What I Learned

This is the most actionable query. Core implementation:

```python
# Looped transformer with deep supervision
for t in range(1, L+1):
    h_t = f_theta(h_{t-1}, x)  # Shared block
    y_hat_t = head(h_t)        # Prediction at step t
    loss_t = lambda_t * criterion(y_hat_t, y)
    total_loss += loss_t
```

**Loss weighting schemes**:

| Scheme | Formula | Best For |
|--------|---------|----------|
| **Uniform** | λ_t = 1/L | Theory, stable training |
| **Linear Ramp** | λ_t = t / Σs | Reasoning, convergence |
| **Exponential** | λ_t = α^(t-1) | Deep refinement |
| **Sigmoid** | λ_t ∝ 1/(1+e^(-β(t-μ))) | Adaptive depth |

The **DIS (Deep Improvement Supervision)** finding is key: Treat recursion as a **diffusion/denoising process**. Early iterations predict noisy outputs, late iterations refine. This reframes the problem—we're not predicting the answer L times, we're progressively denoising it.

**Gradient flow**: Full BPTT through all iterations vs. truncated (last 4–8 steps):
- Full: Better credit assignment, 4× memory
- Truncated: 75% memory savings, approximate early gradients

The theoretical result: Under uniform weighting, looped transformers achieve **linear convergence** despite weight sharing, via "gradient dominance" (gradients always point toward global minima).

### What Intrigues Me

**Progressive/diffusion targets** are fascinating:
- **Iteration 1**: Predict 20% of output (heavily masked)
- **Iteration 4**: Predict 60% of output
- **Iteration 8**: Predict 100% of output

This gives the model a "curriculum within each forward pass"—early steps learn coarse structure, late steps learn details. For grids, this could be:
- Early: Get the output dimensions right
- Middle: Get major objects right
- Late: Get exact cell colors right

The **improvement-based loss** is clever:
```
loss_t = ||y_hat_t - y_hat_{t-1}|| + α * ||y_hat_t - y||
```

This explicitly rewards *progress*—the model should improve each iteration, not just eventually converge. Without this, it might learn to output the answer at step 1 and do nothing after.

**Stop-grad on cross-attention keys** in DIS isolates gradients per iteration—each step learns independently, preventing early steps from "depending" on late corrections.

### What I'm Skeptical About

- **"Linear convergence in O(1/L) epochs"** sounds too good. In practice, optimization has many local minima. The theory assumes population loss; real batches are noisy.
- **Sigmoid ramp with tunable β, μ** adds hyperparameters. For a tiny model, we might not have enough signal to tune these properly. Uniform or linear might be safer.
- **Truncated BPTT losing long-horizon credit** is a real problem for ARC. If the model needs 16 steps but we only backprop through 8, the first 8 steps get no direct gradient. The "consistency loss" (KL between h_t and h_{t+1}) is a hack, not a solution.

### Underappreciated Insight

**"Spectral radius < 1"** for Jacobians at each step is a stability condition. If the Jacobian has eigenvalues > 1, errors amplify across iterations (the "recursion drift" failure mode). We could **monitor this during training** and add a regularization term:

```python
jacobian = torch.autograd.functional.jacobian(f_theta, h_t)
spectral_radius = torch.linalg.eigvals(jacobian).abs().max()
spectral_loss = max(0, spectral_radius - 0.99) ** 2
```

This would directly prevent drift, not just hope the model learns stability.

---

## Query 15: Symbolic Verifiers in Neural Loops

### What I Learned

Three concrete architectures:

| System | Neural Component | Verifier | Feedback Mechanism |
|--------|-----------------|----------|-------------------|
| **NSA** | 452M decoder generating DSL tokens | ARC-DSL execution | Failure trace → TTT gradient (5–10 steps) |
| **Agemo** | GPT-4o/Claude generating code | Python interpreter | Error prompt → chain-of-critique (3–5 revisions) |
| **GridCoder** | 452M decoder generating primitives | On-the-fly DSL parser | Backtrack on partial mismatch |

The common loop:
1. Neural model generates candidate program/output
2. Symbolic verifier executes on input, compares to expected output
3. On mismatch: Feed error back (trace/prompt/gradient)
4. Repeat until match or timeout (10–100 iterations)

**NSA's "failure trace"** is clever: A heatmap of which cells are wrong becomes part of the input for the next attempt. The model learns "the bottom-right corner is wrong, fix that."

**GridCoder's partial verification** is efficient: Don't wait until the full program is generated—verify subtrees as they're built. `rotate` alone can be verified before `rotate + fill` is complete. Early pruning saves 50% compute.

Results: **+10–20% over neural-only** on AGI-2 subsets. NSA hits 45% on AGI-1 (vs. 27% baseline).

### What Intrigues Me

**Hindsight relabeling** in PeARL: When verification fails, the verifier doesn't just reject—it generates **synthetic fixes**. "You output `rotate`, but the correct answer needs `rotate + mirror`. Here's the fixed program." This becomes training data for the neural model.

This is **free supervision**—every failure generates a (wrong_program, correct_program) pair. Over many iterations, the model learns from its mistakes.

**Rejection + mutation** is a genetic algorithm in disguise:
1. Generate 32 candidates
2. Verify all
3. Keep the best (highest partial match)
4. Mutate (swap one primitive)
5. Repeat

For tiny models that can't "think" deeply, this external loop might be more effective than internal recursion.

The **2–5× compute overhead** is actually reasonable. If verification converts 60% of near-misses into successes, that's worth 2× compute. The Pareto curve shifts favorably.

### What I'm Skeptical About

- **"10–100 iterations"** with a 452M model per iteration is not cheap. 100 × 452M = 45B params worth of compute per task. The "tiny model" framing hides this.
- **"Relational DSL enables sub-rule verification"** requires the DSL to be structured for partial execution. If your primitive `conditional_fill(rotate(x))` can't be verified without the full chain, you lose the benefit.
- **Feedback via prompts** (Agemo style) assumes the LLM can understand error messages. For a 7M model, we can't do "chain-of-critique"—the model isn't that smart. Gradient-based feedback (NSA style) is more appropriate.

### Underappreciated Insight

**Partial match thresholds** (e.g., 80% cell accuracy) instead of exact match could be useful for ranking candidates. A candidate that gets 90% of cells right is better than one that gets 50% right, even if both technically "fail." Use this for:
1. Early stopping (if 95%+ match, probably correct)
2. Candidate ranking (pick the highest partial match)
3. Curriculum (train more on 80–95% failures than on 0–50% failures)

The **type-checking during generation** (GridCoder) is clever for tiny models. Instead of generating then verifying, constrain generation to only valid next-tokens. This is like grammar-constrained decoding but for DSLs.

---

# Grand Synthesis: All Queries Combined

Across 8 queries and 4 analysis rounds, here's what I've learned:

## The Architecture Stack (Confirmed)

| Component | Choice | Confidence |
|-----------|--------|------------|
| **Tokenization** | Cell-based (1×1) | High |
| **Positional Encoding** | 2D RoPE | High |
| **Normalization** | Pre-LN + Spectral | High |
| **Looping** | Fixed iterations (start 8, scale to 16) | Medium |
| **Deep Supervision** | Linear ramp, loss at 2,4,6,8 | High |
| **Halting** | Entropy-based threshold (no learned head) | Medium |

## The Training Stack (Confirmed)

| Component | Choice | Confidence |
|-----------|--------|------------|
| **Data** | RE-ARC DSL, 200k–500k samples | High |
| **Augmentation** | 1000× (dihedrals + colors) | High |
| **Curriculum** | Depth (2→8 loops), then category (core→compositional) | High |
| **Optimizer** | AdamW, LR=1e-4, clip=1.0 | High |
| **Supervision** | Deep (linear ramp) + progressive targets | High |

## The Inference Stack (Confirmed)

| Component | Choice | Confidence |
|-----------|--------|------------|
| **TTT** | 20 steps, LR=1e-4, final layer only | High |
| **D_TTT** | 100 augmented examples via L1O | High |
| **Ensembling** | 10 runs, majority vote | High |
| **Verification** | DSL-based, 3 retries on failure | Medium |
| **Early Exit** | Entropy < 0.3 | Low (needs tuning) |

## Key Insights Ranked by Importance

1. **Deep supervision drives 80% of gains** (Query 12, 14). Recursion without intermediate losses is just expensive depth.

2. **ARC-AGI-2's compositional category breaks everything** (Query 13). 85–95% failure on depth 3–6+ tasks. This is the real target.

3. **TTT gives +6× over frozen baselines** (Query 10). Non-negotiable for any competitive system.

4. **Verification loops convert 60% of near-misses** (Query 12, 15). Cheap +20% if we implement correctly.

5. **The "tiny" model is compute-expensive** (Query 12). 7M × 1000 ensembles × 16 loops = not actually cheap.

## The Experimental Plan (Final Version)

### Phase 0: Infrastructure (Week 0)
1. Set up ARC-AGI-2 eval pipeline (GitHub repo)
2. Implement RE-ARC data generation (clone michaelhodel/re-arc)
3. Set up logging: per-category accuracy, attention entropy, loss curves

### Phase 1: Baseline Looped Transformer (Week 1–2)
**Goal**: Establish reproducible baseline on AGI-1, measure AGI-2 gap

1. **Architecture** (10M params):
   - 3-layer shared block, d_model=256, 4 heads
   - Cell tokenization (max 900 tokens)
   - 2D RoPE positional encoding
   - Pre-LN, spectral norm on all linears

2. **Training**:
   - 200k RE-ARC samples
   - Deep supervision: Linear ramp (λ_t = t/36 for t=2,4,6,8)
   - Truncated BPTT: Last 8 steps
   - Depth curriculum: Epochs 1–5 use 4 loops, epochs 6–10 use 8 loops
   - Augmentation: 8 dihedrals × 10 color permutations = 80×

3. **Evaluation**:
   - AGI-1: Target 35–40% (baseline without TTT)
   - AGI-2: Measure gap (expect 5–10%)
   - Per-category breakdown

### Phase 2: Deep Supervision Ablations (Week 3)
**Goal**: Confirm deep supervision > recursion

1. **Ablations**:
   - No deep supervision (final loss only)
   - Uniform weighting (λ_t = 1/4)
   - Progressive targets (mask 80% → 0% over iterations)
   - Improvement loss (penalize no-change iterations)

2. **Metrics**:
   - Convergence speed (epochs to 30%)
   - Per-iteration accuracy (is step 2 useful?)
   - Spectral radius of Jacobians (stability check)

### Phase 3: TTT Integration (Week 4)
**Goal**: +20% from test-time adaptation

1. **Implementation**:
   - D_TTT: 100 examples (dihedrals × L1O)
   - 20 AdamW steps, LR=1e-4
   - Update: Last layer only (memory efficient)

2. **Ablations**:
   - 10 vs. 20 vs. 50 steps
   - All params vs. last layer vs. LoRA
   - With vs. without ensembling (10 runs)

3. **Target**: 
   - AGI-1: 50–55%
   - AGI-2: 15–20%

### Phase 4: Verification Loop (Week 5–6)
**Goal**: Convert near-misses via symbolic checking

1. **Implementation**:
   - ARC-DSL verifier (exact grid match)
   - 3 retry attempts on failure
   - Feedback: Error mask as input augmentation

2. **Metrics**:
   - Near-miss recovery rate (80–99% match → 100%)
   - Compute overhead (should be <3×)

3. **Target**:
   - AGI-1: 55–60%
   - AGI-2: 20–25%

### Phase 5: Category-Specific Training (Week 7–8)
**Goal**: Close the gap on compositional tasks

1. **Curriculum by category**:
   - Week 7: Heavy sampling of Multi-Rule Compositional
   - Add `compose_chain`, `interact_rule` to DSL
   - Train with depth 12–16

2. **Error analysis**:
   - Which categories still fail?
   - Is it capacity (need more params) or architecture (need different inductive bias)?

3. **Final Target**:
   - AGI-1: 60%+ 
   - AGI-2: 25–30% (competitive with o3 standard compute)

---

# Additional Queries (If Needed)

If we hit blockers, I'd want:

> **Query 16:** "What are the exact hyperparameters and training details for the TRM (7M params) that achieved 44.6% on ARC-AGI-1? Looking for learning rate schedules, batch sizes, training duration, and any tricks not mentioned in the paper."

> **Query 17:** "How do researchers handle the output prediction for ARC grids—direct pixel prediction vs. program generation vs. iterative refinement? Looking for decoder architectures for grid-to-grid tasks."

But I think we have enough to start implementing. The research is solid; now it's execution time.

---

**Bottom line**: We have a clear 8-week plan with specific targets. The key innovations are:
1. Deep supervision with progressive targets
2. TTT with L1O augmentation
3. Verification loop with DSL
4. Category-aware curriculum

Ready to start Phase 0 (infrastructure) whenever you are. Should I draft the initial codebase structure?