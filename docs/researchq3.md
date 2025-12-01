# Deep Reflection on TRM Research: Round 2

Let me work through these four new queries with the same depth, then synthesize with the earlier findings.

---

## Query 9: DSLs for Procedural ARC Data Generation

### What I Learned

Three main DSL families emerge:

1. **ARC-DSL (Hodel)**: ~160 primitives, functional composition, Python-embedded
2. **RE-ARC**: Task-specific generators for all 400 training tasks, reverse-engineered distributions
3. **Icecuber**: Brute-force enumeration of short programs (≤4 ops), type-constrained search

The key insight: **DSLs don't just augment data—they define the hypothesis space**. If your DSL has `rotate90` but not `rotate_arbitrary_angle`, your model can only learn discrete rotations. The primitives you choose implicitly bound what's learnable.

The **RE-ARC approach** is clever: Instead of one universal DSL, create 400 mini-generators that each capture one task's "essence." This ensures synthetic data matches the *distribution* of each task, not just random compositions.

### What Intrigues Me

The **NeoGen/HVM type-first approach** is philosophically different: Search the space of *types* before *programs*. For ARC, this means asking "what shape does the output have?" before "what operations produce it?" This could dramatically prune the search space.

```
type State = Vec<4> (Col, Col)
program = \state -> draw_line_if_eq(state)
```

This feels closer to how humans solve ARC—we first recognize the output structure ("it's a 4×4 grid with colored pairs") then figure out the transformation.

The **100k–1M samples** scale is achievable on consumer hardware. RE-ARC generating "simpler/harder than originals" via param randomization is exactly what you need for curriculum learning.

### What I'm Skeptical About

- **"~160 primitives covers ARC"** feels like a post-hoc claim. The primitives were likely designed *after* seeing the tasks. True generalization would require primitives that cover *unseen* tasks.
- **Verification via symbolic execution** only checks consistency, not correctness. A DSL program can perfectly match I/O pairs but implement the wrong rule (e.g., memorizing specific colors instead of learning "swap colors").
- **BARC's GPT-4o augmentation** is a cheat—you're using a 1T+ param model to generate data for a 7M param model. The "tiny model" success is partly inherited from GPT-4o's priors.

### Underappreciated Insight

The **6 tasks requiring adaptations** in RE-ARC are gold for error analysis. These are the tasks where the "obvious" DSL decomposition fails—studying them could reveal what primitives are missing or what compositional patterns break.

The **verifier as a training signal** is underexplored. Instead of just filtering bad generations, you could use verification failure as a *negative example* for contrastive learning. "This program is wrong because it fails on example 3" teaches more than just "here's a correct program."

---

## Query 10: Test-Time Training (TTT) Implementation

### What I Learned

TTT is more structured than I expected. The pipeline:

1. **Pre-fine-tune** on 500k synthetic tasks (baseline priors)
2. **Construct D_TTT**: Augment 2–3 demos → 100–250 examples via dihedrals + L1O
3. **Optimize**: 10–50 AdamW steps at LR=1e-4, update LoRA adapters (0.1% params)
4. **Ensemble**: 2–10 adapted variants via voting

Key hyperparameters:
| Parameter | Value |
|-----------|-------|
| Learning rate | 1e-4 (global), 1e-3 (LoRA only) |
| Gradient steps | 10–50 |
| Batch size | 4–16 |
| D_TTT size | 100–250 augmented examples |

The **L1O (leave-one-out)** strategy is clever: Hold out one demo for validation, train on the rest. This gives you an in-task signal for early stopping.

### What Intrigues Me

**Muon normalization** (SVD-based LR scaling) making LR "relative to token importance" is intriguing. This could prevent the model from over-updating on common tokens while under-updating on rare but important ones.

The **+6× accuracy gain** over frozen baselines is massive. For our tiny model, this suggests TTT should be mandatory, not optional. The architecture matters less than the adaptation mechanism.

**Per-instance LoRA** (separate adapters per task) vs. **shared LoRA** (one across tasks): The 24% drop from shared adapters confirms that TTT is learning task-specific features, not just general "ARC-ness."

### What I'm Skeptical About

- **250 augmented examples from 2–3 demos** is a lot of augmentation. At some point, you're fitting noise in the augmentation process, not the underlying rule. The 8 dihedral symmetries × color permutations × L1O can only stretch so far.
- **10–50 gradient steps** with LR=1e-4 on 0.1% of params—this is barely moving the weights. I wonder if the gains come from the *inference-time ensembling* rather than the actual weight updates.
- **"No access to test label"** is technically true but philosophically murky. You're using the test *input* structure to guide adaptation. If the test input has a distinctive pattern (e.g., larger grid), you're implicitly leaking information.

### Underappreciated Insight

**DPO (Direct Preference Optimization) with negative samples** for regularization is smart. Instead of just L2 decay, explicitly teach the model "this output is wrong because it's too complex." This could prevent the "mode collapse to simple patterns" failure mode.

The **1–5 inner loops in meta-TTT** suggests a meta-learning framing: Learn *how to adapt* during pre-training. This is MAML-style but for test-time. For tiny models, meta-TTT could be more effective than scaling params.

---

## Query 11: Hybrid Latent/Token Architectures with Routers

### What I Learned

Four concrete architectures:

| Model | Router Mechanism | Mode Switch |
|-------|-----------------|-------------|
| **COCONUT** | Gating MLP on hidden entropy | `<bot>` → latent, `<eot>` → token |
| **SwiReasoning** | Per-token entropy threshold | Low confidence → latent loop |
| **TaH** | "Hard token" detector (attention entropy + logit variance) | Selective latent on complex subsequences |
| **LatentMAS** | KV-cache alignment matrix | Latent for inter-agent, tokens for output |

The common thread: **Entropy/confidence as the switch signal**. High uncertainty → think more in latents. Low uncertainty → output tokens.

The efficiency gains are substantial:
- **56–92% token reduction**
- **2–7× speedup**
- **+4–10% accuracy** on reasoning benchmarks

### What Intrigues Me

**TaH's per-token routing** is the most granular: Not "latent mode vs. token mode" but "which specific tokens need more thinking?" This matches intuition—in "3 + 4 × 5", the "×" token needs more processing than the spaces.

The **"reasoning core" + "output decoder"** framing from surveys is clean. The transformer becomes:
1. **Reasoning core**: Recurrent latent loops, no token output
2. **Output decoder**: Lightweight autoregressive generation

This separation could simplify training—train the core on latent objectives (reconstruction, contrastive), train the decoder on generation.

**LatentMAS's KV-cache routing** for multi-agent is clever: Agents share thoughts via cache, not tokens. This is 70–84% fewer tokens with +2.8–4.6% accuracy. For our single-model case, this suggests the hidden states *are* the reasoning—tokens are just for output.

### What I'm Skeptical About

- **Curriculum-based switching** (gradually increase latent proportion) assumes you know which tasks need latents. For novel ARC tasks, you don't know in advance. The router must generalize.
- **"Emergent BFS in latents"** from COCONUT sounds magical. How do we verify the model is actually doing breadth-first search vs. just finding some other solution? Interpretability is still a gap.
- **<1% router overhead** seems too good. A "1–2 layer MLP on hidden states" for 0.6B+ models is tiny, but for our 7M target, even 70k params for a router is 1%. We might need even simpler routing (e.g., fixed schedule).

### Underappreciated Insight

**Auxiliary latent supervision** in TaH for stable training is key. Without it, the router learns to always pick one mode. Supervising the latent states directly (e.g., "after K loops, latent should be close to target embedding") prevents degenerate routing.

The **3–5 cycle cap** to "prevent overthinking" is practical. Infinite latent loops can diverge; hard-capping iterations trades optimality for stability. For tiny models, cap at 8–16 loops seems right based on earlier findings.

---

## Query 12: Failure Modes of Small Recursive Models

### What I Learned

This is the most important query for planning experiments. Key failure modes:

| Failure Type | Frequency | Root Cause |
|--------------|-----------|------------|
| **Partial rule capture** | 40–50% | Can't compose >2–3 ops |
| **Recursion drift** | 20–30% | Noise accumulation after 8–10 steps |
| **Augmentation overfitting** | 15–25% | Shortcuts on symmetries |
| **Boundary/edge errors** | 10–20% | Poor extrapolation of 2D positional encoding |
| **Premature halting** | 10–15% | Halt head games the reward |

Task category breakdown:

| Category | TRM Error Rate (AGI-1 / AGI-2) |
|----------|-------------------------------|
| Core primitives | 10–20% / 5–10% |
| Object manipulation | 30–40% / 40–50% |
| Conditional/compositional | 50–60% / 70–80% |
| Context-sensitive | 60–70% / 80–90% |
| Symbolic interpretation | 70–80% / 90+% |

The **bombshell finding**: "Deep supervision drives ~80% of gains, not loops alone—vanilla tiny ViTs match 30–40% with augs."

This means recursion is less magical than claimed. The gains come from:
1. Deep supervision (loss at each iteration)
2. Heavy augmentation
3. Test-time compute (ensembling)

The looping is just a vehicle for deep supervision, not inherently powerful.

### What Intrigues Me

The **"orbit collapse"** failure (repetitive latents) after 20+ steps is a concrete signal. We could detect this by monitoring latent cosine similarity across iterations—if it plateaus, halt.

**60% of TRM errors are "systematic near-misses"** where pass@2 recovers 20–30%. This means the model has the right idea but wrong execution. A simple retry mechanism (generate 2 outputs, pick the verified one) could be a cheap +20%.

The **errors clustering in corners/edges** points to positional encoding failures. 2D RoPE might not extrapolate well to grid boundaries. We could add **explicit boundary tokens** or **edge-aware attention biases**.

### What I'm Skeptical About

- **"2-layer nets can't encode novel symbols"** feels like a capacity argument, but TRM has 7M params distributed across layers. The bottleneck might be width (embedding dim) not depth. A wider 1-layer model might do better than a deeper narrow one.
- **"$2/task via 1k ensembles"** is expensive for "tiny." At scale (thousands of tasks), this adds up. The Pareto efficiency claim assumes you're okay with O(1000) inference passes per task.
- **"Replications fail ~20% due to inference mismatches"** is concerning. If the results aren't reproducible, how much should we trust the reported numbers?

### Underappreciated Insight

The **"compute illusion"** critique is spot-on: TRM is tiny in params but not in inference compute. A 7M model run 1000× with 16 recursion steps = 112B FLOPs per task. A 1B model run once = 2B FLOPs. The "tiny" model is actually more expensive!

**Curriculum on failure subsets** (train more on tasks where the model fails) could give +10–15%. This is standard hard example mining but underutilized in ARC research. After each eval, identify the failure clusters and generate more synthetic data for those categories.

---

# Synthesized Insights Across All Queries

## 1. The "Deep Supervision > Recursion" Revelation

Combining Query 12's finding ("deep supervision drives 80% of gains") with Query 2's halting analysis:

**Recursion is a delivery mechanism for deep supervision, not a reasoning primitive.**

Implication: We should focus on getting intermediate supervision signals right, not on fancy halting mechanisms. A simple fixed-loop model with loss at each iteration might match learned-halting models.

## 2. The TTT + DSL Synergy

Query 9 (DSLs) + Query 10 (TTT):

**Use DSL-generated synthetic data for pre-training, then TTT for task-specific adaptation.**

The DSL ensures coverage of primitive compositions. TTT handles novel combinations at test time. This two-stage approach could give the benefits of both without their individual weaknesses.

Concrete recipe:
1. Generate 500k samples via RE-ARC DSL
2. Pre-train looped ViT with deep supervision
3. At test time: 20 gradient steps on augmented demos, ensemble 10 runs

## 3. The Router Simplification

Query 11 shows entropy-based routing works across multiple architectures. For a 7M model, we don't need a learned router:

**Use fixed entropy thresholds for mode switching.**

After each loop iteration:
- Compute attention entropy
- If entropy < 0.3 (confident): exit to output
- If entropy > 0.8 (confused): continue looping
- Cap at 16 iterations regardless

This is cheaper than a learned router and might work just as well for tiny models.

## 4. The Failure Mode Taxonomy as Curriculum

Query 12's task categories map to a natural curriculum:

1. **Week 1–2**: Core primitives (rotations, mirrors) — establish baseline
2. **Week 3–4**: Object manipulation (extend lines, count) — add object detection
3. **Week 5–6**: Conditional/compositional — train on multi-op chains
4. **Week 7–8**: Context-sensitive + symbolic — hardest subset, heavy augmentation

Training on failure categories specifically could prevent the model from "coasting" on easy tasks.

## 5. The Verification Loop

Query 9's DSL verification + Query 12's "near-miss" finding:

**Add a symbolic verifier in the inference loop.**

If we have a DSL that can verify I/O consistency:
1. Model generates candidate output
2. Verifier checks against demos
3. If fail: feed error back, generate again
4. If pass: accept

This could convert the 60% "near-misses" into successes without changing the model.

---

# Revised Experimental Plan

Based on both rounds of research:

## Phase 1: Baseline (Week 1–2)
1. **Architecture**: 3-layer looped transformer, ~10M params
   - Cell-based tokenization, 2D RoPE
   - Pre-LN, spectral norm
   - **Fixed 8 iterations** (no learned halting yet)

2. **Training**:
   - Generate 200k samples via RE-ARC DSL
   - Deep supervision: Loss at iterations 2, 4, 6, 8
   - Depth curriculum: Start with 2 loops, increase to 8
   - 1000× augmentation (dihedrals + colors)

3. **Evaluation**:
   - Track per-category accuracy (core → symbolic)
   - Monitor attention entropy evolution
   - Verify reproducibility (3 seeds)

## Phase 2: TTT Integration (Week 3–4)
1. Implement **L1O TTT**:
   - D_TTT: 100 augmented examples from demos
   - 20 AdamW steps, LR=1e-4
   - Update: Only final layer (cheap LoRA alternative)

2. Add **ensemble voting**: 10 runs, majority vote

3. Measure: TTT gain vs. baseline, compute cost

## Phase 3: Verification Loop (Week 5–6)
1. Implement **DSL verifier** from ARC-DSL
2. Add **retry mechanism**: On verification failure, regenerate (max 3 tries)
3. Measure: Recovery rate on "near-miss" tasks

## Phase 4: Hybrid Routing (Week 7–8)
1. Add **entropy-based early exit**:
   - Threshold tuned on val set
   - Cap at 16 iterations

2. Compare: Fixed 8 vs. adaptive 4–16

3. **Error analysis**: Which failure modes persist?

---

# Additional Queries

I want more data on two specific gaps:

> **Query 13:** "What are the specific ARC-AGI-2 task categories and their primitive requirements? Looking for a breakdown of the 1,120 tasks by transformation type, compositional depth, and which primitives are needed."

> **Query 14:** "How do researchers implement deep supervision for looped/recursive transformers? Looking for loss weighting schemes (uniform vs. increasing), gradient flow through iterations, and whether intermediate targets differ from final targets."

> **Query 15:** "Are there examples of using symbolic verifiers in the neural inference loop for ARC-AGI? Looking for architectures that combine DSL execution with neural generation, and how verification failures are fed back."

These would help us:
- **Query 13**: Understand ARC-AGI-2 specifically (since that's the real target)
- **Query 14**: Get the deep supervision implementation right (since it's 80% of gains)
- **Query 15**: Design the verification loop properly

---

What do you think? Should we run these additional queries, or start implementing Phase 1? I'm particularly keen on **Query 14** since deep supervision seems to be the real driver of performance.