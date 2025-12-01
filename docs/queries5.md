### Hyperparameters and Training Details for the Tiny Recursive Model (TRM, 7M Parameters)

The Tiny Recursive Model (TRM) achieving 44.6% accuracy on the ARC-AGI-1 public evaluation set (with self-attention variant) is detailed in the October 2025 paper "Less is More: Recursive Reasoning with Tiny Networks" by Alexia Jolicoeur-Martineau (arXiv:2510.04871). This 2-layer recursive architecture uses ~7M parameters in the trunk (core network), plus task-specific embeddings (~500M+ total effective params across ~1,000 augmented tasks, but the trunk is the focus). It was trained from scratch on augmented ARC-AGI-1 data, emphasizing recursion for refinement (up to 16 steps) without pretraining on external corpora.

The paper's main body provides high-level architecture (e.g., alternating latent think-act cycles: \( z \leftarrow f(x, y, z) \), \( y \leftarrow g(y, z) \)) and results, but defers full hyperparameters to Appendix B ("Hyper-parameters and setup"). Based on the paper, official GitHub repo (SamsungSAILMontreal/TinyRecursiveModels), and community reproductions/discussions (e.g., from Trelis Research and alphaXiv), below is a comprehensive breakdown. "Tricks not in the paper" are drawn from repo configs, ablation logs, and X threads by the author (@jm_alexia) and replicators, including unpublished sensitivities like embedding LR scaling.

#### Architecture Recap (for Context)
- **Trunk**: 2-layer transformer (self-attention or MLP variant; attention for ARC-AGI grids up to 30×30).
- **Embedding Dim**: 512 (puzzle_embed_ndim=512; grids tokenized cell-wise with 2D RoPE).
- **Recursion**: Up to K=16 steps; deep supervision on intermediates (uniform weighting, progressive targets via no-grad loops).
- **Data**: ~800 ARC-AGI-1 training tasks × 1,000 augmentations (dihedral symmetries, color perms, translations) = ~800k effective samples. Includes eval pairs for non-competition training.

#### Training Hyperparameters
| Hyperparameter | Value/Details | Notes/Source |
|----------------|---------------|-------------|
| **Optimizer** | AdamW (β1=0.9, β2=0.999, ε=1e-8, weight decay=0.01) | Standard for stability in small models; repo default. |
| **Learning Rate (LR) Schedule** | Linear warmup (10% of steps) from 0 to peak, then linear decay to 0. Cosine annealing optional in ablations. Peak LR: 1e-4 (trunk), 1e-2 (embeddings). | Differential LR prevents embedding dominance (500M+ params). Not in main paper; from repo `train.py` and Trelis ablation [post:17]. Warmup essential for recursion gradient flow. |
| **Batch Size** | 64–128 (effective; micro-batches of 8–16 due to recursion unrolling). | Balances memory (recursion ×16 doubles effective batch); repo uses 128 on 4×H100 for ARC. |
| **Training Duration** | 100–200 epochs (~48 hours on 4×H100 GPUs/A100s); ~1M–2M total steps. Early stopping at val perplexity plateau (~epoch 80 for 44.6%). | ~2 days total; scales with augs. Paper mentions "small data" (~1k base tasks); repo logs show convergence at 100 epochs for ARC-AGI-1. |
| **Gradient Accumulation Steps** | 4–8 | For effective batch=512+; handles recursion memory (full BPTT through 16 steps). |
| **Gradient Clipping** | Global norm=1.0 | Prevents explosion in loops; uniform across trunk/embeddings. |
| **Recursion Depth (K)** | 16 (train); dynamic halting via BCE loss (no extra head). | Deep sup. on all steps; 3.75× avg. compute multiplier at inference. |
| **Loss** | Cross-entropy (grid prediction); deep sup. uniform (λ_t=1/K) + auxiliary consistency (KL on latents). | Progressive: No-grad loops (6×) per grad step for refinement. |

#### Data and Augmentation Details
- **Dataset**: ARC-AGI-1 training (~800 tasks) + eval pairs (non-competition). Grids flattened to sequences (max 900 tokens/cell-based).
- **Augmentations**: 1,000× per task (8 dihedrals × color shuffles × translations/scales). Synthetic via RE-ARC DSL for primitives.
- **Preprocessing**: Cell tokenization + 2D RoPE; special tokens for input/output separation.

#### Evaluation Setup for 44.6%
- **Inference**: 1,000× ensemble (augmentations) + recursion (avg. 3.75 steps) + pass@2 voting. ~480k forward passes/task ($1.76/task on H100).
- **Metric**: Exact grid match on public eval; two-try (resample on failure).

#### Tricks Not Mentioned in the Paper
The paper focuses on architecture/results, but repo code, author's X threads (@jm_alexia), and replicators (e.g., Trelis [post:11], alphaXiv [post:10]) reveal implementation tweaks for stability/generalization:
- **Differential LR for Embeddings**: Trunk LR=1e-4, embeddings=1e-2 (10× higher) to counter their param dominance (~98% total). Prevents "embedding memorization" on augs; boosts +5–7% OOD (Trelis ablation [post:17]).
- **No-Grad Refinement Loops**: 6× no-grad iterations per grad step during training—simulates test-time compute cheaply. Emergent "orbits" in latents; not emphasized in paper but key for 44.6% (repo `recursion.py`).
- **Truncated BPTT Hybrid**: Full BPTT for short loops (<8 steps), truncated (last 4) for longer—saves 50% memory without loss. From early repo commits; stabilizes small-model training.
- **Entropy Regularization**: Auxiliary loss (0.01 weight) on attention entropy to prevent mode collapse in recursion; +3% on Maze-Hard subsets (author's ablation logs).
- **Seed Sensitivity**: Fixed seed=42 for augs; random seeds drop 2–4% reproducibility. Replicators note cosine annealing (vs. paper's linear) yields +1% at iso-steps [post:17].
- **Hardware-Aware Batching**: Micro-batches on recursion unrolling; uses bfloat16 + FlashAttention for 2× speedup on H100s. Not in paper, but repo README stresses for <48h runs.
- **Overfitting Check**: Monitor per-task perplexity (not aggregate); stop if >5% tasks overfit augs. Trelis found doubling hidden_dim (to 1,024) hurts iso-compute [post:11].

These details enable replication: Clone the repo, run `train_arc.py --hidden_dim=512 --lr_trunk=1e-4 --lr_embed=1e-2 --epochs=100`. Community notes (e.g., [post:15]) confirm 44.6% with these. For ARC-AGI-2 (7.8%), add eval data to pretraining. TRM's efficiency (0.01% LLM params) highlights recursion over scale, but test-time compute (~3.75×) is key.

### Handling Output Prediction for ARC Grids in Research

Researchers tackling ARC-AGI (Abstraction and Reasoning Corpus) grid-to-grid prediction—where models infer transformation rules from few-shot input-output grid pairs and apply them to a test input—employ three primary strategies: **direct pixel prediction** (end-to-end visual mapping), **program generation** (symbolic DSL or code synthesis), and **iterative refinement** (recurrent updates for progressive improvement). These are often hybridized in neurosymbolic or test-time adaptive setups, balancing efficiency, interpretability, and generalization on the benchmark's discrete, low-resolution grids (1–30×30 cells, 10 colors).

Direct pixel prediction treats ARC as image-to-image translation, excelling in perceptual tasks but struggling with rule abstraction (e.g., ~20–30% accuracy without adaptation). Program generation leverages symbolic execution for verifiability, achieving 40–50% on public evals but facing combinatorial explosion. Iterative refinement, via loops or diffusion, enables "thinking harder" with test-time compute, boosting scores to 50–70% in hybrids but risking drift. Decoder architectures are typically lightweight (e.g., transformer decoders or CNN heads) to handle variable grid sizes, often with 2D positional encodings (e.g., RoPE) and tokenization (cell-based flattening).

Below, I compare the approaches, then detail decoder architectures. Insights draw from 2024–2025 advances, where hybrids dominate SOTA (e.g., 71.6% public eval via LLM-guided refinement).

#### Comparison of Prediction Strategies
| Strategy | Description | Pros | Cons | ARC Performance Examples | Sources |
|----------|-------------|------|------|--------------------------|---------|
| **Direct Pixel Prediction** | Model regresses output grid pixels directly (e.g., per-cell color logits) conditioned on input/demos. Often autoregressive or via conv layers. | Fast inference; captures visual patterns (e.g., textures). No discrete search. | Poor on novel rules (curve-fitting fails OOD); hallucinations in sparse grids. | 26–41% on public eval (e.g., ViT with TTT); LLMs like GPT-4o at ~20% zero-shot. | , ,  |
| **Program Generation** | Neural model outputs executable code/DSL sequence (e.g., `rotate90 + fill`); verifier simulates on grids. | Interpretable; generalizes via composition. Handles novelty via search. | Search explosion (10^6+ candidates); needs strong neural prior. | 43–53% with LLM sampling (e.g., GPT-4o k=2048); 18% brute-force. | , ,  |
| **Iterative Refinement** | Starts with initial prediction; loops to refine (e.g., latent updates or diffusion denoising). | Adaptive depth; self-correction (+10–20% gains). Efficient for small models. | Drift/overfitting in loops; compute-heavy (e.g., 16 steps). | 50–71.6% hybrids (e.g., HRM outer loop); +15% via revisions. | , ,  |

Hybrids (e.g., dual-headed CNN: pixel + program) combine strengths, training jointly on losses for shared features. For ARC-AGI-2 (compositional focus), refinement hybrids shine (+20% vs. direct).

#### Decoder Architectures for Grid-to-Grid Tasks
Decoders map latent representations (e.g., task embeddings) to output grids, handling variability via padding or relative encodings. Common backbones: Transformer decoders (autoregressive over flattened cells) or CNNs (for spatial invariance). Tokenization: Cell-based (1×1 patches, ~900 tokens max) with vocab=11 (10 colors + empty).

| Architecture | Key Components | Prediction Mode | Handling Variable Sizes | Examples/Results | Sources |
|--------------|----------------|-----------------|--------------------------|------------------|---------|
| **Transformer Decoder (Autoregressive)** | Single/multi-layer decoder (e.g., 1 layer, 512 dim, 4 heads); cross-attends to encoder (input/demos). Outputs token seq (row-by-row cells). | Direct or program (e.g., DSL tokens); probabilistic for refinement. | Padding to 30×30; 2D RoPE/ALiBi for relativity. | GridCoder: 452M params, 30–40% on easy subsets; faster convergence than T5. NPS: Encoder-decoder TF, 40–50% with execution guidance. | ,  |
| **CNN-Based Decoder** | Shared CNN backbone (e.g., dilated convs) + linear head for pixel logits; dual-head for pixel/program. | Direct pixel (channel-wise regression); refinement via iterative denoising. | Adaptive pooling/aggregators for fixed-size features; maximal padding for 30×30. | Dual-Headed CNN: Joint pixel + DSL loss, ~40% on training pairs. EngramNCA: Hidden dim=128, iterative refinement for partial solves (e.g., 80% cells correct). | ,  |
| **VQ-VAE/Probabilistic Decoder** | VQ encoder compresses grids to tokens; decoder (e.g., transformer) reconstructs from latents. | Direct or latent program (vector to pixels); logsumexp for multi-slice probs. | Token seq length scales with grid; coefficient annealing (0.1→1) prevents collapse. | Latent Program Decoder: Encodes to vector, gradients for ascent, decodes to pixels; 50% public eval. | ,  |
| **Diffusion/Refinement Decoder** | Diffusion head (e.g., U-Net-like) denoises noisy grids iteratively; or recurrent loop around transformer. | Iterative: Predict noise/deltas per step; fixed steps (no halting). | Grid-specific (e.g., maximal padding); aggregator for channels. | DIS: Discrete diffusion targets, +10% on ARC-AGI-1/2; avoids no-grad cycles. HRM Outer Loop: 27M params, recurrent refinement to 45%. | , ,  |

**Implementation Notes**:
- **Training**: Joint losses (e.g., CE for pixels + seq loss for programs); augmentations (dihedrals) densify data. Test-time: TTT (10–50 GD steps, LR=1e-4) adapts decoders.
- **Efficiency**: Small decoders (<500M params) dominate; e.g., 1-layer TF converges faster than T5 on synthetics.
- **Trends**: 2025 shifts to probabilistic decoders for latent programs (+15% OOD); hybrids with DFS search (e.g., 71.6% SOTA open-source) refine via LLM feedback.

These methods evolve toward verifiable, adaptive decoders, with refinement hybrids leading for ARC's novelty demands.