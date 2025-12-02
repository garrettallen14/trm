### ARC-AGI Overview
The Abstraction and Reasoning Corpus for Artificial General Intelligence (ARC-AGI) is a benchmark designed to test fluid intelligence through novel visual reasoning tasks involving grid-based patterns. Each task requires inferring transformation rules from a few input-output examples and applying them to a test grid. Success is measured by **task accuracy**: the percentage of tasks where the model's output grid matches the ground truth exactly (all cells correct). Humans typically achieve 80-98% on public sets, but AI has lagged due to challenges in generalization without prior knowledge.

ARC-AGI has two main versions as of December 2025:
- **ARC-AGI-1** (original, released 2019): 400 public training tasks; evaluation on public eval (400 tasks), semi-private eval (100 tasks), and private eval (100 tasks). Focuses on core abstraction but shows signs of saturation under heavy compute.
- **ARC-AGI-2** (released early 2025): Harder variant with expanded, curated tasks emphasizing multi-step reasoning and efficiency. Designed to resist brute-force scaling; same format but with higher fluid intelligence demands. Public eval: 400 tasks; semi-private: 100 tasks; private: 100 tasks. Pure LLMs score 0%, and it prioritizes cost-efficiency metrics alongside accuracy.

Scores are reported with pass@2 (up to 2 attempts per task) and often include cost per task for comparability to human solving (~$5-150/task). Leaderboards (e.g., ARC Prize) use semi-private/private evals to prevent overfitting.

### SOTA Task Accuracy on ARC-AGI-1
As of December 1, 2025, the state-of-the-art (SOTA) on ARC-AGI-1 private evaluation is **80.0%**, achieved by Anthropic's Claude Opus 4.5 (Thinking mode, 64k context) at $1.47 per task. This surpasses prior highs from test-time adaptation (TTA) and evolutionary methods.

| Model/System | Approach | Private Eval Accuracy | Cost/Task | Date Achieved | Notes |
|--------------|----------|-----------------------|-----------|---------------|-------|
| Claude Opus 4.5 (Anthropic) | Frontier LLM + TTA (recursive refinement) | 80.0% | $1.47 | Nov 2025 | New SOTA for released frontier models; excels in coding/reasoning integration. |
| Poetiq (Gemini-3-b variant) | Recursive self-improving meta-system | ~79.6% (public eval) | <$1 | Oct 2025 | Saturates public eval; LLM-agnostic, uses open-source for adaptation. |
| Evolutionary TTA (Jeremy Berman/Grok-4) | Natural language instruction evolution | 79.6% (semi-private) | $8.42 | Sep 2025 | 25x more efficient than o3; evolves English prompts via sub-agents. |
| TTT + 8B LLM (MIT/Cornell) | Test-time training on synthetic data | 53.0% (public eval) | N/A | Dec 2024 | 6x improvement over base fine-tuning; ensemble of induction/transduction. |
| GPT-4o + Program Synthesis (Redwood) | LLM-guided neurosymbolic search | ~50% (public eval) | N/A | Jun 2024 | Early hybrid; ~35% on private; highlights compute's role but restricted in prizes. |
| Fine-tuned LLM (e.g., GPT-4 variants) | Synthetic ARC data fine-tuning | ~10% | N/A | Pre-2024 | Poor without adaptation; LLMs alone <5%. |

**Key Insights**: Progress accelerated in 2024-2025 via TTA (adapting models at inference) and neurosymbolic hybrids, jumping from ~33% (pre-ARC Prize) to 80%. However, brute-force search could hit ~49% since 2020, so true SOTA emphasizes efficiency. Human baseline: ~85% (panel) at low cost.

### SOTA Task Accuracy on ARC-AGI-2
ARC-AGI-2 remains far harder for AI, with SOTA at **37.64%** on semi-private eval by Claude Opus 4.5 (Thinking, 64k) at $2.40/task. It nearly doubles prior commercial highs and tops Kaggle entries. Tasks demand deeper recursion, resisting the scaling that saturated ARC-AGI-1.

| Model/System | Approach | Semi-Private Eval Accuracy | Cost/Task | Date Achieved | Notes |
|--------------|----------|----------------------------|-----------|---------------|-------|
| Claude Opus 4.5 (Anthropic) | Frontier LLM + TTA (recursive refinement) | 37.64% | $2.40 | Nov 2025 | New SOTA; strong on multi-step grids; outperforms Gemini 3/Sonnet 4.5. |
| Evolutionary TTA (Jeremy Berman/Grok-4) | Natural language evolution | 29.4% | ~$10 | Sep 2025 | Previous best (25%); uses Grok-4 for instruction generation/testing. |
| Grok-4 (Thinking) (xAI) | Optimized reasoning mode | 15.9% | N/A | Nov 2025 | Doubles prior commercial SOTA; tops Kaggle at the time. |
| GPT-5.1 AR (OpenAI) | Advanced reasoning variant | ~18.3% (private) | N/A | Nov 2025 | High-compute; 6x cheaper variant at 17.6%. |
| MindoAI Agent | On-chain data-trained agent | ~16% (est.) | <$1 (consumer HW) | Dec 2025 | Efficient/decentralized; uses unique transaction patterns for generalization. |
| Poetiq (GPT-5.1 variant) | Recursive meta-system | ~25-30% (est.) | Low | Oct 2025 | Redraws cost-performance frontier; unadapted to v2 tasks. |
| Frontier LLMs (e.g., o3-preview) | Base reasoning systems | <10% (single-digit) | High | Early 2025 | 0% for pure LLMs; brute-force limited by efficiency focus. |

**Key Insights**: ARC-AGI-2 stresses efficiency (e.g., log-linear scaling insufficient), with humans at ~100% (smart panels) in ≤2 attempts. SOTA relies on recursive TTA over raw scale; ARC Prize 2025 ($725K+) drives open progress. No system hits the 85% AGI target.

### SOTA for TRMs (Tiny Recursive Models)
TRMs refer to **Tiny Recursive Models**, a paradigm introduced by Samsung AI in late 2024/early 2025. These are ultra-small neural nets (~7M parameters, 10,000x smaller than LLMs like Gemini 2.5 Pro) using recursive loops for iterative refinement (draft → critique → refine). They excel on structured reasoning without pretraining knowledge or DSL priors, making them generalizable to discrete problems (e.g., grids, mazes). TRMs are open-source (MIT license), low-power, and run locally—ideal for edge devices like robots.

On ARC-AGI:
- **ARC-AGI-1**: SOTA ~45% (public eval) for zero-knowledge TRMs; outperforms larger models (e.g., o3, Gemini) via recursion on puzzles like Sudoku/ARC grids. Overall SOTA is higher (80%), but TRMs win on efficiency/no priors.
- **ARC-AGI-2**: ~25-30% (est. semi-private); beats expensive LLMs on cost-adjusted scores but trails frontier TTA hybrids. Public SOTA for pure TRMs.

| Aspect | Details | ARC-AGI-1 Performance | ARC-AGI-2 Performance | Advantages |
|--------|---------|-----------------------|-----------------------|------------|
| **Size/Compute** | 7M params; trains for $0.20-0.30 | 45% (zero-knowledge SOTA) | ~25-30% | 10,000x smaller; local inference. |
| **Mechanism** | Recursive: Iterates drafts/critiques | Outperforms o3/Gemini on grids | Strong on multi-step but compute-limited | No external priors; general to any discrete task. |
| **Limitations** | Struggles on unstructured/open-ended | Below 80% overall SOTA | Trails Opus 4.5 (37%) | Scaling recursion needs more flops for v2 depth. |
| **Applications** | Edge AI (e.g., Optimus robots) | N/A | N/A | Ultra-low power; inspires "cognitive core" stripping in LLMs. |

**Key Insights**: TRMs highlight architectural innovation over scale—proving small models can reason recursively without "encyclopedic bloat." François Chollet (ARC creator) praised them as public SOTA for zero-knowledge approaches, applicable beyond ARC (e.g., any pattern task). They embody the "ghost-like" intelligence shift from LLMs, blending with RL for agents.

For deeper dives, check the [ARC Prize Leaderboard](https://arcprize.org/leaderboard) or François Chollet's [technical report](https://arcprize.org/media/arc-prize-2024-technical-report.pdf). Progress is rapid—expect updates from ARC Prize 2025!