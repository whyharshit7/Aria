# ARIA: Adaptive Recurrent Integrative Architecture
### A Unified Predictive Framework for Perception, Memory, and Reasoning

> **Status:** Theoretical proposal — no experiments conducted yet.  
> Feedback, discussion, and collaboration welcome via Issues or Pull Requests.

---

## Overview

Contemporary deep learning architectures treat **perception**, **memory**, and **reasoning** as separate problems — encoders handle perception, context windows approximate memory, and reasoning is bolted on through chain-of-thought prompting.

**ARIA** argues these three faculties are not separate problems. They are manifestations of a single underlying process: *hierarchical predictive state estimation over multiple timescales*. This is what the brain does, and this paper sketches what an artificial system built on the same principle might look like.

---

## Core Ideas

ARIA is built on three axioms:

1. **Prediction is the universal currency.** The system actively predicts its inputs at every level of a hierarchy. Only prediction errors — the surprises — drive further computation. Easy inputs are cheap; hard inputs trigger deep processing automatically.

2. **Memory is a state, not a database.** Memory is integral to computation, not an external module. It manifests in three forms:
   - **Episodic memory** — fast, one-shot storage of specific experiences
   - **Working memory** — $K$ register slots for active reasoning (think: internal scratchpad)
   - **Semantic memory** — the network weights themselves, updated slowly via consolidation

3. **Reasoning is variable-depth recursion.** Computation depth adapts to problem difficulty. Reasoning is what happens when the recurrent loop runs without new external input, using working memory to hold intermediate results.

---

## Architecture

```
        ┌─────────────────────────────────────┐
        │   Level 3  (τ₃ = 64 steps)         │  Goals, schemas, world models
        └────────────┬──────────▲─────────────┘
               pred  │          │  err
        ┌────────────▼──────────┴─────────────┐
        │   Level 2  (τ₂ = 8 steps)          │  Events, narratives
        └────────────┬──────────▲─────────────┘
               pred  │          │  err
        ┌────────────▼──────────┴─────────────┐
        │   Level 1  (τ₁ = 2 steps)          │  Objects, concepts, entities
        └────────────┬──────────▲─────────────┘
               pred  │          │  err
        ┌────────────▼──────────┴─────────────┐
        │   Level 0  (τ₀ = 1 step)           │  Tokens, pixels, audio frames
        └─────────────────────────────────────┘
                          │
                  ┌───────▼────────┐
                  │ Raw Input Stream│
                  └────────────────┘

  ╔══════════════════════════════════════════════════╗
  ║         Memory Substrate (all levels access)    ║
  ║  ┌───────────┐  ──►  ┌─────────┐  ──►  ┌─────┐ ║
  ║  │ Episodic  │consol.│ Working │consol.│Seman│ ║
  ║  └───────────┘       └─────────┘       └─────┘ ║
  ╚══════════════════════════════════════════════════╝
```

Each level is a **Predictive State Column (PSC)** — a recurrent unit that:
- Receives bottom-up prediction errors from the level below
- Receives top-down predictions from the level above
- Reads from and writes to memory
- Outputs precision weights that gate how much each signal influences its state update

The clock periods follow a geometric progression ($\tau_l = \{1, 2, 8, 64\}$), forcing higher levels to represent slower, more abstract structure — analogous to intrinsic timescale gradients observed across the mammalian cortex.

---

## Key Properties

| Property | How ARIA achieves it |
|---|---|
| Adaptive computation | Prediction errors are the native signal — no separate halting mechanism needed |
| Continuous learning | Consolidation cycle migrates episodic memories into weights (no full retraining) |
| Uncertainty estimation | Precision weights natively encode confidence per signal source |
| Long-range memory | Tripartite memory substrate with importance-weighted retention |
| Compositional reasoning | Working memory slots for explicit variable binding across iterations |
| Biological plausibility | Grounded in predictive coding theory and complementary learning systems |

---

## Mathematical Foundation

The training objective is derived from **variational free energy minimization** (Friston, 2005, 2010):

$$\mathcal{L} = \underbrace{\sum_{l} \lambda_l \|\varepsilon_l\|^2_{\Pi_l}}_{\text{prediction error}} + \underbrace{\beta \cdot D_{\text{KL}}[q(s) \| p(s)]}_{\text{complexity}} + \underbrace{\gamma \cdot \mathcal{L}_{\text{task}}}_{\text{task loss}} + \underbrace{\delta \cdot \mathcal{L}_{\text{compute}}}_{\text{compute cost}}$$

State dynamics are derived as gradient descent on free energy, yielding exactly the predictive coding update rule at each level.

---

## Implementation Roadmap

The paper proposes a four-phase development plan:

| Phase | Goal | Key milestone |
|---|---|---|
| **1** | Language proof-of-concept (~350M params) | Perplexity within 10% of same-size Transformer |
| **2** | Continuous learning | ≥50% reduction in catastrophic forgetting vs. naive fine-tuning |
| **3** | Multimodal extension | Cross-modal prediction and binding via shared higher levels |
| **4** | Embodiment & active inference | Action selection to minimize expected future free energy |

---

## Theoretical Results

- **Adaptive Sparsity (Theorem 5.1):** Expected compute savings of ~72% for natural language under typical predictability values
- **Computational Universality (Theorem 5.2):** ARIA with $K \geq 2$ working memory slots and unbounded iterations is Turing-complete
- **Temporal Abstraction (Proposition 3.1):** Clock-rate hierarchy provably forces each level to represent structure at its own timescale
- **Optimal Precision (Proposition 4.1):** Precision weights converge to inverse prediction error variance — native uncertainty estimation

---

## Comparison with Existing Architectures

| Feature | Transformer | Mamba / SSM | **ARIA** |
|---|---|---|---|
| Memory | Context window | Compressed recurrent state | Episodic + Working + Semantic |
| Computation per input | Fixed | Fixed | Adaptive |
| Continual learning | ✗ | ✗ | ✓ (consolidation) |
| Uncertainty estimation | Requires calibration | ✗ | ✓ (native precision) |
| Reasoning mechanism | Chain-of-thought | Not explicit | Adaptive recurrence + WM |
| Biological plausibility | Low | Medium | High |

---

## Limitations

ARIA is a theoretical proposal and faces real challenges before empirical validation:

- **Training stability** — complex computational graph with recurrent dynamics and memory ops
- **Credit assignment** — truncated BPTT bias, near-discrete memory writes
- **Scaling behavior** — multiple interacting dimensions with unknown composition
- **Engineering complexity** — inherently more sequential than Transformers
- **Sufficiency of prediction error** — biological systems also use reward, emotion, neuromodulation

The paper discusses each of these in detail with proposed mitigations.

---

## Repository Structure

```
.
├── aria_paper.tex        # Full LaTeX source (arXiv-ready)
├── aria_references.bib   # BibTeX references (~50 citations)
└── README.md             # This file
```

### Compiling the Paper

**Overleaf (easiest):** Upload both `.tex` and `.bib` files to a new Overleaf project and hit Recompile.

**Local (requires a LaTeX distribution):**
```bash
pdflatex aria_paper.tex
bibtex aria_paper
pdflatex aria_paper.tex
pdflatex aria_paper.tex
```

---

## Citation

If you find this work useful for your research, please cite:

```bibtex
@article{aria2025,
  title   = {{ARIA}: Adaptive Recurrent Integrative Architecture --- 
             A Unified Predictive Framework for Perception, Memory, and Reasoning},
  author  = {[Author Names]},
  journal = {arXiv preprint},
  year    = {2025}
}
```

---

## Related Work

This paper draws on and synthesizes ideas from:
- **Predictive coding:** Rao & Ballard (1999), Friston (2005, 2010)
- **Hierarchical timescales:** Murray et al. (2014), Hasson et al. (2015)
- **Complementary learning systems:** McClelland et al. (1995), Kumaran et al. (2016)
- **Adaptive computation:** Graves (2016), Banino et al. (2021)
- **State space models:** Gu & Dao (2023), Peng et al. (2023)
- **Memory-augmented networks:** Graves et al. (2014, 2016), Lewis et al. (2020)

Full references are in `aria_references.bib`.

---

## Contributing

This is a theoretical paper and an open invitation for discussion. Feel free to:

- Open an **Issue** to discuss a theoretical claim, point out a gap, or suggest an experiment
- Open a **Pull Request** to fix a typo, improve a proof sketch, or add a related reference
- **Fork** the repo if you want to build on these ideas experimentally

---

## License

This work is released under the [MIT License](LICENSE).
