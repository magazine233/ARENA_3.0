# TARA Project Workspace

Personal workspace for the TARA program. Focus: learn by doing, orient everything toward the capstone.

## Program Schedule

| Phase | Weeks | ARENA Source | Topics |
|-------|-------|-------------|--------|
| Foundations | 1–3 | Chapter 0 | PyTorch, CNNs, ResNets, optimization, backprop |
| Transformers & Interp | 4–6 | Chapter 1 | Transformer from scratch, mech interp, SAEs |
| Applied | 7–11 | Chapters 2–4 | RL, DQN, PPO, RLHF, LLM evals, alignment |
| **Capstone Project** | **12–14** | **Original research** | **AI safety research** |

## Directory Structure

```
project/
├── capstone/       # Final 3-week research project
│   ├── src/        # Source code
│   ├── notebooks/  # Experiment notebooks
│   └── results/    # Figures, data, outputs
└── notes/          # Ideas, session takeaways, HatCat connections
```

## ARENA Chapter Reference

Exercises live in the parent repo:

- **Weeks 1–3** → `../chapter0_fundamentals/exercises/`
- **Weeks 4–6** → `../chapter1_transformer_interp/exercises/`
- **Weeks 7–8** → `../chapter2_rl/exercises/`
- **Weeks 9–10** → `../chapter3_llm_evals/exercises/`
- **Week 11** → `../chapter4_alignment_science/exercises/`

## HatCat Connection

[HatCat](https://hatcat.io/) — interpretability/steering platform for LLMs.
Relevant overlaps with ARENA curriculum:
- **Concept detection via activations** ↔ ARENA Ch1 (mech interp, SAEs, linear probes)
- **Deception detection (thinking vs writing)** ↔ ARENA Ch4 (alignment science)
- **Model steering** ↔ ARENA Ch1 (function vectors, model steering)
