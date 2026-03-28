# Research Prompt: Investigating a TARA Capstone Project at the Intersection of Ontology Design and AI Interpretability

## Context

I am a participant in the TARA (Technical Alignment Research Accelerator) program — a 14-week AI safety training program based on the ARENA 3.0 curriculum. The program culminates in a 3-week capstone project (weeks 12–14) that should produce research-quality output.

My background is in **ontology and taxonomy design**, not ML research. I contributed to the HatCat project (https://hatcat.io/, https://github.com/p0ss/HatCat) — an open-source LLM interpretability/steering platform that uses learned classifiers on model activations to detect 8,000+ concepts in real-time. HatCat's concept taxonomy is built from WordNet/SUMO ontologies, letting the model define each concept in its own representational language. My contribution was in the ontology design layer.

I learn best by doing, not by studying theory first. I want a project I can build iteratively, fail fast on, and learn from — not one that requires deep mathematical prerequisites.

My goal is personal learning and contributing something genuinely useful, not career positioning as an AI researcher.

## The ARENA Curriculum Topics I'll Be Exposed To

The TARA program covers these ARENA 3.0 chapters over 11 weeks before the project phase:

- **Chapter 0 (Weeks 1–3)**: PyTorch fundamentals, CNNs, ResNets, optimization, backpropagation, VAEs/GANs
- **Chapter 1 (Weeks 4–6)**: Transformers from scratch, mechanistic interpretability with TransformerLens, sparse autoencoders (SAEs), linear probes, function vectors and model steering, activation analysis
- **Chapter 2 (Weeks 7–8)**: Reinforcement learning, DQN, PPO, RLHF
- **Chapter 3 (Weeks 9–10)**: LLM evaluations, dataset generation, eval frameworks (Inspect), LLM agents
- **Chapter 4 (Week 11)**: Emergent misalignment, alignment science, persona vectors

## Research Question to Investigate

**How can formal ontology design improve the interpretability of LLM internal representations — specifically, can structured concept hierarchies (taxonomies) make sparse autoencoder features or activation-level concept detection more systematic, more complete, and more useful for AI safety?**

This sits at the intersection of:
1. My expertise (ontology/taxonomy design)
2. The ARENA curriculum (mech interp, SAEs, model steering, evals)
3. HatCat's existing approach (WordNet/SUMO-derived concept classifiers on activations)
4. An active and under-explored research gap

## Key Research to Draw Upon

### Ontology × Interpretability (the core gap)

- **"The Geometry of Concepts: Sparse Autoencoder Feature Structure"** (2025) — SAE feature dictionaries reveal geometric structure (parallelograms, crystals) in concept space. Do these structures align with ontological hierarchies?
  - Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12025678/

- **"Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures"** (OpenReview, Oct 2025) — Modified SAE architecture that explicitly models semantic hierarchy of concepts. Shows hierarchy improves both reconstruction and interpretability.
  - Source: https://openreview.net/forum?id=C7M6F0OJ1l

- **"The Artificial Intelligence Ontology: LLM-Assisted Construction of AI Concept Hierarchies"** (Joachimiak et al., 2024) — Formal ontology construction for AI concepts using LLM assistance.
  - Source: https://journals.sagepub.com/doi/10.1177/15705838241304103

- **"Probing the Representational Power of Sparse Autoencoders in Vision Models"** (ICCV Workshop 2025) — Uses WordNet ontology hierarchies to measure "Ontological Coverage" of SAE features.
  - Source: https://openaccess.thecvf.com/content/ICCV2025W/Findings/papers/Olson_Probing_the_Representational_Power_of_Sparse_Autoencoders_in_Vision_Models_ICCVW_2025_paper.pdf

- **"A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models"** (EMNLP 2025) — Comprehensive survey covering SAE architectures, training, feature explanation, and evaluation.
  - Source: https://arxiv.org/abs/2503.05613

### AI Safety Applications

- **Safety Concept Activation Vectors (SCAV)** — Framework for linearly separating safe/malicious concepts in LLM activation space.
  - Source: https://proceedings.neurips.cc/paper_files/paper/2024/file/d3a230d716e65afab578a8eb31a8d25f-Paper-Conference.pdf

- **Representation Engineering / Activation Steering** — Controlling LLM behavior by intervening on activations during forward pass, using concept vectors.
  - Source: https://www.alignmentforum.org/posts/3ghj8EuKzwD3MQR5G/an-introduction-to-representation-engineering-an-activation

- **"Automated Interpretability-Driven Model Auditing and Control"** (Oxford AIGI, Jan 2026) — Research agenda for using interpretability for safety auditing.
  - Source: https://aigi.ox.ac.uk/wp-content/uploads/2026/01/Automated_interp_Research_Agenda.pdf

- **Mechanistic Interpretability — MIT 2026 Breakthrough Technology** — Anthropic used mech interp in pre-deployment safety assessment of Claude Sonnet 4.5.
  - Source: https://www.technologyreview.com/2026/01/12/1130003/mechanistic-interpretability-ai-research-models-2026-breakthrough-technologies/

### Ontology Foundations

- **SUMO (Suggested Upper Merged Ontology)** — Largest formal public ontology, mapped to all of WordNet. Foundation for HatCat's concept taxonomy.
  - Source: http://www.ontologyportal.org/
  - GitHub: https://github.com/ontologyportal/sumo

- **Google PAIR: Mapping LLMs with SAEs** — Visualizing 16,384 SAE features with hierarchical clustering and LLM-generated cluster labels.
  - Source: https://pair.withgoogle.com/explorables/sae/

- **"OntoLLM: Enhancing LLM grounding with ontologies and knowledge graphs"** (2025) — Using ontologies to ground and constrain LLM behavior.
  - Source: https://www.sciencedirect.com/science/article/abs/pii/S0957417426004185

### HatCat (Existing Platform)

- **HatCat** — Open-source (CC0) interpretability/steering platform. Detects 8,000+ concepts in LLM activations at <25ms latency. Uses WordNet/SUMO-derived ontology. Compares "thinking" (activation lenses) vs "writing" (text lenses) to detect deception.
  - Site: https://hatcat.io/
  - GitHub: https://github.com/p0ss/HatCat

## Task for the Research Team

Investigate and propose **2–3 concrete, scoped capstone project ideas** that I could realistically execute in 3 weeks, given:

- I have ontology design expertise but am learning ML/interpretability during the program
- I have access to the HatCat codebase and the ARENA 3.0 exercise code
- I want to build something, not just write a literature review
- The output should be demonstrable (a notebook, a tool, a visualization, a small study with results)
- It should connect ontology/taxonomy design to AI interpretability in a way that contributes something novel, even if small

For each proposed project, provide:
1. **Research question** (one sentence)
2. **Why it matters** for AI safety
3. **What I'd build** (concrete deliverable)
4. **Which ARENA exercises** serve as direct prerequisites/foundations
5. **How HatCat's codebase helps** (or doesn't)
6. **Risks and fallback scope** (what to cut if time runs short)
7. **Key papers to read** (max 3, with specific sections to focus on)

## Agent Architecture for This Research

Use the following multi-agent workflow to produce high-quality output. Each role has a distinct responsibility:

### Roles

1. **Orchestrator**
   - Decomposes this prompt into sub-tasks
   - Assigns work to other agents
   - Manages dependencies between tasks
   - Synthesizes the final output
   - Resolves conflicts between agent recommendations

2. **Literature Scout**
   - Searches for and retrieves relevant papers, blog posts, and codebases
   - Focuses on the intersection of: ontology/taxonomy design, SAE features, activation-level interpretability, AI safety
   - Prioritizes recency (2024–2026) and direct relevance
   - Returns structured summaries with specific section pointers

3. **Technical Feasibility Analyst**
   - Evaluates whether proposed projects are achievable in 3 weeks by someone learning ML during the program
   - Reviews the ARENA 3.0 codebase (especially Chapter 1 exercises on SAEs, TransformerLens, linear probes) to identify reusable code
   - Reviews the HatCat codebase to identify reusable components
   - Flags hard technical prerequisites and suggests workarounds

4. **Ontology Domain Expert**
   - Thinks from the perspective of ontology/taxonomy design
   - Identifies where formal ontological structure adds genuine value vs. where it's superficial
   - Evaluates how WordNet/SUMO hierarchies map (or fail to map) to learned feature spaces
   - Suggests ontology-native contributions (new taxonomies, mapping methodologies, coverage metrics)

5. **AI Safety Reviewer**
   - Evaluates each project idea against the TARA program's AI safety goals
   - Asks: does this project help us understand, detect, or control unsafe AI behavior?
   - Checks that the project connects to real safety concerns, not just interpretability-for-its-own-sake
   - References the ARENA Chapter 4 (alignment science) material

6. **Critical Reviewer**
   - Challenges every proposal: Is this actually novel? Is the scope realistic? Is the ontology angle genuine or forced?
   - Identifies the weakest assumption in each project idea
   - Suggests what would make each project fail, and how to de-risk it
   - Ensures the final recommendations are honest, not aspirational

### Workflow

```
Orchestrator
  ├── Literature Scout → finds relevant work, identifies gaps
  ├── Technical Feasibility Analyst → reviews ARENA + HatCat code, flags constraints
  ├── Ontology Domain Expert → identifies genuine ontology contributions
  │
  ├── [Synthesis round] Orchestrator combines findings into draft project proposals
  │
  ├── AI Safety Reviewer → evaluates safety relevance of each proposal
  ├── Critical Reviewer → stress-tests each proposal
  │
  └── [Final round] Orchestrator refines proposals based on reviews, produces final output
```

### Output Format

For each of the 2–3 project proposals, structure the output as:

```
## Project [N]: [Title]

### Research Question
[One sentence]

### Why It Matters (AI Safety)
[2–3 sentences]

### What You'd Build
[Concrete deliverable description]

### ARENA Prerequisites
[Which specific exercises/chapters to pay close attention to during weeks 1–11]

### HatCat Integration
[What to reuse, what to build fresh]

### 3-Week Plan
- Week 12: [specific tasks]
- Week 13: [specific tasks]
- Week 14: [specific tasks]

### Risks & Fallback
[What could go wrong, minimum viable version]

### Key Reading
1. [Paper/resource] — focus on [specific section]
2. [Paper/resource] — focus on [specific section]
3. [Paper/resource] — focus on [specific section]

### Reviewer Notes
- Safety Reviewer: [assessment]
- Critical Reviewer: [assessment]
- Feasibility: [assessment]
```
