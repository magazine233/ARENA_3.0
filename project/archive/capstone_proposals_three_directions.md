# TARA Capstone Project Proposals
## Ontology Design × AI Interpretability

*Prepared for Jason — March 2026*

---

## Executive Summary

Three capstone project proposals are presented below, each sitting at the intersection of formal ontology design and mechanistic interpretability. They are ordered by feasibility (most accessible first), and each is designed so that the ontology contribution is genuine — not bolted on — while the ML components build directly on ARENA exercises you'll have completed by week 12.

The proposals were stress-tested against five criteria: (1) Is the ontology angle genuine or forced? (2) Is it achievable in 3 weeks by someone learning ML during the program? (3) Does it produce a demonstrable deliverable? (4) Does it connect to real AI safety concerns? (5) Is there a viable fallback if time runs short?

---

## Project 1: Ontological Coverage Audit of SAE Feature Dictionaries

### Research Question

Do sparse autoencoder feature dictionaries systematically miss entire branches of human-meaningful concept space, and can formal ontology hierarchies reveal these blind spots?

### Why It Matters (AI Safety)

If we're relying on SAE features as the basis for understanding what models represent internally — and increasingly using them for safety audits and model steering — then systematic gaps in what SAEs can "see" are safety-critical. A model could represent concepts that no SAE feature captures, meaning our interpretability tools would be blind to those internal states. This is the interpretability equivalent of a security audit that only checks half the attack surface. Formal ontologies provide a principled way to define "complete concept coverage" rather than relying on ad-hoc sampling.

### What You'd Build

A **coverage analysis tool and report** that:

1. Takes a pre-trained SAE's feature dictionary (e.g., from Gemma Scope or the SAEBench suite of 200+ open-source SAEs)
2. Uses automated interpretability labels from Neuronpedia (LLM-generated descriptions of what each feature does)
3. Maps those feature descriptions onto WordNet/SUMO concept nodes using semantic similarity and ontology traversal
4. Computes coverage metrics across the SUMO hierarchy: which top-level categories are well-represented? Which are sparse? Which are entirely absent?
5. Produces a visual "concept coverage map" — essentially a heatmap of the ontology tree coloured by SAE feature density

**Concrete deliverables:**
- A Jupyter notebook implementing the full pipeline
- Coverage heatmap visualizations across SUMO's top-level categories
- A short writeup (~3 pages) presenting the coverage gaps found and their safety implications
- Optionally: comparison of coverage across different SAE widths (e.g., 16k vs 65k features) to see if scaling closes the gaps

### ARENA Prerequisites

- **Chapter 0 (Weeks 1–3):** PyTorch fundamentals — you'll need basic tensor operations and familiarity with loading pre-trained models
- **Chapter 1, Section 1.3.2 (Weeks 4–6):** This is the critical section. The ARENA SAE exercises introduce SAELens and Neuronpedia, show you how to load different SAE releases (including Gemma Scope), and run them alongside TransformerLens models. Parts 1 (Intro to SAE interpretability) and 2 (Understanding/classifying latents) are essential foundations
- **Chapter 3 (Weeks 9–10):** The eval frameworks section teaches you to think systematically about evaluation methodology, which applies directly to designing coverage metrics

### HatCat Integration

**Heavy reuse.** HatCat's `data/concept_graph/` directory contains the pre-built SUMO/WordNet concept hierarchy — this is the ontology backbone you'd map SAE features onto. HatCat's `concept_packs/` contain model-agnostic ontology specifications with ~8,000+ concepts already organized taxonomically. You can use these as your ground-truth concept space.

You'd build fresh: the mapping algorithm (matching SAE feature descriptions to ontology nodes) and the coverage metric computation.

### 3-Week Plan

**Week 12:**
- Set up the pipeline: load a pre-trained SAE from Gemma Scope via SAELens
- Pull automated interpretability labels from Neuronpedia's API
- Load HatCat's SUMO/WordNet concept graph
- Implement a basic semantic matching algorithm (start simple: keyword/embedding similarity between feature descriptions and concept labels)

**Week 13:**
- Compute coverage metrics across the ontology hierarchy
- Build visualizations (heatmaps of the SUMO tree)
- Run the analysis across 2–3 SAE widths to see how coverage scales
- Identify the most significant coverage gaps

**Week 14:**
- Analyze whether the gaps correlate with safety-relevant concept categories
- Write up findings
- Polish the notebook into a shareable deliverable
- If time permits: compare coverage between standard SAEs and Matryoshka/Hierarchical SAEs

### Risks & Fallback

**Primary risk:** The semantic matching between SAE feature descriptions and ontology nodes could be noisy. LLM-generated feature labels are imperfect, and matching them to formal ontology terms isn't trivial.

**Mitigation:** Start with a simple keyword matching baseline. If that's too noisy, use embedding similarity (sentence-transformers). If even that's poor, narrow scope to a single SUMO domain (e.g., "Mental Processes" or "Social Interactions") where you can manually validate the matches.

**Minimum viable version:** A coverage report for a single SAE on a single SUMO sub-tree, with manual validation of the top 50 matches and top 50 gaps. This is achievable in under 2 weeks even with significant debugging time.

**What could make it fail:** If SAE feature labels are so poorly described that no reasonable mapping algorithm can link them to ontology concepts. This is unlikely given recent improvements in automated interpretability, but check early in week 12.

### Key Reading

1. **SAEBench (Karvonen et al., 2025)** — Focus on Section 3 (evaluation metrics) and Section 5 (results across architectures). Gives you the landscape of available SAEs and how they're evaluated.
   - Source: https://arxiv.org/abs/2503.09532

2. **"The Geometry of Concepts" (Li et al., 2025)** — Focus on Section 4 ("brain-scale" clustering) which shows SAE features cluster into functional groups. Your project asks: do those clusters align with formal ontology categories?
   - Source: https://arxiv.org/abs/2410.19750
   - Code: https://github.com/ejmichaud/feature-geometry

3. **Google PAIR: Mapping LLMs with SAEs** — This exploreable shows how Google organized 16,384 SAE features using hierarchical clustering and LLM-generated labels. Your contribution is replacing their ad-hoc clustering with formal ontological structure.
   - Source: https://pair.withgoogle.com/explorables/sae/

### Reviewer Notes

- **Safety Reviewer:** Strong safety relevance. If SAE dictionaries have systematic blind spots, then safety audits built on SAEs inherit those blind spots. The ontology provides a principled "concept census" against which to measure coverage — this is a novel contribution. Connects to the broader question from Chapter 4 (alignment science): how do we know our interpretability tools are sufficient?

- **Critical Reviewer:** The weakest assumption is that SAE feature labels from automated interpretability are good enough to map to ontology nodes. This could produce a noisy signal. However, the project is valuable even if the mapping is imperfect — showing *where* it breaks is itself informative (which concepts are hard to label? Are those the safety-critical ones?). The scope is realistic. The ontology angle is genuine — you're not just decorating an SAE analysis with ontology labels, you're using the ontology's completeness guarantee to find gaps that no amount of bottom-up analysis would reveal.

- **Feasibility:** High. Uses pre-trained SAEs (no training needed), pre-existing feature labels (no LLM calls needed), and HatCat's pre-built concept graph. The core ML task is loading SAEs and extracting feature vectors, which is directly covered in ARENA 1.3.2. The mapping algorithm is fundamentally an ontology task — right in your wheelhouse. Cloud GPU not needed.

---

## Project 2: Ontology-Guided SAE Feature Organization and the Hierarchy Alignment Test

### Research Question

When SAE features are organized by co-occurrence and geometric proximity, does the resulting structure align with formal ontological hierarchies — and where it diverges, what does that reveal about how LLMs actually organize knowledge?

### Why It Matters (AI Safety)

If LLMs organize concepts differently from how humans formally categorize them, that divergence itself is safety-relevant. It could mean models have "hidden category boundaries" that we wouldn't predict from human ontologies — and those boundaries might determine how concepts like "deception" or "power-seeking" are internally related to seemingly unrelated concepts. Understanding whether LLM concept structure matches or departs from formal ontology tells us whether human-designed safety taxonomies (like HatCat's) are well-calibrated to the model's actual internal organization.

### What You'd Build

A **comparative analysis tool** that:

1. Takes the SAE feature geometry from a pre-trained SAE (feature vectors from the decoder matrix)
2. Clusters features using the same co-occurrence and geometric methods from "The Geometry of Concepts" paper (their code is open-source)
3. Maps features to SUMO/WordNet concepts (reusing methodology from Project 1)
4. Compares two hierarchies: (a) the data-driven hierarchy from clustering SAE features, and (b) the formal hierarchy from SUMO/WordNet
5. Quantifies alignment using tree comparison metrics (e.g., tree edit distance, normalized mutual information between cluster assignments and ontology categories)
6. Produces an interactive visualization showing where the hierarchies agree and where they diverge

**Concrete deliverables:**
- A Jupyter notebook with the full comparison pipeline
- "Alignment score" metrics at multiple levels of the hierarchy
- Visualization of the two trees side-by-side with alignment/divergence highlighted
- Analysis of the most interesting divergences (where the LLM groups concepts differently from SUMO)

### ARENA Prerequisites

- **Chapter 0 (Weeks 1–3):** PyTorch basics, working with tensors and similarity metrics
- **Chapter 1, Section 1.3.2 (Weeks 4–6):** SAE fundamentals — loading SAEs, understanding decoder matrices, feature activation patterns. The section on understanding/classifying latents is especially relevant
- **Chapter 1, Section 1.2 (Weeks 4–6):** The linear probing section covers how to extract and analyze internal representations — the conceptual foundation for understanding what features mean
- **Chapter 1, optional Probing & Representations branch:** If you follow this branch, you'll gain experience with exactly the kind of representation analysis needed here

### HatCat Integration

**Moderate reuse.** HatCat's concept graph provides the formal hierarchy for comparison. HatCat's approach of using WordNet/SUMO to let the model "define each concept in its own representational language" is directly relevant — you're testing whether the model's own geometry agrees with those definitions.

You'd build fresh: the clustering pipeline (adapted from the Geometry of Concepts open-source code), the tree comparison metrics, and the visualizations.

### 3-Week Plan

**Week 12:**
- Fork/adapt the Geometry of Concepts codebase (https://github.com/ejmichaud/feature-geometry)
- Load a pre-trained SAE and compute feature co-occurrence clusters
- Load HatCat's concept hierarchy
- Implement feature-to-concept mapping (from Project 1, or simplified version)

**Week 13:**
- Implement tree comparison metrics between data-driven clusters and ontology categories
- Run the comparison at multiple granularity levels (SUMO top-level categories → mid-level → fine-grained)
- Build visualizations showing alignment and divergence
- Begin identifying the most interesting divergence cases

**Week 14:**
- Deep-dive analysis of 3–5 most interesting divergences
- Investigate whether divergences correlate with known model behaviors (e.g., do "deception" and "persuasion" cluster together in the model despite being in different SUMO categories?)
- Write up findings
- Polish deliverables

### Risks & Fallback

**Primary risk:** This is more technically ambitious than Project 1 — it requires both the feature-to-concept mapping AND a separate geometric clustering analysis. If the Geometry of Concepts code doesn't adapt cleanly to your chosen SAE, you could lose time on engineering.

**Mitigation:** Start with the simplest clustering approach (cosine similarity of decoder vectors → hierarchical clustering) rather than the full crystal/lobe analysis from the paper. The comparison itself is the novel part, not the clustering method.

**Minimum viable version:** Skip the geometric clustering entirely. Instead, take the pre-computed feature labels from Neuronpedia, assign each to its nearest SUMO concept, and compare: are features from the same SUMO category also close in activation space (using cosine similarity of decoder vectors)? This becomes a simpler "do ontologically related concepts have geometrically related features?" test, achievable in ~1.5 weeks.

**What could make it fail:** If the mapping between SAE features and ontology concepts is too noisy (same risk as Project 1), the tree comparison becomes meaningless. Run Project 1's mapping quality checks first.

### Key Reading

1. **"Incorporating Hierarchical Semantics in Sparse Autoencoder Architectures" (Muchane et al., 2025)** — Focus on Section 3 (architecture) and Figures 1 and 6. This paper shows that SAEs can learn explicit hierarchical structure. Your project tests whether *standard* SAEs implicitly learn structure that matches formal ontology.
   - Source: https://arxiv.org/abs/2506.01197

2. **"The Geometry of Concepts" (Li et al., 2025)** — Focus on Section 4 (brain-scale analysis) and the clustering methodology. You'll adapt their co-occurrence clustering and spatial modularity metrics.
   - Source: https://arxiv.org/abs/2410.19750
   - Code: https://github.com/ejmichaud/feature-geometry

3. **SUMO → WordNet mapping documentation** — Focus on understanding the mapping types (equivalence, subsumption, instance) and how to traverse the hierarchy programmatically.
   - Source: https://www.ontologyportal.org/
   - GitHub: https://github.com/ontologyportal/sumo

### Reviewer Notes

- **Safety Reviewer:** Good safety relevance, though one step removed from direct safety applications. The key safety question is whether models organize safety-relevant concepts in ways we'd predict from human ontologies. If "manipulation" clusters with "helpfulness" in model space (rather than with "deception" as SUMO would predict), that's an important finding for alignment. The project produces actionable information about whether human-designed safety taxonomies are well-calibrated to model internals.

- **Critical Reviewer:** This is the most intellectually interesting proposal but also the most ambitious. The risk of scope creep is real — comparing two hierarchies raises endless sub-questions. The critical path is: pick ONE comparison metric, run it, show the result. Don't try to build a comprehensive comparison framework. The biggest weakness: even interesting divergences might just reflect tokenization artifacts or training data distribution rather than deep representational differences. You'd need to distinguish signal from noise. Consider using the H-SAE paper (Muchane et al.) as a positive control — if H-SAEs produce hierarchy that aligns better with SUMO than standard SAEs, that validates your methodology.

- **Feasibility:** Medium. More technically complex than Project 1, but the open-source Geometry of Concepts code and pre-trained SAEs reduce the ML burden. The ontology comparison is where your expertise dominates. The minimum viable version (cosine similarity test) is achievable. Cloud GPU not strictly needed, but helpful if you want to run activations through larger models.

---

## Project 3: Ontology-Driven Safety Concept Testing — Systematic Evaluation of Concept Detection Completeness

### Research Question

Can formal ontology structure systematically generate test cases that reveal which safety-relevant concepts are detectable (and which are invisible) across different interpretability methods — and do ontologically "nearby" concepts share detection patterns?

### Why It Matters (AI Safety)

Current safety concept detection (whether via SAE features, linear probes, or HatCat-style learned classifiers) is tested on hand-picked concepts — typically a few dozen chosen by researchers. Nobody systematically tests whether *all the concepts that matter* are detectable. Ontologies provide a natural way to generate systematic test suites: if you can detect "Deception," can you also detect its ontological siblings ("Manipulation," "Misdirection," "Concealment") and its children ("Lies," "Half-truths," "Omission")? This project builds an evaluation framework that uses ontology structure to generate comprehensive safety concept test suites, moving from ad-hoc testing to principled coverage.

### What You'd Build

A **safety concept detection evaluation framework** that:

1. Starts from a safety-relevant sub-tree of SUMO/WordNet (e.g., concepts under "IntentionalPsychologicalProcess" → "Deception," "Persuading," "Requesting"; or under "NormativeAttribute" → concepts related to ethics, harm, safety)
2. For each concept in the sub-tree, generates test prompts using the ontology's formal definitions (leveraging LLM assistance to expand formal definitions into natural-language scenarios)
3. Tests whether each concept is detectable via: (a) SAE feature activation (does any feature fire?), (b) linear probing (can a probe classify its presence?), and/or (c) HatCat-style learned classifiers (if available)
4. Maps detection success/failure back onto the ontology tree to reveal systematic patterns: are entire branches invisible? Do abstract concepts fail while concrete ones succeed?
5. Produces a "safety concept detection scorecard" organized by ontology structure

**Concrete deliverables:**
- An ontology-derived test suite for safety concept detection (~100–200 test cases across 30–50 concepts)
- A detection scorecard showing which concepts are detectable, partially detectable, or invisible
- Analysis of whether detection follows ontological structure (e.g., "all Physical Harm concepts are detectable but Psychological Manipulation concepts are not")
- A reusable framework that others could extend to new ontology branches

### ARENA Prerequisites

- **Chapter 1, Section 1.3.2 (Weeks 4–6):** SAE fundamentals — loading SAEs, inspecting feature activations on specific inputs
- **Chapter 1, Probing branch (Weeks 4–6):** Linear probing is one of the detection methods you'll test. The ARENA exercises teach you to train probes on model activations
- **Chapter 3 (Weeks 9–10):** The eval design section is directly applicable — you're building a structured evaluation
- **Chapter 4 (Week 11):** The alignment science material provides the safety framing. Persona vectors and emergent misalignment directly connect to the kinds of safety concepts you'd be testing for

### HatCat Integration

**Significant reuse.** HatCat's concept packs already define ~8,000+ concepts with ontological structure, many of which are safety-relevant. HatCat's training pipeline (in `scripts/training/`) shows how to train concept classifiers on model activations — you can adapt this for your detection tests. HatCat's approach of using "thinking vs. writing" divergence to detect deception is also relevant: your framework could test whether divergence detection generalizes across ontologically related concepts (e.g., does deception detection generalize to "concealment" or "misdirection"?).

You'd build fresh: the test case generation pipeline, the systematic evaluation framework, and the coverage analysis.

### 3-Week Plan

**Week 12:**
- Select 2–3 safety-relevant SUMO sub-trees (e.g., deception, harm, power-seeking)
- Extract concept hierarchies and formal definitions
- Generate test prompts for each concept (~5 prompts per concept, using LLM assistance to expand formal definitions into scenarios)
- Load a pre-trained SAE and set up basic feature activation inspection

**Week 13:**
- Run detection tests: for each test prompt, check SAE feature activations and (if time allows) train simple linear probes
- Score each concept: detectable / partially detectable / invisible
- Map results back onto the ontology tree
- Look for patterns: do entire branches fail? Do abstract concepts fail more than concrete ones?

**Week 14:**
- Analyze whether HatCat's existing concept lenses cover the gaps (if HatCat lens packs are available for your model)
- Compare: which detection method works best for which concept types?
- Write up findings with the "safety concept detection scorecard"
- Polish the framework for reusability

### Risks & Fallback

**Primary risk:** Generating good test prompts from formal ontology definitions is non-trivial. SUMO's formal definitions (in SUO-KIF logic) don't directly translate to natural language scenarios that would reliably activate a concept in model activations.

**Mitigation:** Use an LLM (Claude, GPT-4) to translate formal definitions into test scenarios. You're not training anything here — just using an LLM to expand "IntentionalDeception: an IntentionalPsychologicalProcess where the agent attempts to cause a false belief in another agent" into concrete text passages. This is straightforward prompt engineering.

**Minimum viable version:** Focus on a single safety concept family (e.g., "deception and its ontological neighborhood" — 10–15 concepts). Generate 5 test prompts per concept. Test detection via SAE feature activation only (no probes). Produce a simple scorecard. This is achievable in under 2 weeks.

**What could make it fail:** If the SAE features don't activate meaningfully on short test prompts (SAE features sometimes need longer context), the detection test becomes unreliable. Mitigation: use longer passages (200+ tokens) as test inputs rather than short prompts.

### Key Reading

1. **"Safety Concept Activation Vectors (SCAV)" (NeurIPS 2024)** — Focus on Section 3 (methodology for separating safe/malicious concepts in activation space). Your project extends their approach from ad-hoc concept selection to ontology-systematic concept selection.
   - Source: https://proceedings.neurips.cc/paper_files/paper/2024/file/d3a230d716e65afab578a8eb31a8d25f-Paper-Conference.pdf

2. **"Incorporating Hierarchical Semantics in SAEs" (Muchane et al., 2025)** — Focus on Section 4 (evaluation) which discusses feature absorption and the first-letter classification benchmark. Your project proposes an ontology-structured alternative to these benchmarks for safety concepts.
   - Source: https://arxiv.org/abs/2506.01197

3. **ARENA Chapter 3 exercises** — The eval design methodology is directly applicable. Pay attention to how they structure benchmark creation and how they use LLMs to generate evaluation datasets.
   - Source: https://github.com/callummcdougall/ARENA_3.0 (Chapter 3 materials)

### Reviewer Notes

- **Safety Reviewer:** Strongest direct safety relevance of the three proposals. This project directly addresses the question: "Can we trust our interpretability tools to find the safety-relevant concepts?" Moving from ad-hoc concept testing to systematic, ontology-driven testing is exactly what governance frameworks (like the EU AI Act and Australian AI governance requirements) will eventually need. The connection to HatCat's real-world governance application strengthens this further.

- **Critical Reviewer:** The risk is that this becomes a large-scale prompt engineering exercise rather than an interpretability research contribution. The ontology angle is genuine — you're using formal structure to ensure completeness — but the ML depth is lighter than the other two projects. The detection testing is relatively simple (check if features activate, train a probe). The novelty is in the *methodology* (systematic coverage via ontology) rather than in the *technique*. This is fine for a capstone, but be clear about what the contribution is. The weakest assumption is that concept detection via SAE feature activation on short prompts is a meaningful test — you might need longer contexts, which adds complexity.

- **Feasibility:** Medium-high. The test case generation is mostly ontology work and prompt engineering (your strengths). The ML component (loading SAEs, checking activations, optionally training probes) is straightforward given ARENA exercises. The main unknown is how well formal ontology definitions translate into effective test prompts. Cloud GPU not needed for detection (inference on pre-trained models), but helpful if you want to train probes.

---

## Comparison and Recommendation

| Dimension | Project 1: Coverage Audit | Project 2: Hierarchy Alignment | Project 3: Safety Testing |
|-----------|--------------------------|-------------------------------|--------------------------|
| **Feasibility** | Highest | Medium | Medium-High |
| **Ontology depth** | Medium | Highest | High |
| **ML depth** | Low | Medium | Low-Medium |
| **Safety relevance** | High | Medium-High | Highest |
| **Novelty** | Medium | Highest | High |
| **Fallback scope** | Very clear | Clear | Clear |
| **GPU needed?** | No | Helpful | No |
| **HatCat reuse** | Heavy | Moderate | Significant |

### My recommendation: Start with Project 1, with Project 3 as your stretch goal.

**Rationale:**

Project 1 is the lowest-risk, highest-payoff starting point. It produces a clear, useful deliverable (coverage gaps in SAE dictionaries), directly leverages your ontology expertise, requires minimal ML beyond what ARENA teaches, and has a very clear minimum viable version. If you finish it quickly, the mapping pipeline you build is directly reusable in Projects 2 or 3.

Project 3 has the strongest safety story and the most direct connection to your government AI governance work. If Project 1's mapping pipeline works well, extending it to safety concept testing is a natural next step.

Project 2 is the most intellectually ambitious and would produce the most novel research contribution — but it carries more risk of scope creep and requires more ML sophistication. Consider it if you find yourself ahead of schedule and confident with the SAE tools after week 11.

### A practical note on compute

All three projects work primarily with pre-trained SAEs (loaded from HuggingFace via SAELens) and pre-existing models. You don't need to train any large models. The main compute requirements are:
- Loading SAEs and running inference (fits comfortably on Colab Pro or a modest cloud GPU)
- Running model activations on test prompts (similarly modest)
- The HatCat lens training pipeline would need more compute if you go that route, but it's not required for any minimum viable version

Your TARA compute budget should be more than sufficient for all three projects.

---

## Appendix: Key Tools and Resources

### Pre-trained SAEs you can use immediately
- **Gemma Scope:** JumpReLU SAEs for Gemma 2 (2B and 9B), various widths — available via SAELens
- **SAEBench suite:** 200+ SAEs across 7 architectures on Pythia-160M and Gemma-2-2B — open-source on HuggingFace
- **Neuronpedia:** Interactive interface with automated feature labels — https://www.neuronpedia.org/

### Code to build on
- **SAELens:** Python library for loading and working with SAEs — https://github.com/jbloomAus/SAELens
- **TransformerLens:** Interpretability-friendly model loading — https://github.com/TransformerLensOrg/TransformerLens
- **Geometry of Concepts code:** Feature clustering and analysis — https://github.com/ejmichaud/feature-geometry
- **HatCat:** Concept taxonomy and detection — https://github.com/p0ss/HatCat
- **ARENA 3.0 exercises:** Chapter 1, Section 1.3.2 — https://github.com/callummcdougall/ARENA_3.0

### Ontology resources
- **SUMO:** https://github.com/ontologyportal/sumo
- **WordNet (NLTK):** `import nltk; nltk.download('wordnet')` — provides the synset hierarchy in Python
- **SUMO-WordNet mappings:** Available in the SUMO repository
