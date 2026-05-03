# Execution Plan: Knowing the Graph

*From v2 proposal to running code. First-time-execution oriented.*
*Companion to `capstone_proposal_v2.md` and `participant_project_guide_filled.md`.*

---

## The single most important insight

**Most of the work doesn't need a GPU.**

The pipeline splits cleanly into expensive GPU work (loading Gemma 2 9B, running forward passes, extracting residual-stream activations) and cheap CPU work (everything else: probe training, statistics, plotting, bootstrap analysis, write-up).

The Anthropic emotions recipe is essentially:

```
[STORIES] --GPU--> [ACTIVATIONS .pt files] --CPU--> [PROBE VECTORS] --CPU--> [METRICS]
                       ↑                                ↑
                   ~100MB on disk              tiny — these are just vectors
```

Once you extract activations once and save them to disk, every subsequent analysis (training Conditions A/B/C probes, evaluating, bootstrapping, comparing, plotting) runs on your laptop in seconds.

This means your Lambda Labs spend is *bounded* by activation extraction — and you can be aggressive about minimising GPU time by batching everything into one or two extraction runs rather than spinning up GPUs interactively.

---

## Hardware: Lambda Labs sizing and budget

### Instance choice

| Instance | VRAM | $/hr | Verdict |
|---|---|---|---|
| A10 | 24 GB | $0.75 | Too tight for Gemma 2 9B in fp16 (~18 GB + activations) |
| **A100 40GB** | 40 GB | **$1.10** | **Recommended.** Gemma 2 9B fits with room to batch. |
| A100 80GB | 80 GB | $1.79 | Overkill; fine if you want bigger batches |
| H100 80GB | 80 GB | $2.50–3.00 | Overkill; faster but doesn't change wallclock by much for our workload |

**Pick A100 40GB on-demand.** Reserve nothing — your usage is bursty, not continuous.

### Budget

| Line item | Estimate | Notes |
|---|---|---|
| Story generation (Claude API) | **~$80** | 12K stories × ~200 output tokens × $15/MTok ≈ $36; round up for prompts, retries, Condition C articulations |
| Activation extraction (A100, ~20 GPU-hr) | **~$22** | One big run, batched |
| Steering experiments (A100, ~10 GPU-hr) | **~$11** | Stretch goal only |
| Debugging / re-runs buffer | **~$80** | First-time setup will burn some hours; budget for it |
| **Total expected** | **~$195** | Of your $400. Comfortable headroom. |

### Lambda discipline (the rule that costs people hundreds of dollars)

**Lambda instances do not auto-stop.** If you spin one up and walk away, you keep paying $1.10/hr until you remember. A weekend = $53. A week = $185.

Three habits to internalise:

1. **Terminate (not just stop) when done.** Stopping still bills storage. Terminating destroys the instance.
2. **Download artifacts to local *before* terminating.** Activation tensors, trained probes, anything you want. Lambda's filesystem is ephemeral on terminate.
3. **Use tmux for long jobs.** SSH disconnect kills foreground processes. `tmux new -s extract`, run job, `Ctrl-B D` to detach, `tmux attach -t extract` to come back.

Set a Lambda spending alert at $100 and again at $250.

---

## Software: stack and key library choices

### Local environment (your laptop)

- **Python 3.11+** (3.12 fine)
- **uv** for package management (`pip install uv` once, then `uv pip install ...`). Faster than pip; lockfiles work the same. Or just stick with `pip` + `venv` if uv is unfamiliar.
- **VS Code** or your editor of choice. Cursor if you want LLM-in-the-loop coding.
- **git** + a GitHub account.

### Core libraries

| Library | Purpose | Why this one |
|---|---|---|
| `torch` | Tensor ops everywhere | Standard |
| `transformers` | Loading Gemma 2 9B | HuggingFace standard |
| **`transformer_lens`** | **Activation extraction + steering hooks** | **You used it in ARENA Ch1; supports Gemma 2; clean hooks API** |
| `anthropic` | Claude API for story generation | Official SDK |
| `numpy`, `scipy`, `scikit-learn` | Probe math, PCA, bootstrap | Standard |
| `matplotlib`, `seaborn` | Plotting | Standard |
| `pyyaml` or `tomli` | Reading concept config | Trivial |
| `pandas` | Test-set labels, results tables | Standard |

Optional: `nnsight` is a more modern alternative to transformer_lens for activation extraction. Stick with transformer_lens since you already know it.

### Accounts you need (set up *before* any coding)

1. **Anthropic API key.** https://console.anthropic.com → API keys. Add a payment method.
2. **HuggingFace account** + **accept the Gemma 2 license**. Gemma is gated. Visit https://huggingface.co/google/gemma-2-9b-it, click "Acknowledge license," wait for approval (usually instant). Generate an HF access token (Settings → Access Tokens, "read" scope).
3. **Lambda Labs account.** https://lambdalabs.com → sign up → add payment method → SSH keys (generate one if you don't have one, paste public key into Lambda dashboard).
4. **GitHub** for the capstone repo. Public is recommended for portfolio visibility, but private is fine for now and can be flipped later.

---

## Repository structure (recommended)

Create a **separate** repo for the capstone, not inside ARENA_3.0. ARENA is course material; the capstone is your own work.

```
capstone-knowing-the-graph/
├── README.md                     # Project overview, how to run, key findings
├── pyproject.toml                # Or requirements.txt
├── .gitignore                    # Ignore data/, activations/, .env, results/*.png
├── .env.example                  # Template — never commit real .env
│
├── data/
│   ├── concepts.yaml             # The 12 concepts + their SUMO neighbours
│   ├── stories/                  # Generated stories per concept (gitignored)
│   ├── articulations/            # Condition C boundary examples (gitignored)
│   ├── neutral_corpus/           # For PCA confound projection (gitignored)
│   └── test_set/
│       ├── passages.csv          # Hand-labelled passages (committed)
│       └── sources.md            # Where each came from (committed)
│
├── activations/                  # Extracted residual-stream tensors (gitignored, ~hundreds of MB)
├── probes/                       # Trained probe vectors per condition (committed, small)
├── results/
│   ├── figures/                  # Plots
│   ├── tables/                   # AUC tables, bootstrap CIs
│   └── logs/                     # Run logs
│
├── notebooks/                    # Numbered for sequencing
│   ├── 01_explore_anthropic_recipe.ipynb    # Smoke-test the pipeline on 2 concepts
│   ├── 02_generate_stories.ipynb            # Local; Claude API
│   ├── 03_extract_activations.ipynb         # Lambda; loads Gemma, saves .pt
│   ├── 04_train_probes.ipynb                # Local; mean-diff + PCA cleaning
│   ├── 05_evaluate_test_set.ipynb           # Lambda for forward passes; local for analysis
│   ├── 06_bootstrap_analysis.ipynb          # Local
│   ├── 07_causal_validation.ipynb           # Lambda; probe-target × steering correlation
│   └── 08_steering_stretch.ipynb            # Lambda; only if Phase 5 lands clean
│
└── src/
    ├── __init__.py
    ├── concepts.py               # Load concept graph, neighbours, etc.
    ├── generation.py             # Story generation via Claude API
    ├── activations.py            # Forward-pass extraction with TransformerLens
    ├── probes.py                 # Mean-diff probe + PCA confound projection
    ├── evaluate.py               # AUC, confusion matrix, paired bootstrap
    ├── steering.py               # Hook-based interventions
    └── utils.py
```

Notebooks for *exploration and sequencing*; `src/` modules for *the actual logic* (so notebooks stay thin and code stays testable). Tests are nice-to-have, not required for a 3-week project.

---

## Phase-by-phase execution

### Phase 0 — This week, before anything else (≤4 hours)

The goal is to verify the *whole pipeline works on toy data* before touching the real experiment.

**Checklist:**

- [ ] All four accounts set up (Anthropic, HuggingFace + Gemma access, Lambda, GitHub)
- [ ] New `capstone-knowing-the-graph` repo created on GitHub, cloned locally
- [ ] Local Python env created and activated; `transformer_lens`, `transformers`, `anthropic`, `torch` installed
- [ ] `.env` file with `ANTHROPIC_API_KEY` and `HF_TOKEN` (gitignored — never commit this)
- [ ] **Smoke test 1 (local):** generate one story via the Claude API and print it. ~10 lines of code.
- [ ] **Smoke test 2 (Lambda):** spin up an A100, ssh in, install requirements, load Gemma 2 9B, run `model("Hello world")`, extract one residual-stream activation at layer 21, save the tensor, terminate the instance. **Time-box this to 2 hours and don't forget to terminate.**

If both smoke tests work, you're ready for Phase 1. If smoke test 2 chews 4 hours and still doesn't work, that's important information — surface it to Ahmed.

### Phase 1 — Story generation (Week 12 days 1-2, all local, no GPU)

- Define the 12 concepts in `data/concepts.yaml` with their SUMO neighbours
- Write `src/generation.py`: a function that takes a concept and a topic, prompts Claude Sonnet 4.5 with the Anthropic-style template ("Write a paragraph in which a character experiences/embodies [concept]…"), returns the story
- Run for all 12 concepts × ~80 topics × ~12 stories per topic ≈ 12K stories, save to `data/stories/<concept>.jsonl`
- Generate Condition C boundary articulations: for each concept pair (66 pairs), generate ~50 short passages explicitly contrasting the two
- Build a neutral corpus (~5K passages — Wikipedia paragraphs, technical writing, etc.) for PCA confound projection
- **Cost:** ~$80 Claude API
- **Wallclock:** half a day if you parallelise the API calls properly; a day if serial

### Phase 2 — Activation extraction (Week 12 days 3-4, GPU-required)

This is the big GPU run.

- Spin up A100, install requirements
- Run `notebooks/03_extract_activations.ipynb` (or a `.py` script — better for long jobs in tmux):
  - For each story, run forward pass on Gemma 2 9B
  - Read residual-stream activations at layer 21
  - Take the mean over token positions 50+
  - Save per-concept tensors: `activations/<concept>.pt` (shape `[n_stories, d_model]`)
- Also extract activations for the neutral corpus → `activations/neutral.pt`
- **Download all activation files to your laptop**, then **terminate the instance**
- **Cost:** ~$22 (20 GPU-hours at $1.10)
- **Wallclock:** ~1 day including setup + babysitting

### Phase 3 — Probe training (Week 12 days 5-7, all local, no GPU)

Now everything is fast and cheap.

- Implement `src/probes.py`:
  - `MeanDiffProbe`: takes per-concept activations, returns the difference vector (concept mean − grand mean across all concepts)
  - `pca_confound_projection`: fit PCA on the neutral activations, project out top components covering 50% variance
- Train Condition A and B probes for all 12 concepts (B differs from A only in which negatives are sampled — no separate forward passes needed)
- Save probe vectors to `probes/condition_a/` and `probes/condition_b/`
- **Sanity check:** does Condition A reproduce Anthropic-style structure? Run PCA on the bank of 12 probes — does PC1 separate "honest" concepts (persuasion) from "deceptive" concepts (deception, lying)? Doesn't have to be perfect, but should be *something*.

### Phase 4 — Test set construction + Condition C (Week 13 days 1-3)

Two parallel tracks:

- **Track A (your time, no GPU):** hand-label remaining ~90 test passages from external sources. Pull from MASK benchmark (Anthropic 2025), ethics case studies, news transcripts. Keep `data/test_set/passages.csv` with columns `passage_id, source, text, primary_concept, secondary_concepts, notes`.
- **Track B (local CPU):** train Condition C probes. The boundary articulations from Phase 1 contribute additional positive/negative examples beyond the synthetic stories — assemble these into the per-concept training set, then re-run the mean-diff + PCA pipeline.

### Phase 5 — Evaluation (Week 13 days 4-6, light GPU for test-set forward passes)

- Spin up A100 briefly to extract activations for the 120 test passages (~5 minutes of GPU time, but you need the GPU). Save tensors to `activations/test_set.pt`. Terminate.
- Locally: project test-set activations onto each probe direction → score per probe, per passage
- Compute per-pair AUC across all 66 concept pairs for each condition (A, B, C)
- Run the paired bootstrap (10K resamples) comparing A vs B, B vs C, A vs C
- Generate the confusion matrices and per-pair AUC heatmap

### Phase 6 — Causal validation (Week 13 day 7, GPU-required, light)

- Spin up A100. For each of the 12 probes in each condition:
  - Run a small set of "neutral" prompts (~50)
  - Compute baseline probe activation on each
  - Add the probe direction × small coefficient to the residual stream at layer 21 (intervention)
  - Re-run forward pass, measure shift in completion behaviour (or downstream probe activations)
- Plot probe-target-correlation vs. steering-magnitude. Anthropic got r=0.85 for emotions; you want this correlation to be high for at least one of the conditions (otherwise the probes aren't causally meaningful)
- Terminate

### Phase 7 — Steering stretch (Week 14, GPU, ~10 hours, optional)

Only if Phase 6 results are clean. Pre-register the "collateral disturbance" measure first, then run.

### Phase 8 — Write-up (Week 14)

- 4–6 page write-up in markdown, then convert to LessWrong/Alignment Forum format
- Slides for Saturday June 13 presentation
- Polish notebook for reproducibility — `pip install -r requirements.txt; python run_all.py` should work end-to-end on a fresh checkout

---

## Common pitfalls (forewarned is forearmed)

1. **Forgetting to terminate Lambda instances.** $1.10/hr × 168 hours/week = $185/week of accidental burn. Set a calendar reminder for "did you terminate the GPU?" every evening of an active phase.
2. **Generating stories serially via the Anthropic API.** A naive `for concept in concepts: for topic in topics: ...` loop will take ~hours. Use `asyncio` + `anthropic.AsyncAnthropic` to parallelise — 10× faster, same total cost.
3. **Saving activations at the wrong granularity.** If you save *all* token-position activations, you'll have ~150 GB of tensors. You only need the token-50+ mean per story → ~100 MB total. Collapse the time dimension *before* saving.
4. **Not pinning Gemma 2 9B's specific revision.** HuggingFace model snapshots can change. In `transformer_lens.HookedTransformer.from_pretrained("gemma-2-9b-it")`, you can pin a specific commit — do so for reproducibility.
5. **Burning days on test-set labelling.** It will take longer than you expect. Start labelling in the *first day* of Phase 1 in parallel with story generation, not in Phase 4.
6. **Conflating `gemma-2-9b` (base model) with `gemma-2-9b-it` (instruction-tuned).** The Anthropic emotions paper uses an instruction-tuned model. Use `gemma-2-9b-it`. Different residual stream behaviour.
7. **Treating Conditions A, B, C as three separate training runs.** They share the same activation tensors — you only extract once. The conditions differ in which negatives the probe construction uses, not in what activations exist on disk.
8. **Letting the notebook drift from `src/`.** When you write logic in a notebook, immediately move it into a `src/` module and import from there. Otherwise nothing is reusable across notebooks.

---

## Day-1 starter checklist (do this before Saturday)

In order:

1. ☐ Create `capstone-knowing-the-graph` repo on GitHub
2. ☐ Clone locally, `mkdir -p` the directory structure above
3. ☐ Anthropic console: get API key, set $50 spending limit (you can raise it later)
4. ☐ HuggingFace: accept Gemma 2 9B license, generate access token
5. ☐ Lambda Labs: account, payment, SSH key uploaded
6. ☐ Local Python env: `python3.11 -m venv .venv && source .venv/bin/activate && pip install transformer_lens transformers anthropic torch numpy scipy scikit-learn matplotlib seaborn pandas pyyaml python-dotenv`
7. ☐ `.env` with `ANTHROPIC_API_KEY=...` and `HF_TOKEN=...`. Add `.env` to `.gitignore`.
8. ☐ Smoke test 1: write a 10-line script that generates one Claude story and prints it. Commit.
9. ☐ Smoke test 2: spin up A100, ssh in, run `from transformer_lens import HookedTransformer; m = HookedTransformer.from_pretrained("gemma-2-9b-it"); print(m("Hello world"))`. **Terminate.** Note the wallclock and cost.
10. ☐ If 8 and 9 work: you're set for Phase 1 next week.

---

## What to do *today* (after reading this)

1. Re-read the Day-1 starter checklist above. Decide whether you can do it before Saturday's TARA session.
2. Once you're comfortable with the plan, ask me to scaffold the actual repo skeleton — I can generate the initial `src/` modules, `.gitignore`, `pyproject.toml`, and the smoke-test scripts so you have working code on Day 1, not blank files.
3. Mention to Ahmed at the next session that you're ready to start execution and would value a check-in mid-Phase 0 to make sure the smoke tests passed.

The biggest unknown for first-time execution is always Phase 0 — the moment when "I have an account at Lambda" turns into "I have working code that loads Gemma and reads activations." Once that's working, the rest is incremental.
