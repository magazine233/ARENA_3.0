# Capstone — MAP (start here)

*Navigational index for the relational-probe study. The one place to orient. Updated 2026-08-05.*

## Iteration ↔ HatCat-"Path" key (the naming that was getting confusing)

| Our label | What it tested | HatCat "Path" | Verdict |
|---|---|---|---|
| **iteration 1** | def-vs-relational probing + Menger (multiclass) | — | null (later understood as the *wrong question*) |
| **iteration 2** | pair-matched, powered, pre-registered Menger | the validity + power fix | **powered NULL** (partial Spearman −0.125) |
| **iteration 3** | relations as **hard negatives** | **Path 2** | **bounded positive** — local + ~⅓ lexical |
| **iteration 4** (done 2026-06-12) | scale-up retest of alternate-path / Menger | **Path 1 (scale) × Path-2 held-out-edge** | **powered NULL** — top volume +0.005, CI spans 0, MDE 0.014 |
| **near-OOS rejection** (done 2026-08-05) | does anything seal the *legitimate-influence* face? | §7 deployment add-on | **NULL** — 0.164/0.175/0.182, no scheme beats random; positive controls fire (−0.286, p<0.001) |
| (later) | generate-mode vs read-mode | **Path 3** | first run done by p0ss (non-factor); contrast-prompted variant not started |

## Owners
- **Jason (lead):** iterations 1–4 — pre-registrations, controls, the paper + slides.
- **p0ss (mentor):** `REPLICATION_WRITEUP.md`, `scripts/capstone_*`, `results/capstone_replication/` — E4B replication, related-negative + graph-contrastive follow-ups.

## Artifacts by type
- **Index / status:** `MAP.md` (this) · `CAPSTONE_PROGRESS_SNAPSHOT.md` · `README.md`
- **Cross-cutting log:** `LESSONS_LEARNT.md`
- **iter 1:** `TIMELINE.md` · `archive/iteration1_relational_probing/`
- **iter 2:** `ITERATION2_TIMELINE.md` · `R4_PREREGISTRATION.md` · `R4_LAMBDA_RECIPE.md` · `WRITEUP.md` · `R4_results_explorer.ipynb`
- **iter 3 (Path 2):** `ITERATION3_TIMELINE.md` · `PATH2_PREREGISTRATION.md` · `PATH2_CONTROLS_NOTE.md` · `results/path2_*.json`
- **iter 4 (scaling):** `iteration4_scaling/` — `PRIMER.md` (concepts, start here) · `RUNBOOK.md` (step-by-step run) · `PREREGISTRATION.md` (locked) · `ITERATION4_results_explorer.ipynb` (STEP 4: decision rule + E4B overlay + figs) · `figures/` (presentation PNGs) · `results/iteration4_scaling/` · near-OOS add-on for p0ss: `scripts/generate_near_oos_controls.py` + `audit/audit_near_oos.py` → `data/near_oos_controls.json`
- **p0ss (replication):** `REPLICATION_WRITEUP.md` · `results/capstone_replication/` · `scripts/capstone_*`
- **near-OOS rejection:** `NEAR_OOS_RESULT_NOTE.md` (result + caption + bounds) · `scripts/near_oos_rejection.py` ·
  `scripts/render_near_oos_figure.py` · `results/near_oos_rejection/` (JSON + `fig8_fpr_by_face.png`) ·
  data/audit from 2026-06-11: `data/near_oos_controls.json`, `audit/near_oos_audit.json`
- **deliverable:** `paper/` — **`PAPER_DRAFT.md` is canonical**; `PAPER_DRAFT_voice.md` is parked and stale (carries
  a warning banner); `PAPER_OUTLINE.md` is the source map; `slides/`
- **shared machinery — DO NOT MOVE (both pipelines reference these paths):** `scripts/` `data/` `activations/` `audit/` `pivot/` `results/`

## Current state (one line)
**Near-OOS rejection test complete (2026-08-05): a second, deployment-side NULL.** Legitimate-influence passages
(Negotiation/Advertising/Lobbying) leak through the manipulation lenses at FPR 0.164/0.175/0.182 @95% TPR and *no*
negative scheme moves it (graph-sibling +0.010 p=0.86; model-mined +0.018 p=1.00) — while the same instrument
registers a −0.286 seal on the neighbour face (p<0.001), so the null is instrumented, not underpowered. It also
**reproduces §7's "you seal the faces you train against" rule on Gemma-2-9B** (siblings 0.318→0.032, mining
0.335→0.119), upgrading that rule from a single-model extension to a replicated finding. Note + figure:
`NEAR_OOS_RESULT_NOTE.md`, `results/near_oos_rejection/`. This closes the open item left for p0ss.

Prior: iteration 4 (2026-06-12) powered pre-registered NULL on alternate-path/Menger — held-out-edge flat-to-declining
{+.008/+.012/+.009/+.005/+.005} across 7→70/ctx, top-volume CI spans 0, MDE 0.014 shown; direct effect positive but
declining (+.061→+.040 — relational negatives are a *low-data lever*).

**Next = finish the paper.** `paper/PAPER_DRAFT.md` is canonical (decided 2026-08-05), carries every result, has all
8 figures wired and a completed numbers audit; §§1–4 have had the voice pass. Remaining: voice pass on §§5–10, and
tell p0ss (Jason, in hand).

## Conventions (going forward)
- New **code** → `scripts/` with a stage-clear name (`scaling_*`). New **results** → `results/<stage>/`. New **stage docs** → a `<stage>/` folder.
- **Pre-register** before any powered run; **adversarially verify positives**; **bound claims** honestly.
- The **final paper** (in `paper/`) *synthesises* the iterations — assembled at the end, not mid-flight. A one-pass tidy/archive of completed-iteration docs happens then.
