# 8-minute capstone talk — slide scaffold + speaker notes

*Audience: TARA participants (they know probes, residual streams, logistic regression from ARENA — they don't know
this project). 9 slides; per-slide budgets sum to 7:40, leaving 20s buffer. Each slide:
**SHOW** (what's on screen) / **SAY** (speaker-note draft — rewrite in your voice; the beats are what matter).
Slide 6 has the [ITER-4 SLOT]. Figures: `../../iteration4_scaling/figures/` (fig1–4 exist; fig5–7 export from
`ITERATION4_results_explorer.ipynb` after STEP 3).*

---

## 1. Title (0:15)
**SHOW:** "Knowing the Graph: does relational structure improve safety-concept probes?" — Jason Boudville.
One line: *"Four iterations, two models, independently replicated — and a bounded answer."*
**SAY:** This is a story about testing an appealing hypothesis carefully enough to trust the answer — including the
parts of the answer that are "no."

## 2. The question (1:00)
**SHOW:** A small concept graph (Manipulation — Disinformation — Propaganda — Sycophancy…) with edges highlighted.
The hypothesis in one sentence: *"Probes for related concepts should help each other — and the benefit should scale
with connectivity (alternate paths)."*
**SAY:** Safety monitoring wants per-concept detectors — lenses — over families of related behaviours. These
concepts aren't independent: manipulation, deception, sycophancy form a graph. The hypothesis we inherited from a
collaborating interpretability project: relationship-aware training improves detection, and if two concepts are connected by many independent
paths, you can triangulate their boundary even without direct data — like inferring a border from the neighbours'
borders. Intuitive, testable, and if true, very useful: graphs are cheap, data is not.

## 3. Setup in 30 seconds (0:35)
**SHOW:** fig1 (the conditions/pipeline figure). Bullets: 15 AI-safety concepts · curated 26-edge graph · passages
in 4 surface-distinct contexts · train on 3, test on the held-out context (OOD) · linear probes on layer-18
residual stream, Gemma-2-9B (+ E4B replication) · 30 seeds.
**SAY:** Everything is the ARENA toolkit you know: mean-pooled residual stream, logistic regression. The one design
choice to remember: out-of-distribution evaluation — train on three contexts, test on the fourth — because without
it, surface vocabulary saturates everything and every probe looks perfect.

## 4. Act 1 — the honest null (1:00)
**SHOW:** Timeline graphic: iter-1 "positive!" → struck through (noise) → iter-2 powered, pre-registered →
**partial Spearman −0.125 ≈ NULL** *(gloss on slide: "rank correlation between connectivity and probe benefit,
controlling for confounds — ≈ zero")*.
**SAY:** Iteration 1 looked positive — and died under adversarial verification: a noise artifact, plus we'd
operationalised "knowing the relations" as extra *positive* passages. Iteration 2 fixed validity and power — 105
matched pairs, decision rule locked before the run — and the answer was a clean null: connectivity does not predict
probe benefit. We pre-registered, so we report it as a result, not a failure. Hold that thought, because the null
gets *explained* in two slides.

## 5. Act 2 — the reframe that worked (1:15)
**SHOW:** Table, two models side-by-side (ΔF1 vs random negatives):
direct neighbours **+0.057 / +0.059**, graph-aware mixed +0.043 / +0.039, held-out-edge +0.008 / +0.012
(E4B = independent replication, Gemma-2-9B = this work). Separate labelled box — *"Path-2 controls (Gemma-2-9B):"*
placebo-graph p=0.002; "~⅓ of the effect is vocabulary."
**SAY:** Our collaborator's diagnosis: the mechanism isn't extra positives, it's *hard negatives* — train Manipulation's probe
with its actual siblings as the negative class. That works: +0.06 F1, and the declared graph beats shuffled placebo
graphs at p=0.002 — the curated edges are real confusability structure, not just "any graph." Two honest bounds:
about a third of the effect is shared vocabulary, and — see the right column — the effect lives entirely in
*direct* adjacency. Same numbers, two models, two codebases, fully independent runs.

## 6. Act 3 — connectivity isolated [ITER-4 SLOT] (1:15)
**SHOW:** fig5 — the headline sweep: held-out-edge ΔF1 vs data volume {7,14,28,56,70}/context (red), direct (blue),
with the independent E4B replication points overlaid hollow. Caption on slide: *"E4B = independent corroboration; the headline trend is
within-model (Gemma-2-9B)."* fig7 inset if space (per-edge strip at top volume).
**SAY:** The clean test of *connectivity* — as opposed to adjacency — is the held-out-edge condition: train with all
graph-aware negatives EXCEPT the sibling you're tested against. If alternate paths carry signal, they should
partially reconstruct that withheld boundary. They don't: ≈+0.01.
**[FILL after iteration 4 — NULL variant:]** And the obvious rebuttal — "you just need more data" — is what
iteration 4 killed: we scaled to 10× under a locked pre-registration, and the line stays flat. Say aloud only the
top-volume mean and CI [from the verdict cell]; the per-volume numbers and p-values live on the slide. Within-model,
pre-registered, and with the minimum detectable effect reported: the alternate-path prediction is refuted with
shown — not asserted — power.
**[FILL — CONFIRM variant:]** And iteration 4 surprised us: at 10× volume the held-out-edge delta rises —
[numbers] — connectivity was power-limited, and the next question is *which* edges recovered signal.

## 7. Why — the boundary-coverage picture (0:35)
**SHOW:** Cartoon: a centroid with boundary facets; nearby negative pushes a facet (labelled "+0.06"); far
constraint touches a different facet ("≈0"). Caption: *"Positives sharpen the centroid; independent negatives place
the facets. Our graph's 'alternate paths' all route through one hub — correlated, not independent."*
**SAY:** One model predicts everything you've seen: a probe's boundary is set by the negatives nearest each facet.
Extra positives — nothing (iterations 1–2). Sibling negatives — the facet you're tested on (+0.06). Held-out-edge —
the surviving constraints face the wrong way, and in our graph the alternate paths all pass through the
ManipulativeCommunication hub, so they're correlated. Menger counted paths; what matters is *independent*
constraints. **[If iteration 4 CONFIRMS instead, this slide's close becomes: "—and iteration 4 showed this model
undersold independence: some alternate paths do carry signal at volume. Finding which ones is the next
sub-analysis."]**

## 8. So what — the deployment recipe (1:00)
**SHOW:** The faces table (FPR@95%TPR — *gloss on slide: "false-positive rate at an operating point keeping 95% of
true positives"*): neighbours — graph seals 0.19→0.05 · off-graph confusers — graph 0.26 ≈ random 0.27, mined →
0.06 · far-OOD — "other" tier → 0.00. Title: *"You seal exactly the faces you train against."*
**SAY:** This cashes out practically. A flat 15-probe argmax has no "none of these" — it confidently labels a dog
passage as manipulation; a cheap OTHER class fixes that completely. And negative selection is a budget allocation:
graph siblings seal the neighbour face for free; the cheap "other" tier seals far-OOD; but the graph gives ~zero
protection against the model's *actual* off-graph confusers — only model-mining seals those, a 20-point FP cut. So:
cheap tiers everywhere, audit for the gap, spend mining only on the gap.

## 9. What this taught me + what's next (0:45)
**SHOW:** Three lines: *Pre-register before powered runs (saved us twice) · Adversarially verify your positives
(killed a false one) · A bounded answer + a mechanism beats an unbounded yes.* Next: the
contrast-prompted generation test; the near-out-of-set rejection test (controls already generated + audited);
counting independent constraints, not edges.
**SAY:** The meta-result for this room: iteration 1's false positive would have been a fun talk and wrong. The
discipline — pre-registration, adversarial verification, honest bounding — is what let two fully independent runs
converge on numbers we trust. The relationships turned out to matter not as classifiers but as a map: where to look next.
Thanks — questions.

---

## Delivery notes
- 8 minutes is unforgiving: slides 2, 5, 6 carry the talk; if running over, compress 3 and 7 (the cartoon can carry
  §7 in 20 seconds).
- Numbers to say out loud (rounded, consistent): **+0.06** local effect, **~⅓** lexical, **≈+0.01** held-out-edge,
  **p=0.002** placebo, **20-point** FP cut. Everything else lives on the slides.
- Rehearse the slide-6 fill BOTH ways until iteration 4 lands; the talk's spine works under either verdict.
- Likely Q&A: "is +0.06 even meaningful?" (it's ~the whole remaining headroom at a 0.93 ceiling; and the
  FPR-at-operating-point analysis is where deployment value shows) · "does this generalise beyond SET G?" (single
  family — honest limitation, §10 of the paper) · "why linear probes?" (deployment-realistic; MLP spot-checks
  matched) · "who ran the second model?" (an experienced collaborator who prefers to stay behind the scenes; the
  shared repo records full provenance — be straightforwardly honest here, just don't volunteer the name).
