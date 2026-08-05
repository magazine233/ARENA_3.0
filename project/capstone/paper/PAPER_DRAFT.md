# Knowing the Graph: Does Relational Structure Improve Linear-Probe Detection of Related AI-Safety Concepts?

*Jason Boudville. — FULL DRAFT: scaffold assembled 2026-06-11; **iteration-4 verdict landed 2026-06-12 and is
filled in throughout** (§5.4 = powered, pre-registered NULL; abstract and §8 resolved); **the §7 near-OOS rejection
test landed 2026-08-05 and is filled in throughout** (abstract, §6.1, §7, §8, §9, §10). This file is the canonical
draft (decided 2026-08-05); `PAPER_DRAFT_voice.md` is parked and does not carry the near-OOS edits. **The voice pass
is complete across §§1–10 (2026-08-05)** and all 8 figures are wired. What remains is Jason's own read-through — the
prose is in his register but it is not yet his — plus the two open items in the checklist below. (Second-model work credited as "peer replication" throughout — the crediting
they requested, confirmed 2026-06-12.)
Everything except §5.4 is determined by completed work; §5.4 has both pre-written outcome variants (legitimate
because the decision rule is locked in `../iteration4_scaling/PREREGISTRATION.md` §3 — we fill in the verdict, we
don't choose it). Numbers verified against the result artifacts named in `PAPER_OUTLINE.md`. Jason: every section
is yours to rewrite in your voice — the scaffold gets the structure, numbers, and bounded claims right.*

---

## Abstract

Concept-detection probes are a core interpretability tool, and AI-safety concept sets are naturally *relational* —
manipulation, deception, and sycophancy form a graph of siblings, parents, and hubs, not a list of independents. We
test whether knowing that graph improves linear-probe detection of 15 related AI-safety concepts in the residual
stream of Gemma-2-9B (replicated on Gemma-4-E4B). Across four iterations — every powered run pre-registered after
the first — with independent peer replication of every key result, we find a consistent, bounded answer. (1) Relational structure helps **only as local hard-negative
contrast**: training a concept's probe with its directly-related siblings as negatives improves discrimination by
~+0.06 F1, replicated across both models and both codebases — but roughly one third of that effect is shared
vocabulary, not deep structure. (2) The graph's *connectivity* shows no usable signal in the completed
runs: the held-out-edge test (can the other relationships reconstruct a withheld boundary?) yields ΔF1 ≈ +0.01 —
slightly positive but small and unreliable — at every layer, both models, and every data volume: a pre-registered,
within-model 10× scaling sweep (to 70 definitions/context) confirms the flat line — top-volume mean +0.005 with a
95% CI spanning zero, against a shown minimum detectable effect of 0.014 — so this graph's connectivity adds
nothing at any volume we could buy. We explain both results
with a boundary-coverage model in which positives sharpen a concept's centroid while *independent* contrastive
negatives place its decision-boundary facets — connectivity counts only insofar as it counts independent
constraints, and our graph's alternate paths route through a single hub. The practical payoff is a cost-tiered
recipe for deployable safety lenses — cheap graph-sibling negatives seal the neighbour face, a cheap "other" tier
seals far-out-of-distribution inputs completely, and expensive model-mined confusers are justified only on the
measured off-graph gap, where they cut false positives by 20 points — together with the gap that recipe does *not*
close. On a pre-committed near-out-of-set test, legitimate-influence passages (negotiation, advertising, lobbying)
leak through at FPR ≈0.23 on held-out context, and **no** negative scheme moves it (graph siblings +0.010, p=0.86;
model-mined +0.018, p=1.00, paired across 14 lenses) while the same instrument detects a −0.286 seal on the
neighbour face at p=0.0001. Every face this study can seal is a face it trained against; the hard rejection regime
stays open.

## 1. Introduction

Reach for an ontology and the promise is seductive: structure is free, data is not. If you already know that
manipulation borders deception, and deception borders propaganda, surely a detector can lean on that map — more data
per boundary, a scaffold to generalise along, the neighbours' borders standing in for the one you cannot see.
Deployable safety monitoring wants exactly this: per-concept lenses over a whole family of related behaviours (here,
the SET G manipulation / deception / influence family), built cheaply and trusted in production. So the question is
worth asking carefully, which is what this paper is about.

The specific hypothesis we inherited has a sharp form. Relationship-aware training improves discrimination, and the
benefit scales with graph *connectivity* — Menger's quantity, the number of vertex-disjoint paths between two
concepts. The intuition is that if A and B are joined by many independent routes through the graph, a probe should
triangulate their boundary even with no direct A-versus-B data. It is a clean, testable, and genuinely useful claim
if it holds.

It mostly does not — and saying so with confidence took four iterations. After the first one fooled us (more on that
below, because the way it fooled us is the methodological point of the paper), every powered run was pre-registered
before execution, every positive result was adversarially verified, and the whole study was independently
peer-replicated and extended on a second model by the researcher who originated the hypothesis. What we are left
with is not a yes or a no but a *bounded* claim with a mechanism behind it — and, at the end, a measured account of
where that mechanism stops paying.

Five things we want to foreground, because the discipline is the contribution as much as the result:

1. A valid, powered, pre-registered test of the relational hypothesis — built by fixing iteration 1's validity and
   power errors, rather than publishing them.
2. The controls that fence the surviving effect in: a vocabulary regression (~⅓ of it is lexical), a placebo-graph
   permutation (the declared graph is genuinely special, p=0.002), and the pre-registered scaling sweep of the peer
   replication's held-out-edge condition — the one test that cleanly separates connectivity from adjacency.
3. Cross-model convergence under peer replication: Gemma-2-9B (this work) and Gemma-4-E4B (peer replication), two
   codebases, the same numbers to two decimal places.
4. A reframing — from vertex connectivity to independent-constraint boundary coverage — that predicts the whole
   pattern of results, and the deployment recipe that falls out of it.
5. The boundary of that recipe, measured rather than guessed: a pre-committed rejection test on legitimate influence
   shows the scaffold seals the faces it trains against and leaves the hard regime wide open (§7).

## 2. Background & setup

The substrate is deliberately plain, so that anything we find belongs to the question and not to the machinery. SET G
is 15 AI-safety concepts from the manipulation / deception / influence family, wired together by a curated relational
graph (`SET_G_N15_graph_contrastive_boundary_v1`, 26 rationalised edges). Each concept gets a one-sentence
definition; Claude writes the passages, across four surface-distinct contexts — workplace, online, relationships,
marketplace — and we always evaluate out-of-distribution, training on three contexts and testing on the fourth. That
last choice is not bookkeeping; it is the whole reason the experiment has headroom. The v1 attempt skipped it, and
surface vocabulary promptly saturated every probe to AUC ≈ 1.0, leaving nothing to measure.

Probing is equally unglamorous: mean-pooled residual stream at layer 18 (Gemma-2-9B, d=3584; the E4B analogue is
`hidden_states[19]`), logistic regression run one-vs-rest or pairwise, 30 seeds. The layer-18 choice is not
load-bearing — a sweep over all 43 layer positions (§6.2) confirms it later. Menger's connectivity k(A,B), the count
of vertex-disjoint paths, is the quantity the inherited hypothesis says probe benefit should track.

A word on how the work was done, because it shapes how much you should trust it. Jason leads the iterations and this
paper; the study is independently replicated and extended on E4B by the peer replication, credited as such at their
request. The standing rules were three: pre-register before any powered run, adversarially verify every positive, and
bound every claim honestly. The first iteration is why those rules exist.

## 3. The honest null (iterations 1–2)

Iteration 1 looked positive. It was wrong.

It compared definitional against relational probing with a Menger correlation, multiclass, and it came back with the
result we were hoping for — until adversarial verification took it apart and found a noise artifact wearing the
costume of a signal. Worse, in hindsight, it had asked the wrong question entirely: it treated "knowing the
relations" as a licence to add more *positive* passages, which is not what the hypothesis was ever about. We logged
the lesson and rebuilt the design.

Iteration 2 (R4) did it properly: pair-matched, powered across 105 pairs, the decision rule locked before a single
number came back. The Menger correlation came in at a partial Spearman of **−0.125** — a clean, powered null on the
claim that connectivity predicts pairwise probe benefit. Because we had pre-registered, there was nowhere to hide it
and no reason to; it is reported as a result, not buried as a disappointment.

What survived into the next iteration was a single negative lesson, and it turned out to be the productive one: if
relational structure matters at all, it is not through more positive data strung along the graph.

## 4. The reframe: relations as hard negatives, and iteration 3 / Path 2

The peer replication supplied the diagnosis we had missed. The capstone had operationalised "knowing the relations"
as extra *positives*; the hypothesis's real mechanism uses related concepts as *hard negatives*. Contrast, not
coverage. You do not tell a probe more about what manipulation *is* — you show it the sibling it keeps being confused
with, and make that sibling the thing to push away from.

Iteration 3 / Path 2, pre-registered on Gemma-2-9B, put that to the test, and it held. Declared siblings really are
more confusable than non-siblings — a placebo-graph permutation pins this at **p=0.002**, so the curated graph is
genuinely special and not just an artifact of having *a* graph. The clean primary effect is **+0.055 F1** (the
sibling-vs-random confusability gap of Experiment A; the in-distribution training-benefit variant runs larger but is
inflated, so we do not headline it). And the honest discount: a vocabulary regression attributes about **a third** of
the effect to lexical overlap — an upper bound, in fact, since the regression over-controls when vocabulary itself
partly carries the relatedness. What is left, graph beyond vocabulary, is positive but marginal.

It replicates across models. The peer replication's graph-contrastive test on E4B (their design, their first run, 30
seeds) gives direct-neighbour negatives **+0.057**, graph-aware mixed **+0.043**, held-out-edge **+0.008**. Our
Gemma-2-9B baseline at 14 definitions per context lands at **+0.059 / +0.039 / +0.012**. Same pattern, two models,
two codebases.

## 5. Connectivity isolated: the held-out-edge test and the scaling sweep

### 5.1 The negative-set design (four conditions)
Everything turns on which concepts you hand the probe as negatives. For each directed edge — a target concept and one
held-out sibling — we train the target's probe four ways: against (a) count-matched random non-graph concepts, (b)
direct neighbours, (c) a graph-aware mixed set, and (d) graph-aware negatives that *exclude the very sibling being
evaluated*, the held-out-edge condition. That last one is the clean alternate-path test. If connectivity carries
transferable signal, the *other* relationships ought to reconstruct the boundary we withheld. **Fig. 1** shows what
the conditions mean on the concept graph itself.

### 5.2 The local effect is real; the connectivity effect is ≈0
Direct ~+0.06, mixed ~+0.04, held-out-edge ~+0.01 — both models, the Section 4 numbers, side by side in **Fig. 2**.
The benefit tracks how close a negative sits to the boundary being tested, not how many paths run to it. The graph
helps where it touches; one hop away, it stops.

### 5.3 Robustness (peer-replication extensions)
Two extensions close off the easy escape routes. A **layer sweep** reruns the conditions at layers 8, 18, 24 and 42:
the deltas reproduce at every depth, discrimination is near-ceiling everywhere (lexically separable from the
embedding layer up), and held-out-edge is dead throughout — layer 18 is representative, not cherry-picked. A **4×
volume** run on E4B (28/context) holds the held-out-edge mean flat (+0.008 → +0.010) while the spread of the paired
deltas tightens (std 0.106 → 0.065): more data made the estimate steadier without making it bigger. Even at 4×,
held-out-edge came out positive in only about 40% of comparisons — a coin nudged barely off fair.

### 5.4 Iteration 4: the powered scaling sweep — result: NULL under the locked rule

This is the one we built to settle the rebuttal that every null invites: *you simply didn't have enough data.*
Pre-registered and locked on 2026-06-10 — Gemma-2-9B, volumes {7, 14, 28, 56, 70} per context subsampled from a
single 70/context (10×) generation and extraction, 4,200 passages at 95% blind-audit self-match, held-out-edge ΔF1
versus volume as the primary endpoint, decision rule fixed in advance. Run on 2026-06-12, 30 seeds.

We bought ten times the data, and the line did not move.

Held-out-edge ΔF1 by volume came in at **+0.008, +0.012, +0.009, +0.005, +0.005** — no monotonic rise; if anything it
drifts *down* past 14/context (**Fig. 5**, the headline sweep with the E4B points overlaid; **Fig. 6**, reliability
against volume; **Fig. 7**, the per-edge picture at top volume). At the top volume, paired across the 52 directed
edges, the mean is **+0.0046** with a **95% CI of [−0.0053, +0.0144]** (paired-t p=0.36, Wilcoxon p=0.78) — the
interval spans zero, which is the pre-registered NULL criterion exactly. And the power is shown, not asserted: the
minimum detectable effect at 80% power was **ΔF1 = 0.014**, and the same instrument, at the same volume, registers
the direct-neighbour effect at roughly three times that (+0.040, CI [+0.021, +0.059], p=0.0001). The alternate-path /
Menger prediction — taking hop-count as the proxy, on this graph — is refuted at adequate power. Relational benefit
is local direct-neighbour contrast, and nothing more. It converges with the peer replication's 4× E4B result
(+0.010, flat) and closes the arc the iteration-2 null opened: two nulls and one bounded positive, all of a piece
under the model in §6.1.

On the pre-registered vocabulary control at scale (§4): the gross held-out-edge result is already ≈0, and shared
vocabulary among graph-relatives could only have *helped* reconstruct the withheld boundary — so the
net-of-vocabulary reading is, a fortiori, ≈0. (The full regression on the scaled data is deferred; the direction of
the bias makes it non-load-bearing here.)

One secondary result is worth more than its pre-declared footnote. The direct-neighbour effect stays clearly positive
at every volume — but it *shrinks* as data grows, +0.061 down to +0.040 from 7 to 70/context. The value of a smart
negative is largest precisely when data is scarce, which is exactly the regime a long-tail safety concept lives in.
The boundary-coverage model predicts this: more data sharpens the random-negative baseline too, and the premium you
pay for cleverness erodes as the floor rises.

## 6. Why: from vertex connectivity to independent-constraint boundary coverage

### 6.1 The model (from the peer replication; REPLICATION_WRITEUP §6)
Picture a concept as a centroid with a cloud around it, and the probe's decision boundary sitting where that cloud
thins out. Positives pull the centroid into focus. Negatives do something different — each one presses on a single
*facet* of the boundary, the face that points toward it. That one picture predicts everything we saw. Extra positives
do nothing once the centroid has saturated (the iterations 1–2 null). A sibling negative moves the boundary because
it leans on the facet you are testing (+0.06, iteration 3). The benefit orders itself by proximity — direct beats
mixed beats held-out — because a far constraint presses on a different face. And held-out-edge sits at ≈0 *here* for
a concrete reason: SET G's nominally disjoint alternate paths run through the ManipulativeCommunication hub, so they
are correlated, not independent, and correlated constraints cannot pin a boundary they do not face. Menger counted
paths. What a boundary actually needs is *independent* constraints, and k matters only to the extent that it counts
them.

The model also made a prediction that was then tested rather than fitted, which is the harder thing for a
post-hoc story to do. Near-out-of-set inputs are simply another facet, and no scheme in this study ever placed a
constraint on that one — so the model says it should leak, whatever negatives you choose. §7 went and measured it. It
does, at ≈0.23, flat across all three schemes. The size of the leak is now measured instead of assumed.

### 6.2 The geometry is consistent (with an anisotropy correction) [peer replication]
Raw cosine first told a misleading story — everything looked like it overlapped (adjacent pairs 0.971, non-adjacent
0.961), which is the anisotropy of the space talking (random passage pairs already sit at ~0.89). Mean-centre it and
the picture sharpens: adjacent pairs separate by +0.166, far-domain controls fall clearly outside (−0.147), but
*graded* graph distance still refuses to track representation (r ≈ −0.30 at layer 18; −0.27 to −0.39, median −0.34,
across all 43 layer positions). The geometry tells the same story the probes do — local adjacency is real, multi-hop
structure isn't — and the probe numbers are untouched by the correction anyway, since a linear classifier divides out
the common direction. The structure was never hidden. The raw metric was.

### 6.3 Generation mode is a non-factor (Path 3 first run) [peer replication]
We checked whether the model's *generating* state differs from its *reading* state in a way the probes could exploit.
It doesn't: true autoregressive states and teacher-forced re-reads graduate at 0.692 versus 0.693, near-identical, as
causal equivalence would predict; the read-own gap (0.454) is prompt-grounding, not a generation effect. The
contrast-*prompted* variant — asking the model to write A in a way that is explicitly *not* B — is still untested,
and is the natural next move, since Path 2 already shows the contrast-as-negative mechanism is the one that pays.

## 7. The deployment payoff: rejection scaffolds and tiered negatives

*(The two-level probe, multi-scale negatives and confuser-coverage results are peer-replication extensions on
Gemma-4-E4B; the near-OOS controls and the rejection test are this work, on Gemma-2-9B.)*

The flat probe's real weakness is not telling concepts apart — it is not knowing when to keep its mouth shut. A
15-way argmax assigns 100% of far-domain inputs to *some* concept, confidently calling a passage about the weather a
form of manipulation; a confidence threshold rejects only 26% of them. Add a coarse OTHER class trained on cheap
far-domain controls and out-of-family inputs are rejected at recall 1.0 with zero false rejection of real concepts.
Note what *doesn't* fix it: hierarchical *classification* actively hurts (flat 0.910 > soft two-level 0.857 > hard
0.850). The scaffold's job is rejection; resolution stays with the flat lens.

The governing rule, once you measure it, is blunt: **you seal exactly the faces you train against.** Holding the
negative count fixed and reading false-positive rate at a 95%-true-positive operating point, graph-sibling negatives
seal the neighbour face (0.192 → 0.046) but do no better than random on off-graph confusers (0.257 against random's
0.270); model-mined confusers seal *that* face (→ 0.058, a twenty-point cut) but let the others slide; and a cheap
"other" tier shuts the far-OOD face completely (→ 0.000). The graph quietly misses the hub-attractor confusions — a
majority of actual misclassifications (24 of 42, on a small base) fall on non-edges, so it is the shape of the miss,
not the exact count, that matters. **The rule now replicates across models and codebases:** on Gemma-2-9B (this work,
layer 18) siblings seal the neighbour face 0.318 → 0.032 and mining seals the off-graph face 0.335 → 0.119 —
different absolute values, identical structure, each scheme abandoning the face it does not train (**Fig. 8A**).

So the recipe is straightforward. Cheap tiers everywhere — graph siblings plus an "other" class — then a cheap audit
to find the off-graph gap, then expensive mining spent only on that gap. One caution, and it is not hypothetical:
calibration drift is a live failure mode. Thresholds set on training positives collapsed to 34% held-out true-positive
rate, and in production there is no held-out label to catch it before it costs you.

That is where the recipe stops, and we went looking for the edge deliberately. The far controls are the easy
rejection regime — a passage about basketball is never going to read as gaslighting. The hard regime is the
legitimate neighbour: influence that is honest but adjacent. We generated and blind-audited that set — Negotiation,
Advertising, Lobbying, 84 passages, 84 of 84 clean on a blind self-match with zero drift into the in-set concepts —
and quarantined the PersuasiveCommunication lens by hand, because its own training data contains honest
advertisements, so a fire there is a true positive by construction rather than a leak. The analysis was pre-committed
before the run: score the other 14 lenses, report FPR both pooled and split by whether a passage's context was held
out of the scoring lens's training fold.

**Nothing seals that face** (**Fig. 8**). Legitimate-influence passages leak at 0.164 / 0.175 / 0.182 under random,
graph-sibling and model-mined negatives respectively, and paired across the 14 scored lenses no scheme improves on
random at all: graph-sibling **+0.010**, 95% CI [−0.074, +0.095], p=0.86; model-mined **+0.018**, CI [−0.101,
+0.136], p=1.00. The null is instrumented, not asserted — the same lenses, the same thresholds and the same paired
test register a **−0.286** seal on the neighbour face (p<0.001) and **−0.216** off-graph (p=0.006), so a seal of
that size here would have been unmissable. The pre-committed context split earns its keep: leakage is significantly
worse when the passage's context was held out of the lens's training fold (0.225 against 0.158, +0.068, p=0.016), so
the honest out-of-distribution figure is **≈0.23 — roughly one honest-influence passage in four** — not the
flattering pooled 0.16.

And the leak is concentrated, not diffuse, which makes it actionable. **UnwarrantedPraise (0.589) and
ExcessiveAgreement (0.497)** carry most of it while four lenses sit at 0.000: honest advertising reads as unwarranted
praise, cooperative negotiation as excessive agreement. Those are precisely the two concepts iteration 2's audit
already flagged as definitionally umbrella-ish, which is a satisfying place for a deployment failure to land. The
quarantine was vindicated by the data rather than justified after the fact — PersuasiveCommunication fires at
0.238–0.523, well clear of every scored lens — and calibration drift reproduces in direction though more mildly than
on E4B (train-calibrated thresholds drop held-out TPR to 0.75–0.83, not 0.34). *One scope note: `multi_scale` is not
run on this side. It trains against far-domain controls and only one usable far set exists here, so training and
evaluating on it would leak; far-OOD is evaluation-only in this run and its rate is not comparable to the sealed
0.000 above.*

The practical upshot is a sentence a deployment team can act on. The recipe buys you the easy rejection regime and
the two faces you can name in advance; it does not buy the near-out-of-set face, which is the one a real monitor
meets most often, because legitimate influence is far commoner in the wild than either far-domain noise or textbook
manipulation. Until that face is trained against explicitly, assume the monitor fires on about one honest-influence
input in four — and assume it does so through two specific lenses.

## 8. Synthesis & conclusion

So here is what relational structure is worth, measured rather than assumed. Direct relations make the best cheap
hard negatives — about +0.06 F1, a third of it lexical, holding across two models and an independent replication —
and nothing further flows through the graph's connectivity at any data volume we could buy (the 10× sweep's
top-volume interval spans zero against a minimum detectable effect of 0.014). The grand version of the hypothesis,
the one where the graph's topology does the heavy lifting, is not what the powered test found.

But the relationships were not the wrong thing to study — only the wrong thing to expect *discrimination* from. Their
real job, under the boundary-coverage picture, is not classification but navigation: a map of where to look next, the
edges marking the routes out from the concepts you have mapped toward the confusers the model has and you have not
named yet.

And the recipe that falls out of all this comes with its own boundary, which we measured rather than guessed. It
seals the faces it trains against. It leaves the hard rejection regime — honest influence, the input a real monitor
actually meets — open at roughly one passage in four, through two nameable lenses. That is the most useful sentence
in the paper for anyone planning to deploy one of these things, and we would not have it if we had stopped at the
result that flattered the method.

A negative result, but a bounded one — with a mechanism that predicts it, a recipe you can deploy, and an honest map
of where that recipe runs out. We will take that over an unbounded yes.

## 9. Methods (the rigor section)

Every powered run after iteration 1 was pre-registered with its decision rule locked in advance (R4, Path 2,
iteration 4, and the §7 rejection test). F1 was the pre-registered primary metric throughout; balanced accuracy and
AUC give the same ordering in every table, with AUC the most ceiling-compressed of the three. Evaluation is
out-of-distribution by held-out context. The controls are count-matched negatives, placebo-graph permutation,
vocabulary regression, held-out-edge isolation, and anisotropy-corrected geometry. Every positive result was
adversarially verified by independent refutation passes — the mechanism that caught iteration 1's noise-positive
before it could become a finding. The whole study was cross-model peer-replicated, and all data, code and result
JSONs are committed with provenance.

One rule is worth stating on its own, because it is what makes three nulls readable. **Every null in this paper ships
with a positive control on the same instrument** — the §5.4 direct-neighbour effect standing behind the held-out-edge
null, the §7 neighbour and off-graph seals standing behind the near-OOS null. A null measured by an instrument that
has just been shown to detect a comparable effect is a different object from a null measured by an instrument of
unknown sensitivity. It keeps "we did not detect it" separable from "we could not have detected it", which is the
distinction iteration 1 failed and everything after it was built to respect.

## Acknowledgements

The second-model results throughout are an independent **peer replication** — the crediting requested by the
researcher who performed it (confirmed 2026-06-12). We thank them for the hard-negative reframe, the E4B
replication, the layer/anisotropy/two-level/multi-scale extensions, and the boundary-coverage model. The shared
repository records the full provenance of each result.

## 10. Limitations & future work

The honest fences. This is one concept family (the SET G manipulation / deception cluster) and one probe family
(linear / logistic, with MLP spot-checks), over a curated human graph — though §6.2 shows that graph's local layer is
real in the model, not imposed on it. Discrimination runs near ceiling, which compresses the headroom for *any*
manipulation of the negatives, so the deltas are small by construction; the sensitive instrument here is the
false-positive rate at an operating point (§7), not pairwise AUC.

The §7 result carries a bound of its own. It rests on 84 passages across three control concepts, seven per context.
That is adequate for the headline null — the intervals are tight around zero and the positive controls fire hard —
but the per-control breakdown (Advertising 0.190, Lobbying 0.179, Negotiation 0.155) sits within noise and should not
be read as a ranking of which honest activity trips a manipulation lens hardest. Answering *that* would need a
larger control set, and it is cheap to build.

Several things remain untested by design: contrast-prompted self-generation (the Path 3 graph-aware variant) feeding
a graduation loop; a topology retest that counts *independent* constraints rather than raw connectivity; and
activation-cell cartography — finding the dark, unmapped regions of the model's own representation — as the
principled way to grow coverage. The most immediate of them is the direct follow-on from §7: whether training
*against* near-OOS negatives seals that face without costing the others. The boundary-coverage model says it should,
and unlike most of what is listed here, it needs one generation run and an afternoon.

---

## Figures (manifest — all committed, all regenerable from committed JSONs)

| # | File | Cited | What it carries |
|---|---|---|---|
| 1 | `../iteration4_scaling/figures/fig1_conditions.png` | §5.2 | What the negative-set conditions mean on the concept graph |
| 2 | `../iteration4_scaling/figures/fig2_baseline_bars.png` | §5.2 | F1 gain by condition, Gemma-2-9B vs E4B side by side |
| 3 | `../iteration4_scaling/figures/fig3_volume_curve.png` | §5.4 (alt) | Held-out-edge gain stays flat as volume grows |
| 4 | `../iteration4_scaling/figures/fig4_conclusion.png` | talk only | One-slide takeaway; not needed in the paper |
| 5 | `../iteration4_scaling/figures/fig5_sweep_with_e4b_overlay.png` | §5.4 | **Headline sweep**, E4B points overlaid |
| 6 | `../iteration4_scaling/figures/fig6_reliability_vs_volume.png` | §5.4 | Scale buys reliability, not magnitude |
| 7 | `../iteration4_scaling/figures/fig7_per_edge_topvolume.png` | §5.4 | Per-edge deltas at top volume |
| 8 | `../results/near_oos_rejection/fig8_fpr_by_face.png` | §7 | **FPR by face × scheme**; near-OOS sealed by nothing |

Rebuild: figs 1–7 `python iteration4_scaling/figures/render_figures.py`; fig 8
`python scripts/render_near_oos_figure.py`. Captions for 1–7 in
`../iteration4_scaling/figures/CAPTIONS.md`, for 8 in `../NEAR_OOS_RESULT_NOTE.md`.

**Fig. 5 caption must state (pre-reg §4):** "E4B points are independent corroboration; the headline trend is
within-model (Gemma-2-9B)."

---

## Assembly checklist (delete before submission)

- [x] §5.4 filled (2026-06-12, NULL verdict from `ITERATION4_results_explorer.ipynb`); variants deleted; MDE and
      vocab-regression a-fortiori note included.
- [x] Abstract + §8 brackets resolved (NULL).
- [x] §7 near-OOS rejection test run (2026-08-05) — result folded into the abstract, §6.1, §7, §8, §9, §10.
      Source: `../NEAR_OOS_RESULT_NOTE.md`, `../results/near_oos_rejection/near_oos_rejection.json`.
- [x] §7 figure built (2026-08-05): **Fig. 8**, `../results/near_oos_rejection/fig8_fpr_by_face.png` — panel A
      levels by face × scheme, panel B paired effects with 95% CIs (positive controls clear zero, near-OOS
      straddles it). Caption in `../NEAR_OOS_RESULT_NOTE.md`; rebuild via `scripts/render_near_oos_figure.py`.
- [ ] Decide whether `PAPER_DRAFT_voice.md` is retired or brought forward — it does **not** carry any of the
      2026-08-05 near-OOS edits and is now materially out of date.
- [x] Figures wired (2026-08-05) — manifest above; figs 1/2 cited in §5.2, 5/6/7 in §5.4, 8 in §7. **No Path-2
      placebo or vocabulary figure exists** (only the JSONs); build one or drop the idea — it is not currently
      referenced anywhere.
- [x] Numbers audit complete (2026-08-05). §5.4 and all §7 near-OOS numbers verified earlier; §§4, 6.2, 6.3 and the
      peer-replication half of §7 verified against source JSONs this pass. **Everything matched except one fix
      applied here:** §6.2's graded-distance correlation range was "−0.27 to −0.40 across all 42 layers"; the
      committed `layer_sweep_geometry.json` gives −0.2744 to −0.3949 (median −0.3388) over 43 layer positions, so
      the text now reads "−0.27 to −0.39, median −0.34, across all 43 layer positions".
      **One provenance caveat:** §7's "thresholds set on train positives collapsed to 34% held-out TPR" is
      documented in `../REPLICATION_WRITEUP.md` prose, not in any committed JSON — cite it as the peer
      replication's reported figure, or ask them to emit the artifact.
- [x] Jason's voice pass on §§1–4 (2026-08-05), ported from the parked `PAPER_DRAFT_voice.md` and updated for the
      near-OOS result (contribution 5 added to §1; §2's layer-sweep count corrected to 43).
- [x] Voice pass on §§5–10 (2026-08-05). §§5.1–5.4, 6.1–6.3, 7, 8, 9, 10 converted from scaffold bullets to
      continuous prose; §7's five bullets became a single argued section that ends on a sentence a deployment team
      can act on; §9 now states the positive-control-with-every-null rule explicitly. Every number re-verified
      against the JSONs after the rewrite.
- [ ] **Jason's own read-through.** The prose is *in* your register — it was ported from your voice draft — but you
      have not read a word of it. The peaks are at: §3's "It was wrong.", §5.4's "We bought ten times the data, and
      the line did not move.", §6.2's "The structure was never hidden. The raw metric was.", and §8's close. If any
      of them feels forced, flatten it; a paper punishes forte harder than a blog does.
- [ ] Decide the 34% provenance question (§7) with p0ss, and build-or-drop a Path-2 figure.
