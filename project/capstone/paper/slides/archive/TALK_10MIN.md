# 10-minute capstone talk — VOICE PASS (spoken script)

*The ~10-minute cut of `TALK_8MIN_voice.md`. Same voice, same numbers, same claims — **two slides added** to the
8-minute version, both genuinely additive (not padding): (3) HatCat, the programme this hypothesis comes from, which
also sets up the close; (9) the geometry near-miss, the strongest non-probe evidence for the mechanism AND the second
"we nearly fooled ourselves" rigor beat. Built by `build_deck_10min.py` → `TALK_10MIN.pptx`. 11 slides, ~9:35 +
buffer. The 8-minute voice deck (`TALK_8MIN_voice.*`) is kept intact as the shorter option.*

*Budget: 0:15 + 0:55 + 1:15 + 0:30 + 1:00 + 1:10 + 1:15 + 0:35 + 0:55 + 0:55 + 0:50 ≈ 9:35, ~25s buffer.*

---

## 1. Title (0:15)
**SHOW:** "Does relational structure improve safety-concept probes?" — Jason Boudville.
One line: *"Four iterations, two models, independently replicated — and a bounded answer."*
**SAY:** Here's the short version, so you know where we're headed. We took an idea that sounds obviously true, and we
tested it hard enough to trust the answer — including the part of the answer that turned out to be no.

## 2. The question (0:55)
**SHOW:** the SET G graph (hub emphasised). One line: *"Graphs are cheap. Data is not."*
**SAY:** Safety monitoring wants a detector for each behaviour we care about — a lens for manipulation, one for
deception, one for sycophancy. But these concepts aren't strangers to each other. They overlap, they blur at the
edges, they form a graph. So here's the idea: you lean on that graph. You train each lens against its relatives. And
where two concepts are joined by lots of independent paths, you pin the boundary between them without ever showing
the probe that boundary directly. It's the appealing kind of idea. Structure is free. Data isn't.

## 3. The bigger picture — HatCat (1:15) [NEW]
**SHOW:** bullets — *HatCat: building a navigable map of the concepts a model represents* · *an atlas of per-concept
"lenses," wired together by a relational graph* · *the bet: know how concepts relate → better lenses + cheaper
coverage* · big accent line: *"Does the graph actually help — and does connectivity scale?"*
**SAY:** A word on where this comes from, because it's the whole reason the question matters. It isn't ours. It sits
inside a programme called HatCat — an attempt to build something like a map of the concepts a model carries in its
head. Not one detector bolted on the side, but a whole atlas of them: a lens per concept, and a graph wiring the
related ones together. The bet is simple, and ambitious. Know how the concepts relate, and you can build better
detectors — you train each one against its neighbours. And you cover the space more cheaply. The graph tells you
where to look. The connections let you reach concepts you've got no direct data for. That last part — reaching a
concept through its connections — is the boldest claim. It's the one we set out to test, in its sharpest, most
falsifiable form. Does the graph actually earn its keep? And does the connectivity scale?

## 4. Setup in 30 seconds (0:30)
**SHOW:** fig1. Bullets: 15 concepts · 26-edge graph · 4 contexts · train-3-test-1 (OOD) · layer-18 linear probes ·
Gemma-2-9B (+ E4B replication) · 30 seeds.
**SAY:** Thirty seconds of setup, all of it the ARENA toolkit. Mean-pooled residual stream, logistic regression on
top. The one choice that matters: we always test out of distribution — train the probe in three settings, test it in
a fourth it's never seen. Skip that and surface vocabulary does all the work, every probe scores perfect, and you've
measured nothing.

## 5. Act 1 — the honest null (1:00)
**SHOW:** the iter-1 → iter-2 timeline (iter-1 struck through; iter-2 → partial Spearman −0.125 ≈ NULL).
**SAY:** Iteration one looked positive. It was wrong. We pulled it apart and found noise, not signal — and worse,
we'd asked the wrong question, feeding the probe more examples of each concept instead of teaching it the boundary.
So we rebuilt it properly. Iteration two: a hundred and five matched pairs, powered, the decision rule locked before
we looked. The answer came back a clean null — connectivity does not predict how much a probe improves. Because we'd
pre-registered, there was nowhere to hide it and no reason to. Hold onto that null. A few slides from now it stops
being a disappointment and becomes the point.

## 6. Act 2 — the reframe that worked (1:10)
**SHOW:** two-model table (direct +0.057/+0.059, mixed +0.043/+0.039, held-out-edge +0.008/+0.012); boxed Path-2
controls — placebo p=0.002, ~⅓ vocabulary.
**SAY:** Here's what we'd missed. The fix isn't more examples of a concept — it's the right negatives. Train
manipulation's probe against its actual siblings, the concepts it keeps getting confused with, and discrimination
jumps about six points of F1. And the graph earns its keep: shuffle the edges into a fake graph and the effect
vanishes — p of nought-point-zero-zero-two — so these particular relationships are real, not just any old structure.
Two honest caveats, because that's the job. About a third of the lift is shared vocabulary, not deep structure. And —
this is the column on the right — all of it lives in direct neighbours. Same numbers, two models, two codebases, run
independently.

## 7. Act 3 — connectivity isolated: the pre-registered NULL (1:15)
**SHOW:** fig5 (the headline sweep, E4B overlaid hollow). Bullets land on: *"They don't — at any volume"* /
*"top volume +0.005, CI spans zero"* / *"VERDICT: pre-registered NULL."*
**SAY:** Now the clean test of connectivity, separated from mere adjacency. Train with all the graph's negatives
except the very sibling you're being tested on. If the graph's deeper structure carries anything, those other
relationships should reconstruct the boundary you held back. They don't — about a hundredth of a point. And the
obvious comeback is "you just didn't have enough data." So we settled it. We bought ten times the data,
pre-registered, one model start to finish. The line didn't move. At the top volume it's five-thousandths of a point,
with a confidence interval sitting squarely across zero — say just that. And before anyone asks whether we had the
power to see an effect: the design could catch one of fourteen-thousandths, and it picks up the direct effect — three
times that — without breaking a sweat. The instrument works. The thing we were testing just isn't there.

## 8. Why — the boundary-coverage picture (0:35)
**SHOW:** the boundary-coverage cartoon (near negative places the facet; far constraint points the wrong way).
**SAY:** One picture explains all of it. A probe's boundary is set by whichever negative sits closest to it. More
examples of the concept — nothing; the centre's already sharp. A sibling negative — it leans right on the face
you're testing, so it helps. And the held-out-edge stays at zero because in our graph the alternate routes run back
through the same hub — manipulation — so those paths aren't independent. They're the same information wearing
different hats. Menger counted paths. What a boundary actually needs is independent constraints.

## 9. A second time we nearly fooled ourselves (0:55) [NEW]
**SHOW:** before/after — *raw cosine: adjacent 0.971 vs non-adjacent 0.961 → "the graph isn't even in the model?"*
vs *mean-centred: adjacent separate +0.166, far controls −0.147, but graded distance still flat (r ≈ −0.30)*.
**SAY:** Quick aside, because it's the kind of thing that should make you trust the rest. When we looked at the raw
geometry — just the cosine between concept clusters — it said everything overlapped. Adjacent concepts at
point-nine-seven, non-adjacent at point-nine-six. For a moment it looked like the graph wasn't even in the model. But
that flatness is an artifact: transformer activations get squashed into a narrow cone, so everything looks similar to
everything. Subtract that out, and the structure snaps back — related concepts pull apart, and far-away ones, dogs
and weather, fall right outside the cluster. Same story the probes told: the local structure is real, the long-range
structure isn't. That's twice now the raw data nearly lied to us, and twice a control caught it. That's the job.

## 10. So what — the deployment recipe (0:55)
**SHOW:** the faces table (FPR@95%TPR by negative scheme). Title: *"You seal exactly the faces you train against."*
**SAY:** This pays off in something you can actually build. A flat fifteen-way probe has no "none of these" — show it
a passage about a dog and it'll still confidently call it manipulation. Add one cheap reject class and that problem's
gone. After that, picking negatives is just budget. Graph siblings seal the neighbour face for nothing. A cheap
"other" tier seals the far-away stuff completely. But the graph gives you almost no protection against the model's
real confusers — the ones it actually trips on — and only the expensive option, mining the model's own confusions,
seals those: a twenty-point cut in false positives. So: cheap everywhere, find the gap, spend the expensive money
only on the gap.

## 11. What this taught me + what's next (0:50)
**SHOW:** three lines (pre-register / verify your positives / bounded answer + mechanism beats unbounded yes); a
"Next" line.
**SAY:** What I'll take from this. Twice the data nearly fooled us — iteration one's noise, then the geometry
artifact — and both times the discipline caught it: pre-register, try to kill your own positives, bound what you
claim. That's the only reason I can stand here and trust these numbers. And the relationships — HatCat's whole bet?
They turned out not to be a way of telling concepts apart at all. They're a map — of where to look next. Which is
exactly what HatCat set out to build. Thanks. Happy to take questions.

---

## Delivery notes
- ~10 minutes. The two added slides (3 HatCat, 9 geometry) are the new material; everything else carries over from
  the 8-minute voice cut, lightly re-budgeted. If you get squeezed back to 8, drop slides 3 and 9 — the talk still
  stands (that IS the 8-minute version).
- The beats to actually *land* (slow down): "It was wrong." (5) · "The line didn't move." (7) · "Structure is free.
  Data isn't." (2) · "twice a control caught it. That's the job." (9) · "a map — of where to look next." (11).
- HatCat (3) and the close (11) bookend: open by naming HatCat's cartography bet, close by handing the result back to
  it. Let the audience feel the loop.
- Numbers to say aloud, rounded: **six points** (local), **a third** lexical, **a hundredth of a point**
  held-out-edge, **p = 0.002** placebo, **twenty-point** FP cut; geometry: **point-nine-seven vs point-nine-six raw**.
- Slide 9 trap: the figure shows all the geometry numbers, but say ONLY "point-nine-seven vs point-nine-six" — let
  the picture carry the rest (it's the before/after *shape* that lands, not the digits). Reading them all = blown 55s.
- Q&A unchanged from the 8-min notes (incl. "who ran the second model?" → peer replication, credited as they asked).
