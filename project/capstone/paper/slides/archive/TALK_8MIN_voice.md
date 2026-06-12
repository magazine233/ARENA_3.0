# 8-minute capstone talk — VOICE PASS (spoken script)

*Parallel to `TALK_8MIN.md`. Same slides, same numbers, same claims — the **speaker notes are reworked toward
Jason's spoken voice** (dynamic range: plain, conversational delivery, with technique spiking only at four beats —
the open, the iteration-1 confession, the flat-line, the close). On-slide text stays terse; the voice lives in what
you SAY. Built by `build_deck_voice.py` → `TALK_8MIN_voice.pptx`. The title is plain (the manufactured colon-clause
was the style guide's named tell — same fix as the paper voice pass). Audience: TARA participants; they know probes,
residual streams, logistic regression. 9 slides, ~7:40.*

---

## 1. Title (0:15)
**SHOW:** "Does relational structure improve safety-concept probes?" — Jason Boudville.
One line: *"Four iterations, two models, independently replicated — and a bounded answer."*
**SAY:** Here's the short version, so you know where we're headed. We took an idea that sounds obviously true, and we
tested it hard enough to trust the answer — including the part of the answer that turned out to be no.

## 2. The question (1:00)
**SHOW:** the SET G graph (hub emphasised). One line: *"Graphs are cheap. Data is not."*
**SAY:** Safety monitoring wants a detector for each behaviour we care about — a lens for manipulation, one for
deception, one for sycophancy. But these concepts aren't strangers to each other. They overlap, they blur at the
edges, they form a graph. So here's the idea we inherited — and someone later checked it on a second model. You lean
on that graph. You train each lens against its relatives. And where two concepts are joined by lots of independent
paths, you pin the boundary between them without ever showing the probe that boundary directly. It's the appealing
kind of idea. Structure is free. Data isn't.

## 3. Setup in 30 seconds (0:35)
**SHOW:** fig1 (conditions/pipeline). Bullets: 15 concepts · 26-edge graph · 4 contexts · train-3-test-1 (OOD) ·
layer-18 linear probes · Gemma-2-9B (+ E4B replication) · 30 seeds.
**SAY:** Thirty seconds of setup, all of it the ARENA toolkit. Mean-pooled residual stream, logistic regression on
top. The one choice that matters: we always test out of distribution — train the probe in three settings, test it in
a fourth it's never seen. Skip that and surface vocabulary does all the work, every probe scores perfect, and you've
measured nothing.

## 4. Act 1 — the honest null (1:00)
**SHOW:** the iter-1 → iter-2 timeline (iter-1 struck through; iter-2 → partial Spearman −0.125 ≈ NULL).
**SAY:** Iteration one looked positive. It was wrong. We pulled it apart and found noise, not signal — and worse,
we'd asked the wrong question, feeding the probe more examples of each concept instead of teaching it the boundary.
So we rebuilt it properly. Iteration two: a hundred and five matched pairs, powered, the decision rule locked before
we looked. The answer came back a clean null — connectivity does not predict how much a probe improves. Because we'd
pre-registered, there was nowhere to hide it and no reason to. Hold onto that null. Two slides from now it stops
being a disappointment and becomes the point.

## 5. Act 2 — the reframe that worked (1:15)
**SHOW:** two-model table (direct +0.057/+0.059, mixed +0.043/+0.039, held-out-edge +0.008/+0.012); boxed Path-2
controls — placebo p=0.002, ~⅓ vocabulary.
**SAY:** Here's what we'd missed. The fix isn't more examples of a concept — it's the right negatives. Train
manipulation's probe against its actual siblings, the concepts it keeps getting confused with, and discrimination
jumps about six points of F1. And the graph earns its keep: shuffle the edges into a fake graph and the effect
vanishes — p of nought-point-zero-zero-two — so these particular relationships are real, not just any old structure.
Two honest caveats, because that's the job. About a third of the lift is shared vocabulary, not deep structure. And —
this is the column on the right — all of it lives in direct neighbours. Same numbers, two models, two codebases, run
independently.

## 6. Act 3 — connectivity isolated: the pre-registered NULL (1:15)
**SHOW:** fig5 (the headline sweep, E4B overlaid hollow). Bullets land on: *"They don't — at any volume"* /
*"top volume +0.005, CI spans zero"* / *"VERDICT: pre-registered NULL."*
**SAY:** Now the clean test of connectivity, separated from mere adjacency. Train with all the graph's negatives
except the very sibling you're being tested on. If the graph's deeper structure carries anything, those other
relationships should reconstruct the boundary you held back. They don't — about a hundredth of a point. And the
obvious comeback is "you just didn't have enough data." So we settled it. We bought ten times the data, pre-registered,
one model start to finish. The line didn't move. At the top volume it's five-thousandths of a point, with a
confidence interval sitting squarely across zero — say just that. And before anyone asks whether we had the power to
see an effect: the design could catch one of fourteen-thousandths, and it picks up the direct effect — three times
that — without breaking a sweat. The instrument works. The thing we were testing just isn't there.

## 7. Why — the boundary-coverage picture (0:35)
**SHOW:** the boundary-coverage cartoon (near negative places the facet; far constraint points the wrong way).
**SAY:** One picture explains all of it. A probe's boundary is set by whichever negative sits closest to it. More
examples of the concept — nothing; the centre's already sharp. A sibling negative — it leans right on the face
you're testing, so it helps. And the held-out-edge stays at zero because in our graph the alternate routes run back
through the same hub — manipulation — so those paths aren't independent. They're the same information wearing
different hats. Menger counted paths. What a boundary actually needs is independent constraints. *(If time: the
direct effect itself shrinks as data grows, +0.061 down to +0.040 — so smart negatives matter most exactly when data
is scarce, which is where a long-tail safety concept lives.)*

## 8. So what — the deployment recipe (1:00)
**SHOW:** the faces table (FPR@95%TPR by negative scheme). Title: *"You seal exactly the faces you train against."*
**SAY:** This pays off in something you can actually build. A flat fifteen-way probe has no "none of these" — show it
a passage about a dog and it'll still confidently call it manipulation. Add one cheap reject class and that problem's
gone. After that, picking negatives is just budget. Graph siblings seal the neighbour face for nothing. A cheap
"other" tier seals the far-away stuff completely. But the graph gives you almost no protection against the model's
real confusers — the ones it actually trips on — and only the expensive option, mining the model's own confusions,
seals those: a twenty-point cut in false positives. So: cheap everywhere, find the gap, spend the expensive money
only on the gap.

## 9. What this taught me + what's next (0:45)
**SHOW:** three lines (pre-register / verify your positives / bounded answer + mechanism beats unbounded yes); a
"Next" line.
**SAY:** What I'll take from this. Iteration one's false positive would have made a great talk — and it would have
been wrong. The discipline is the only reason I can stand here and trust these numbers: pre-register, try to kill
your own positives, bound what you claim. And the relationships? They turned out not to be a way of telling concepts
apart at all. They're a map — of where to look next. Thanks. Happy to take questions.

---

## Delivery notes
- Eight minutes is unforgiving. Slides 2, 5, 6 carry the talk; if you're running over, compress 3 and 7.
- The four beats to actually *land* (slow down, let them sit): "It was wrong." (4) · "The line didn't move." (6) ·
  "Structure is free. Data isn't." (2) · "a map — of where to look next." (9). Everything else, just talk.
- Numbers to say out loud, rounded: **six points** of F1 (local), **a third** lexical, **a hundredth of a point**
  held-out-edge, **p = 0.002** placebo, **twenty-point** FP cut. The rest stays on the slides.
- Slide 6, three numbers only: "+0.005, CI across zero, could've caught 0.014." Don't read the table aloud.
- Likely Q&A — "is six points even meaningful?" (it's about the whole remaining headroom at a 0.93 ceiling, and the
  false-positive analysis is where the deployment value really shows) · "does this generalise past SET G?" (one
  family — honest limit, in the paper's §10) · "why linear probes?" (deployment-realistic; MLP spot-checks matched) ·
  "who ran the second model?" (a peer replication by an experienced researcher who asked to be credited exactly that
  way — be straight about it, just don't volunteer the name).
