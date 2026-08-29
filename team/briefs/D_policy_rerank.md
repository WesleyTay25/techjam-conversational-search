# Brief D — Information-gain question policy and offline reranker

**Branch:** `feat/policy-rerank`
**Pillar:** II (proactive guidance), III (self-evolution), IV (precision)
**Read first:** `team/00_BLUEPRINT.md`, `team/02_EVALUATOR_MECHANICS.md` §3, `src/contracts.py`

## Why this branch matters

You own both ends of the efficiency metric. The reranker decides MRR — whether
the target lands at rank 1 or rank 9 once it is in the pool. The question policy
decides MTTC — whether we converge in three turns or nine. Efficiency is 20% of
the technical score and MRR another 30%, so half the score is yours.

## Files you own

- `src/dialog/question_policy.py`
- `src/rank/reranker.py`
- `tests/unit/test_policy.py`

## Task 1 — `InfoGainQuestionPolicy`

Score every askable attribute by how much its answer would shrink the live
candidate pool, then pick the best. For attribute `a` over pool `P`:

```
values(a, P)  = the observed value of a for each product in P (None if absent)
H(a)          = Shannon entropy of that value histogram, excluding None
coverage(a)   = fraction of P with a non-None value for a
p_answer(a)   = prior that the customer holds an undisclosed constraint of class a
gain(a)       = H(a) * coverage(a) * p_answer(a) * (1 - already_asked_penalty)
```

The brand case from the pitch falls straight out of this: brand has high raw
entropy but low `coverage` in a catalog full of no-name stores, so it loses to
colour. That is the arithmetic doing the work — do not special-case it.

`p_answer(a)` comes from inverting the simulator's disclosure router
(`contracts.classify_constraint`, documented in
`team/02_EVALUATOR_MECHANICS.md` §3): given what has already been disclosed,
which classes plausibly still hold an undisclosed constraint. Attributes in
`state.dead_attributes` get `p_answer = 0`.

**The open probe.** `ask_attribute="other"` bypasses the class filter and
returns the next two undisclosed constraints of any class. Model it honestly in
the same framework — its expected yield is the sum over classes weighted by
`p_answer`, so it wins naturally while the constraint set is unsaturated and
loses once only one specific slot is missing. **Model it; do not hard-code it.**
A policy that returns `"other"` unconditionally would score well on the public
set, but it is unexplainable in the pitch, contributes nothing to Innovation,
and is exactly the kind of thing that collapses when the private simulator
differs. The information-gain framing gets the same behaviour *and* a defensible
story.

**Over-generality cutoff (the problem statement asks for this by name).** When
`state.pool_size` exceeds ~5,000, ranking is noise. Force a clarification turn:
still ship ten recommendations, but pick the attribute with maximum raw entropy
and phrase the question as a structured choice ("Is this more for a formal
occasion, work, or everyday?") rather than an open one. Structured options
converge faster than open questions when the pool is broad.

**Question text.** Natural, one sentence, referencing what we already know so
it does not read like a form ("Got it — cotton, midi length. Is there a colour
you're set on?"). `message` is customer-facing and is 20% Impact plus 10%
Presentation; a robotic question costs real marks even when the retrieval is
right.

**Stop asking when it is over.** If the pool is under ~15 or every attribute's
gain is below threshold, return `(None, <explanation of the top picks>)`. A
pointless question is cognitive load, which the problem statement explicitly
penalises.

## Task 2 — `FeatureReranker`

Deterministic, offline, no network. This is the layer that carries the official
run if the organizer disables network access.

Features per candidate (extend as you find signal):

| Feature | Why |
|---|---|
| `constraint_exact_hits` | count of disclosed constraints matched exactly — expect this to dominate |
| `constraint_coverage` | fraction of active hard slots satisfied |
| `constraint_jaccard_mean` | soft version, carries the paraphrase case |
| `tail_exact` | category tail matches |
| `price_delta` | `\|price - target\| / target`, 0 when no target |
| `color_match`, `material_match` | structured agreement |
| `bm25_z`, `dense_cos` | route scores, z-normalised within the candidate set |
| `profile_tag_overlap` | `user_profile.preference_tags` against `text_blob` |
| `rating_prior` | `log1p(rating_number) * average_rating`, small weight |
| `erased_slot_match` | matches a slot the customer *overrode* — **negative weight** |
| `already_shown` | shown in a previous turn without converting — small negative |

Two ranking rules on top of the linear score:

- **Novelty across turns.** A product shown three turns ago and not converted is
  weak evidence against itself. Small penalty only — the customer never
  explicitly rejects anything, so do not over-read it.
- **Diversity for browsing.** On the browsing track, apply light MMR over the
  top 10 so we cover several sub-categories instead of ten near-duplicates. Ten
  colourways of one product is one guess, not ten. On the buying track, skip
  MMR — precision beats coverage there.

**Weight fitting.** Start hand-tuned so the system works end to end. Then fit
with logistic regression or coordinate ascent on the 200 public sessions, but:
cap the fitted weights at ~10, use 5-fold CV, and report mean ± sd in your PR.
200 sessions is a small sample and the private set is 4x larger and differently
generated — a weight set that only wins on the full-fit number is overfitting.
If CV variance is high, ship the hand-tuned weights and say so.

## Task 3 — Probe bandit (pillar III, do after M3)

The agent instance persists across all sessions in a run, so the policy can
learn *from the run itself*: track, per category tail, which first probe led to
the shortest conversion, and let a simple ε-greedy or Thompson bandit bias
`p_answer`. Reward is the observed conversion turn — the harness stops calling
`respond` after a hit, so the previous session's length is observable at the
next `reset`.

Two rules: keep it to *policy* learning (which probe to ask), never target
memorization — public and private sessions use different products, so
memorization cannot transfer and would just be overfitting. And keep it seeded
and deterministic so `results.json` is reproducible. Document the mechanism in
the report; this is the most defensible "self-evolution" claim we have.

## Acceptance

`tests/unit/test_policy.py` on the mini catalog:

- On a pool where every product has a distinct colour and one shared brand, `choose` returns `color`, not `brand`.
- An attribute in `dead_attributes` is never returned and scores `gain == 0`.
- With zero disclosed constraints, `other` scores highest; with three of four disclosed and only a colour slot missing, `color` beats `other`. This must come out of `expected_gain`, not an `if`.
- `pool_size = 20000` forces a non-None question with structured options in the text.
- `pool_size = 8` returns `ask_attribute is None`.
- Reranker: a candidate matching two exact constraints outranks one matching one, holding other features equal.
- Reranker: a candidate matching an *erased* slot ranks below an otherwise identical candidate that does not.
- Reranker output is a permutation of its input — no drops, no duplicates.
- Two runs on identical input produce identical ordering, including under the bandit.

## Definition of done

See `team/01_WORKFLOW.md`. Report in your PR: MTTC before/after the policy, MRR
before/after the reranker, the 5-fold CV spread on fitted weights, and the mean
number of questions asked per session by scenario.
