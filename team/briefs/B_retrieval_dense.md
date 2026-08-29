# Brief B — Dense semantic route and multi-route fusion

**Branch:** `feat/retrieval-dense`
**Pillar:** I (Browsing track, hybrid pipeline)
**Read first:** `team/00_BLUEPRINT.md`, `team/02_EVALUATOR_MECHANICS.md`, `src/contracts.py`

## Why this branch matters

40% of sessions are Browsing: the customer opens with a category and "I'm still
exploring", and no keyword overlap exists between what they say and what they
want. You own the arm that maps *situation* to *product* — "for a wedding" onto
formal, midi, silk, higher price band — and the fusion layer that decides how
much each route is trusted per track.

You are also the insurance policy. Branch A's exact matching is precise but
brittle; if the private sessions paraphrase, your cosine route is what still
finds the target. Design for that, not for the public set.

## Files you own

- `src/retrieval/dense.py`
- `src/retrieval/fusion.py`
- `tests/unit/test_retrieval_dense.py`

## Task 1 — `DenseRoute`

**Constraint from the rules:** in-memory only, no external vector database, no
network at scoring time, no model fine-tuning. So:

**Default (must work, ships in the offline run):** TF-IDF over
`Product.text_blob` (word 1–2 grams, `min_df=2`, sublinear tf) ->
`TruncatedSVD(n_components=256)` -> L2-normalise -> keep a single
`float32 [50000, 256]` matrix. Query = same transform, retrieval = one
`numpy` matmul plus `argpartition`. That is ~50 MB resident and single-digit
milliseconds per query. Fit at construction, seeded, deterministic.

**Optional upgrade (behind `TECHJAM_DENSE=st`):** a local sentence-transformers
MiniLM encoding, embedded once at build time and cached to
`data/embeddings.npy` (gitignored). Only pursue this after the default path is
merged and only if M2 is already met — it adds a heavy dependency for a gain we
have not yet measured. Benchmark both on the same dev slice and let the number
decide.

Query construction matters as much as the model. Build the query text from
state, not from the raw last message: `category_tail` + all active slot values
weighted by `Slot.weight` + a small contribution from `user_profile.summary`.
Erased slots (weight 0) must not appear.

Add **scenario expansion** for browsing: a small hand-written lexicon mapping
situational phrases to product vocabulary (`wedding -> formal, gown, midi,
maxi, chiffon, satin`; `gym -> athletic, moisture wicking, performance,
spandex`). Keep it in a module-level dict of at most ~30 entries, keep it
readable, and ablate it — if it does not move the browsing hit rate, delete it
rather than defend it.

## Task 2 — `reciprocal_rank_fusion`

Weighted RRF: `score(d) = Σ_r w_r / (k + rank_r(d))`, default `k=60`. RRF over
raw score blending because our routes produce incomparable scales (BM25 is
unbounded, cosine is [-1,1], the constraint route is a set membership).

Preserve provenance: the fused `Candidate.evidence` must carry each route's
rank and contribution. Branch D's reranker consumes it and the customer-facing
explanation string is generated from it.

## Task 3 — `track_weights` (this is the dual-track router's teeth)

Weights are a function of `state.track`, `state.turn`, and how many hard slots
are filled. Starting point — tune these, do not treat them as given:

| Track | constraint | structured | lexical | dense |
|---|---|---|---|---|
| buying | 0.45 | 0.25 | 0.20 | 0.10 |
| browsing | 0.10 | 0.10 | 0.25 | 0.55 |
| override, pre-override turns | 0.20 | 0.15 | 0.25 | 0.40 |
| override, post-override | 0.50 | 0.25 | 0.15 | 0.10 |
| boundary | 0.15 | 0.10 | 0.25 | 0.50 |

Plus a continuous shift: as the number of active hard slots rises, migrate
weight from `dense` toward `constraint`. A session that starts as browsing and
accumulates three hard constraints should end up scored like a buying session.
Make that a function, not a table lookup — it is the "runtime workflow
re-orchestration" the problem statement asks for, and it should be visible in
the code as one.

## Task 4 — `dynamic_truncation`

Return retrieval depth per route for this turn:

- pool > 5,000 -> shallow (50/route); the pool is too broad for ranking to
  matter, so the turn's value is in the clarification, and Branch D will force
  a question. Signal this by setting `state.pool_size`.
- 200 < pool <= 5,000 -> 200/route.
- pool <= 200 -> score the whole pool; skip the dense route entirely (it costs
  latency and adds nothing once the pool is that small).

## Acceptance

`tests/unit/test_retrieval_dense.py` on the mini catalog:

- The SVD matrix is `float32`, L2-normalised (row norms within 1e-5 of 1.0), and shaped `[n_products, dim]` with `dim = min(256, n_features)`.
- Two constructions from the same fixture produce identical vectors — determinism.
- A query of `"dress for a wedding"` ranks `B0MINI0001`/`B0MINI0003` (formal dresses) above `B0MINI0002` (athletic dress). This is the whole thesis of the branch; if it fails, the scenario lexicon needs work.
- A query of `"dress for the gym"` inverts that ordering.
- `reciprocal_rank_fusion` with a single route reproduces that route's ordering exactly.
- A document ranked first by two routes outranks one ranked first by a single higher-weighted route (proves it is rank fusion, not score blending).
- `track_weights` sums to 1.0 for every track, and constraint weight is monotonically non-decreasing in the number of active hard slots.
- `dynamic_truncation` returns 0 for the dense route when `pool_size <= 200`.

## Definition of done

See `team/01_WORKFLOW.md`. Report in your PR: build time, matrix memory, mean
query latency, and browsing-scenario hit rate before/after the scenario lexicon.
