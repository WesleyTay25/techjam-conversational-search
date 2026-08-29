# Brief A — Constraint index, lexical BM25, structured filters

**Branch:** `feat/retrieval-lexical`
**Pillar:** I (high-precision Buying track)
**Read first:** `team/00_BLUEPRINT.md`, `team/02_EVALUATOR_MECHANICS.md` §2–3, `src/contracts.py`

## Why this branch matters most

You own the pool. Every downstream component ranks whatever set you hand it, so
a good reranker over a bad pool scores nothing. Two of your indices carry the
project:

- **category tail** — present in every opening message, cuts 50,000 to a few
  hundred before any scoring happens;
- **constraint exact key** — the simulator speaks strings that live verbatim in
  the target's own `features`/`details`, so a disclosed constraint is a hash
  lookup that typically leaves tens of rows.

## Files you own

- `src/catalog/constraint_index.py`
- `src/retrieval/lexical.py`
- `src/retrieval/structured.py`
- `tests/unit/test_retrieval_lexical.py`

Do not edit anything else. `src/catalog/loader.py` and `src/contracts.py` are
already written and frozen — `Product.constraint_keys`, `Product.category_tail`,
`Product.material`, `Product.color` and `Product.price` are computed for you.

## Task 1 — `ConstraintIndex`

Build, once, at construction:

| Index | Key | Value |
|---|---|---|
| `_by_constraint` | `normalize_key(clean_constraint(s))` for every flattened feature and `"{k}: {v}"` detail | set of asins |
| `_by_tail` | `product.category_tail` (already computed) | set of asins |
| `_by_price` | price rounded to cents | set of asins |
| `_by_material`, `_by_color` | token | set of asins |
| `_by_token` | each token of each constraint key | set of asins — this is the **backoff** layer |

`by_constraint(s)` must be graded, not brittle:

1. exact `normalize_key` hit;
2. else token-set Jaccard ≥ 0.6 against constraint keys sharing a rare token
   (use `_by_token` to keep the comparison set small — do not scan 50k);
3. else empty set, and let the dense route handle it.

Step 2 is not optional. See `team/02_EVALUATOR_MECHANICS.md` §5: if the private
sessions carry real intent cards, exact matching degrades and the Jaccard layer
is what keeps us scoring.

`candidate_pool(state)` composes the signals:

- start from the category-tail set if `state.category_tail` is known, else all
  asins;
- intersect with each hard constraint's asin set, **in descending order of
  selectivity** (smallest set first);
- if an intersection would empty the pool, skip that constraint and record it in
  `Candidate.evidence` as unmatched rather than dropping to zero;
- if the pool exceeds `2000`, return it anyway and set `state.pool_size` — the
  question policy uses pool size to decide whether to force a clarification.

Backoff on the tail too: `Women Dresses` -> `Dresses` -> unrestricted. Category
paths in the catalog are messy; an exact-only tail lookup will silently return
nothing on some sessions.

## Task 2 — `LexicalRoute`

Keep SQLite FTS5 (stdlib, no deps, fast enough) but fix the starter's three
weaknesses:

1. **Phrases, not unigram soup.** Disclosed constraints are multi-word strings.
   Query them as FTS5 phrases (`"machine wash cold"`) with an OR of the
   individual terms as a lower-weighted fallback, not as a flat OR of tokens.
2. **Field weights.** The starter's `bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5,
   1.5, 1.0)` is a guess. Grid-search the seven weights on the 40-session dev
   slice and report the winner in your PR. Expect `features`/`details` to
   deserve more weight than the starter gives them, because that is where
   constraint text actually lives.
3. **Pool restriction.** When `state` carries a pool under ~5,000, restrict the
   query to it rather than scoring the whole catalog. Cheapest correct way:
   score globally with a generous `LIMIT` and post-filter, then fall back to a
   full scan only if the filtered result is short.

Return `Candidate` with `evidence={"bm25": raw_score}`.

## Task 3 — `StructuredRoute`

Hard filters, no scoring, for signals that are exact:

- **price** — a `budget around $X` disclosure gives an exact figure. Filter to
  `abs(price - X) <= max(0.01, 0.02 * X)`. This is the single most selective
  filter available; make sure the slot extractor's value reaches you as a float
  in `state.price_target`.
- **material**, **colour** — token match against `Product.material` /
  `Product.color`, plus a substring check on `text_blob` for values outside the
  mirrored regex vocabularies (e.g. "navy", "merino").
- **department / size** — from `Product.details`.

Filters must be *soft-failing*: if a filter empties the pool, drop it and mark
`evidence["filter_dropped_<name>"] = 1.0`. Never return an empty list when a
non-empty one exists — an empty turn is a wasted turn.

## Acceptance

`tests/unit/test_retrieval_lexical.py` on `tests/fixtures/mini_catalog.jsonl`:

- `by_constraint("100% Cotton")` returns exactly the three cotton fixtures.
- `by_constraint("100 % cotton  ")` (whitespace/case noise) returns the same set — proves normalization.
- `by_constraint("cotton 100%")` returns the same set via Jaccard backoff — proves graded matching.
- `by_tail("Women Dresses")` returns the five dress fixtures, and `by_tail("Dresses")` backs off to the same.
- `by_price(49.99)` returns the two 49.99 fixtures.
- `candidate_pool` on a state with tail `Women Dresses` + constraint `100% Cotton` returns exactly `{B0MINI0001}`.
- `candidate_pool` with a contradictory constraint (`Genuine Leather` + tail `Women Dresses`) returns the five dresses, not the empty set.
- `LexicalRoute.search` never returns duplicates and never returns an asin outside the catalog.
- `StructuredRoute` with an impossible price returns the unfiltered pool with `filter_dropped_price` set.

Plus: the full evaluator runs 200 sessions with no exception.

## Definition of done

See `team/01_WORKFLOW.md`. Additionally report in your PR: index build time,
resident memory after construction, and mean pool size after turn 1 across the
200 public sessions. Those three numbers go straight into the feasibility
section of the report.
