# Project Description

## Overview

We built a conversational shopping agent for the TechJam Conversational
E-Commerce Search Challenge. Given a short customer message and an anonymized
preference profile, the agent has at most 10 turns to place the customer's
hidden target product inside a ranked top-10 list of catalog items, while
asking useful clarification questions along the way.

On the 200-session public set:

| Metric | Starter baseline | Ours |
|---|---|---|
| TechnicalScore | 0.10671 | 0.68569 |
| Hit Rate@10 | 0.125 | 0.780 |
| MRR | 0.068034 | 0.518633 |
| MTTC | 9.81 | 3.995 |

Both numbers come from the same unmodified evaluator. The run is deterministic:
two consecutive runs produce identical results.

## How our solution addresses the problem statement

The provided starter agent puts the customer's words into BM25 and ranks 50,000
products. We replaced that with elimination, based on two ideas.

**1. The candidate pool matters more than the ranking.**
The opening message always contains the target's coarse category path, such as
"Women Dresses", and any constraint the customer discloses is literal catalog
text. Both are hash lookups rather than searches, so we use them to cut 50,000
rows down to a few hundred before scoring anything. Across 755 turns the median
surviving pool is 5 products.

**2. Only ask questions that cut the pile.**
Every turn we score each askable attribute (color, material, size, brand,
budget, style, use case) by how much it is expected to split the current
candidate pool, and ask the best one. This replaces a fixed question script,
so we do not waste a turn asking about brand when almost every remaining item
has no brand.

We also never ask empty-handed. The API allows a question and ten
recommendations in the same response, and the session ends the moment the target
appears, so every reply carries both. Most of the starter's 9.81 MTTC is turns
spent asking without guessing.

Each turn runs through this pipeline:

1. An intent router classifies the turn as buying, browsing, intent override, or
   boundary.
2. A slot extractor and dialog state machine accumulate constraints, decay old
   ones, and erase them when the customer changes their mind mid-session.
3. Four retrieval routes run in parallel: exact-key constraint lookup,
   structured filters, lexical BM25 over SQLite FTS5, and a dense TF-IDF + SVD
   route.
4. Results are merged with weighted reciprocal rank fusion.
5. A feature reranker with MMR diversification produces the final top 10.

Every stage is individually guarded. If a retrieval route, the reranker, or the
question policy fails, it falls back to the previous stage, so a session never
crashes.

Results by scenario: browsing 0.850 hit rate, buying 0.800, boundary 0.800,
intent override 0.533. Intent override is our weakest track and the main area
left to improve.

## Development tools used

- VS Code as the editor, macOS terminal, Python virtual environments
- Git and GitHub, with one branch per component and a shared frozen contract
  file so the five workstreams could be developed in parallel
- Python `unittest` for the test suite (87 tests, runnable without the catalog)
- The organizer's local evaluator as the single source of truth for every
  reported number

No Colab or Jupyter notebooks were used. The whole system runs as a command
line pipeline so any reported metric can be regenerated with one command.

## APIs used

None on the scored path. The submitted configuration is fully offline, and the
reported token usage is 0 prompt tokens and 0 completion tokens.

There is an optional semantic reranking layer behind an environment flag
(`TECHJAM_LLM=1`) that targets the Anthropic Claude API. It is disabled by
default, is not used for any reported result, and no API keys are stored in the
repository.

## Libraries and frameworks used

- Python standard library, which does most of the work: `sqlite3` (FTS5 index
  for the BM25 route), `re`, `json`, `math`, `collections`, `dataclasses`
- NumPy for the fusion and reranking math
- scikit-learn for `TfidfVectorizer` and `TruncatedSVD` in the dense route

Everything is CPU only, in memory, and offline. No GPU, no PyTorch, and no
external vector database.

## Datasets and assets used

- The frozen competition catalog of 50,000 products from the
  `Clothing_Shoes_and_Jewelry` category, verified against the published
  SHA-256 checksum. We use the title, features, details, description,
  categories, store, price, and `parent_asin` fields. The catalog is read only.
- The 200 labeled public sessions provided by the organizer (80 buying,
  80 browsing, 30 intent override, 10 boundary), used for development and for
  every number reported here.
- `docs/baseline_results.json`, the organizer's starter score, used as our
  control.
- Both the catalog and the sessions come from Amazon Reviews 2023 by McAuley
  Lab, UCSD, using text and structured product metadata only. See
  `DATA_ATTRIBUTION.md`.

We did not add any external, scraped, or manually labeled data.
