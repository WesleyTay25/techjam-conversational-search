# TechJam Conversational E-Commerce Search Challenge

Build an AI shopping agent that asks useful follow-up questions and recommends the customer's hidden target product within at most 10 turns.

---

# Our Submission

> Team documentation. Everything below `## What You Receive` is the organizer's
> original participant kit and is unmodified.

## Overview

The starter agent throws the customer's words at BM25 and guesses: Hit Rate
12.5%, MTTC 9.81. We replace guessing with **elimination**, on two bets:

1. **The pool matters more than the ranking.** Every opening message contains
   the target's own coarse category path (`Women Dresses`), and disclosed
   constraints are literal catalog text. Both are hash lookups, not searches, so
   they collapse 50,000 rows to a few hundred before a single score is computed.
2. **Only ask questions that cut the pile.** Every askable attribute is scored
   by expected information gain against the *live* pool each turn, rather than
   following a fixed script. And we never ask empty-handed: a question and ten
   guesses cost the same single turn, so every response carries both.

## Architecture

```
reset(session_id, profile)
        |
        v
+--------------------------- per-turn loop ----------------------------+
|  message                                                             |
|     v                                                                |
|  [C] Intent router --> track: buying | browsing | override | boundary|
|     v                                                                |
|  [C] Slot extractor --> [C] State machine                            |
|            accumulate / decay / ERASE-on-override --> DialogState    |
|     +--------------+---------------+---------------+                 |
|     v              v               v               v                 |
|  [A] Constraint [A] Structured  [A] Lexical     [B] Dense            |
|      exact-key      filters         BM25/FTS5       TF-IDF+SVD       |
|     +--------------+---------------+---------------+                 |
|            [B] Weighted RRF fusion + dynamic truncation              |
|            [D] Feature reranker --> [E] optional LLM rerank          |
|            top 10  +  [D] info-gain question policy                  |
|     v                                                                |
|  {message, ask_attribute, recommendations, usage}                    |
+----------------------------------------------------------------------+
```

| Module | Role |
|---|---|
| `src/agent.py` | Orchestrator — the `reset`/`respond` entry point |
| `src/nlu/`, `src/dialog/state.py` | Track routing, verbatim slot extraction, state machine |
| `src/catalog/constraint_index.py` | Exact-key constraint and category-tail indexes |
| `src/retrieval/` | Lexical (FTS5/BM25), structured filters, dense TF-IDF+SVD, RRF fusion |
| `src/dialog/question_policy.py` | Information-gain probe selection + cross-session bandit |
| `src/rank/reranker.py` | Feature reranker with MMR diversification |
| `src/rank/llm_reranker.py` | Opt-in LLM layer (`TECHJAM_LLM=1`), falls through when off |

`starter/agent.py` is the harness entry point and simply delegates to
`src/agent.py`.

## Setup and Reproduction

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# The catalog is not in the repo; fetch it from the participant-kit release.
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
shasum -a 256 -c SHA256SUMS

python3 -m unittest discover tests      # 87 tests, no catalog needed
python3 -m evaluator.local_evaluator --output results.json
```

Two escape hatches, both off by default:

| Variable | Effect |
|---|---|
| `TECHJAM_BASELINE=1` | Routes the harness back to the original BM25 starter, so the M0 control score stays reproducible from the same command. |
| `TECHJAM_LLM=1` | Enables the optional LLM rerank layer. Unset, nothing in the pipeline touches the network. |

The reported configuration is fully offline and deterministic: no RNG on the
response path, and two runs produce byte-identical `results.json`.

## Disclosure

- **Model choice:** none on the scored path. The offline pipeline is pure
  Python + numpy/scikit-learn, so reported token usage is genuinely `0`.
- **Optional layer:** `TECHJAM_LLM=1` targets `claude-haiku-4-5-20251001`,
  bounded by a hard timeout and capped to a 20-candidate shortlist. Its model
  call is not yet implemented (see Limitations), so it currently costs nothing.
- **Latency:** the orchestrator budgets 150 ms per turn and logs a warning when
  a turn exceeds it. All index construction happens once in `__init__`.
- **Fallback behaviour:** every stage is individually guarded. A dead retrieval
  route, a broken reranker, a failing question policy, or a missing numpy wheel
  each degrade to the stage before them; `respond` never raises, because the
  harness scores a thrown exception as a miss with no traceback.

## Limitations and Known Gaps

- **No public-set score is reported yet.** `data/catalog.jsonl` is distributed
  via the participant-kit release and is not in this repo, so the integrated
  pipeline has been verified end to end against `tests/fixtures/mini_catalog.jsonl`
  only. The 12-product fixture makes a top-10 nearly free, so its metrics prove
  *integration*, not performance. The full 200-session run is the next gate.
- **The LLM rerank layer is a pass-through.** It is wired, contract-conformant,
  opt-in, and safe, but `_call_model` still returns `[]`. Owned by Branch E.
- **`tools/` is a skeleton.** The ablation, paraphrase-stress, slice, and
  profiling harnesses the blueprint calls for are not built yet, so there is no
  ablation table or latency profile.

## Contributions

| Branch | Scope |
|---|---|
| `feat/retrieval-lexical` | Constraint index with graded backoff, FTS5/BM25 route, structured filters |
| `feat/retrieval-dense` | TF-IDF+SVD dense route, weighted RRF fusion, dynamic truncation |
| `feat/dialog-state` | Two-layer intent router, verbatim slot extraction, state machine, override erasure, context distillation |
| `policy-rerank` | Information-gain question policy, probe bandit, feature reranker with MMR |
| `feat/llm-evalops` | LLM reranker scaffold, report and tooling skeleton |
| Lead | Frozen contracts, catalog loader, orchestrator, harness wiring, integration tests |


## What You Receive

- A frozen catalog of 50,000 products from the `Clothing_Shoes_and_Jewelry` category of Amazon Reviews 2023.
- 200 labeled public sessions for local development.
- A weak BM25 starter agent and deterministic local evaluator.
- The Agent API contract and scoring rules.

The organizer keeps 800 additional sessions private for final evaluation.

## Task

For each session, your agent receives an anonymized preference profile and a short customer message. Raw user IDs, review text, timestamps, and purchase history are never disclosed. On every turn the agent may:

- ask a natural clarification question in `message` and identify one requested field in `ask_attribute`;
- return a ranked list of up to 10 catalog `parent_asin` values;
- do both in the same response.

The session ends when the target product appears in the scored Top 10 or after turn 10. Sessions cover Buying, Browsing, Intent Override, and Boundary behavior.

## Download the Catalog

Download `catalog.jsonl.gz` from the GitHub Release attached to this repository, then run:

```bash
gzip -dk catalog.jsonl.gz
mv catalog.jsonl data/catalog.jsonl
```

Verify the downloaded file using the published `SHA256SUMS` file.

## Run the Starter

Python 3.10 or later is recommended. The starter uses only the Python standard library.

```bash
python3 -m evaluator.local_evaluator
```

Edit `starter/agent.py` to implement your system. Do not edit the evaluator or public labels when reporting your local score.
The command writes per-session results and aggregate metrics to `results.json`.

The included weak BM25 starter scores Hit Rate@10 `0.125`, MRR `0.068034`, and
MTTC `9.81` on the released public set. See `docs/baseline_results.json`.

## Agent Interface

```python
class Agent:
    def reset(self, session_id: str, user_profile: dict) -> None:
        ...

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int) -> dict:
        return {
            "message": "Do you have a material preference?",
            "ask_attribute": "material",
            "recommendations": [
                {"parent_asin": "B000..."},
                {"parent_asin": "B001..."}
            ],
            "usage": {"prompt_tokens": 120, "completion_tokens": 30}
        }
```

`ask_attribute` is one of `category`, `material`, `color`, `size`, `style`, `brand`, `budget`, `feature`, `use_case`, `other`, or `null`. See `docs/agent_api_contract.json`.

## Technical Metrics

- **Hit Rate@10:** fraction of sessions that find the target within 10 turns.
- **MRR:** mean reciprocal rank of the target; a miss contributes zero.
- **MTTC:** mean first-hit turn; a miss is assigned turn 11.
- **Reported token usage:** prompt and completion tokens returned by the team's model client.

```text
TechnicalScore = 0.50 × HitRate@10 + 0.30 × MRR + 0.20 × Efficiency
Efficiency = clip((11 - MTTC) / 10, 0, 1)
```

`TechnicalScore` is an objective input to the `Technical Execution` assessment. It is not a separate judging criterion and does not represent the entire `Technical Execution` score.

Only exact `parent_asin` equality produces a hit. Core metrics are also reported by scenario.

## Model Choice and Cost

Teams may use any legally accessible LLM API or local model. Teams manage their own credentials and must never commit API keys. Model choice, estimated cost, token usage, and latency must be disclosed. Token usage is a feasibility metric, not part of the core technical score. The organizer does not provide or reimburse model API credits; teams are responsible for any costs incurred through optional external services.

## Files

```text
data/public_set.jsonl             200 labeled development sessions
docs/competition_specification.md participant rules and evaluation protocol
docs/agent_api_contract.json      machine-readable Agent contract
docs/evaluation_config.json       scoring configuration
docs/baseline_results.json        reproducible weak-starter reference score
starter/agent.py                  editable weak starter
evaluator/local_evaluator.py      public-set simulator and scorer
```

## Judging and Submission Policy

- Participant submission requirements: `docs/submission_rules.md`
- Organizer-only final judging controls: `organizer/JUDGING_RUNBOOK.md`
- Organizer private release checklist: `organizer/private_release_checklist.md`
- Judging day operations SOP: `organizer/JUDGING_DAY_SOP.md`

## Data Source

The catalog and sessions are derived from Amazon Reviews 2023 by McAuley Lab, UCSD. See `DATA_ATTRIBUTION.md` before using or redistributing the data.
Sessions are sampled deterministically from the official Clothing 5-core leave-last-out split and joined to the frozen catalog.
