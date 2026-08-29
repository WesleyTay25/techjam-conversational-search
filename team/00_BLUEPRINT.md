# Blueprint — TechJam Conversational Search

Owner: team lead. Read this first, then your brief in `team/briefs/`.

## 1. The problem in one paragraph

The evaluator picks one product out of 50,000 and role-plays a customer who
wants it. We must place that exact `parent_asin` in a ten-item list, in as few
turns as possible, over at most ten turns. The starter agent throws the
customer's words into BM25 and guesses: Hit Rate 12.5%, MTTC 9.81 out of 11.
Everything below is about replacing guessing with *elimination*.

## 2. The two bets

**Bet 1 — the pool matters more than the ranking.** Every team will rank
"dress" well. The leverage is deciding *which* few hundred rows we rank at all.
Two signals do almost all of that work:

- The customer's opening line always contains the target's own coarse category
  path (`Women Dresses`, `Shoes Boots`, …). That is a hashable key, not a
  keyword — it collapses 50,000 rows to a few hundred before a single score is
  computed.
- Situational language ("for a wedding", "for the gym") selects a *different*
  pool, not a heavier weight on the same pool. Dress-for-a-wedding and
  dress-for-the-gym share the head noun and almost no products.

**Bet 2 — only ask questions that cut the pile.** A clarification is worth a
turn only if the answer partitions the surviving candidates. With 800 rows left,
"what colour?" splits into ~30 buckets and "what brand?" splits 780 of them into
one no-name bucket. So we score every askable attribute by expected information
gain *against the live pool* each turn, rather than following a fixed script.

And the standing rule that falls out of the API: **never ask empty-handed.** The
contract permits `message` + `ask_attribute` + ten recommendations in a single
response, and the session ends the instant the target appears. The starter's
9.81 MTTC is almost entirely turns spent asking without guessing.

## 3. Architecture

```
reset(session_id, profile)
        |
        v
+---------------------------- per-turn loop ----------------------------+
|                                                                       |
|  message                                                              |
|     |                                                                 |
|     v                                                                 |
|  [C] Intent Router ------> track: buying | browsing | override        |
|     |                              |                                  |
|     v                              v                                  |
|  [C] Slot Extractor        [C] State Machine                          |
|     |                       accumulate / decay / ERASE-on-override    |
|     +--------> DialogState <-------+                                  |
|                    |                                                  |
|     +--------------+---------------+---------------+                  |
|     v              v               v               v                  |
|  [A] Constraint [A] Structured  [A] Lexical     [B] Dense             |
|      exact-key      filters         BM25/FTS5       TF-IDF+SVD        |
|     (precision)   (price/colour)  (phrase)        (semantic)          |
|     +--------------+---------------+---------------+                  |
|                    v                                                  |
|            [B] Weighted RRF fusion + dynamic truncation               |
|                    v                                                  |
|            [D] Feature reranker  ->  [E] optional LLM rerank          |
|                    v                                                  |
|            top 10  +  [D] info-gain question policy                   |
|                    v                                                  |
|            {message, ask_attribute, recommendations, usage}           |
+-----------------------------------------------------------------------+
```

Letters are branch owners (§5).

## 4. Mapping to the four pillars

| Pillar | Where it lives | What the judge sees |
|---|---|---|
| I. Dual-track routing + hybrid pipeline | `src/nlu/intent_router.py`, `src/retrieval/*`, `src/retrieval/fusion.py` | Buying flips to a filter-first track with structured+constraint weight; Browsing flips to a diversity-first dense track. Same query, different pool. |
| II. Multi-turn state machine + proactive guidance | `src/dialog/state.py`, `src/dialog/question_policy.py` | Slot accumulation, time decay, and hard erasure on override; a pool-overload cutoff that forces a clarification turn. |
| III. Self-evolution / dynamic context programming | `src/dialog/state.py` (distillation), `src/rank/reranker.py` (profile prior), `src/dialog/question_policy.py` (cross-session probe bandit) | The probe ordering re-tunes itself at runtime from observed conversion turns; ranking re-weights from the anonymized profile. |
| IV. Evaluation matrix | `tools/ablate.py`, `tools/profile_run.py` | Per-scenario Hit@10 / MRR / MTTC, plus a leave-one-component-out ablation table and latency/token telemetry. |

## 5. Branches and ownership

One branch per person. **File ownership is disjoint, so merges do not conflict.**
Nobody edits a file they do not own; if you need a change in someone else's
file, open an issue on the shared board and use a stub locally.

| Branch | Owner | Files owned | Depends on |
|---|---|---|---|
| `main` | Lead | — (merge only) | — |
| `feat/contracts` | Lead (D0, merged first) | `src/contracts.py`, `src/catalog/loader.py`, all `__init__.py`, `tests/fixtures/`, `requirements.txt` | — |
| `feat/retrieval-lexical` | **A** | `src/catalog/constraint_index.py`, `src/retrieval/lexical.py`, `src/retrieval/structured.py`, `tests/unit/test_retrieval_lexical.py` | contracts |
| `feat/retrieval-dense` | **B** | `src/retrieval/dense.py`, `src/retrieval/fusion.py`, `tests/unit/test_retrieval_dense.py` | contracts |
| `feat/dialog-state` | **C** | `src/nlu/intent_router.py`, `src/nlu/slot_extractor.py`, `src/dialog/state.py`, `tests/unit/test_dialog.py` | contracts |
| `feat/policy-rerank` | **D** | `src/dialog/question_policy.py`, `src/rank/reranker.py`, `tests/unit/test_policy.py` | contracts |
| `feat/llm-evalops` | **E** | `src/rank/llm_reranker.py`, `tools/*`, `team/REPORT.md` | contracts |
| `feat/orchestrator` | Lead | `src/agent.py`, `starter/agent.py`, root `README.md` | A–E stubs |

**Collapsing for a smaller team.** 4 people: lead takes E. 3 people: lead takes
B and E; A takes structured+dense. Never merge C and D into one person — the
state machine and the question policy are the two places where a bug is
invisible until the score drops.

## 6. Milestones

| # | Gate | Exit criteria |
|---|---|---|
| **M0** | Kit ready | Catalog downloaded and SHA-256 verified; `python3 -m evaluator.local_evaluator` reproduces baseline 0.10671; `python3 -m unittest discover tests` green; `feat/contracts` merged. |
| **M1** | Skeleton runs end to end | Orchestrator wired to all five components; every branch's stub replaced by a working-but-naive implementation; **no crash on 200 sessions**; score ≥ baseline. |
| **M2** | Beats baseline 3x | TechnicalScore ≥ 0.35 on the public set. Category-tail pooling and constraint-exact matching live. Per-scenario table published. |
| **M3** | Feature complete | TechnicalScore ≥ 0.60. Override erasure, info-gain policy, reranker, dynamic truncation all merged. Ablation table shows each component earns its place. |
| **M4** | Ship | TechnicalScore ≥ 0.70 target. README, report, ablations, latency/token disclosure, demo video, one narrated multi-turn session. Full offline run passes with network disabled. |

Score arithmetic for calibration: `0.50·HR + 0.30·MRR + 0.20·Efficiency`. At
HR 0.85 / MRR 0.60 / MTTC 3.0 that is `0.425 + 0.180 + 0.160 = 0.765`.

## 7. Standing engineering rules

1. **Never import from `evaluator/`.** The submission must run under the
   organizer's private harness. Where we need simulator logic, it is *mirrored*
   in `src/contracts.py` with a docstring naming the source function.
2. **Never edit `evaluator/`, `data/`, or `docs/`.** Those are organizer
   artifacts; touching them invalidates the run.
3. **Offline is the default path.** Any LLM call is opt-in behind
   `TECHJAM_LLM=1`, hard-timeout bounded, and falls through to the deterministic
   reranker. The official score must be reproducible with the network down.
4. **No secrets in git.** Keys come from environment variables only.
5. **Catalog is read-only.** No mutations, no synthetic ASINs, no writes.
6. **Everything expensive happens in `__init__`.** `respond()` has a per-turn
   latency budget of 150 ms offline; the harness may enforce timeouts.
7. **Every branch ships tests that run on `tests/fixtures/mini_catalog.jsonl`**,
   so CI does not need the 50k catalog.
8. **Determinism.** Seed every RNG. Two runs of the evaluator must produce
   byte-identical `results.json`.

## 8. Risk register

| Risk | Impact | Mitigation | Owner |
|---|---|---|---|
| Private sessions ship real `intent_card`s, so constraint strings differ in distribution from the public set | High — exact-key matching degrades | Graded backoff: exact key -> token-set Jaccard -> dense cosine. Never let a route be the only path to a hit. | A + B |
| Organizer paraphrases customer messages | High — template routing breaks | Intent router is two-layer; the semantic fallback is tested by paraphrasing the public set ourselves (`tools/paraphrase_stress.py`). | C |
| Network disabled at final scoring | Total if we depend on an API | LLM layer opt-in only; offline run is the reported number. | E |
| Overfitting the 200 public sessions when tuning reranker weights | Medium — private score collapses | 5-fold CV, report mean±sd, cap the number of tuned weights at ~10. | D |
| Memory / latency blowup at 50k rows | Medium | Budget: < 1.5 GB resident, < 150 ms/turn. `tools/profile_run.py` gates it at every milestone. | E |
| Merge conflicts stalling the last day | Medium | Disjoint file ownership + frozen contracts + daily rebase onto `main`. | Lead |

## 9. Deliverables checklist (Devpost)

- [ ] Public GitHub repo, commented code, README with overview / setup /
      reproduction / limitations / contributions
- [ ] Written project description: tools, APIs, libraries, datasets
- [ ] Demo video on YouTube (public), linked from Devpost — narrated multi-turn
      session plus the ablation table
- [ ] Disclosure: model choice, token usage, estimated cost, latency, fallback
      behaviour
- [ ] `results.json` for the offline configuration, reproducible from a clean
      clone with one command
