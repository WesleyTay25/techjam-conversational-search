# Brief E — Optional LLM rerank, evaluation tooling, report

**Branch:** `feat/llm-evalops`
**Pillar:** I (LLM semantic ranking), IV (evaluation matrix)
**Read first:** `team/00_BLUEPRINT.md`, `team/02_EVALUATOR_MECHANICS.md` §5, `src/contracts.py`

## Why this branch matters

Two jobs. First, the LLM layer that lets us claim a genuine "multi-route
retrieval -> LLM semantic ranking" pipeline without betting the score on a
network we may not have. Second — and this is the one that decides three of the
five judging criteria — the evidence. Ablations, per-scenario tables, latency
and cost disclosure, paraphrase robustness. A system that scores 0.7 with no
evidence of *why* loses to one that scores 0.65 and can prove every component
earns its place.

## Files you own

- `src/rank/llm_reranker.py`
- `tools/` (everything)
- `team/REPORT.md`

## Task 1 — `LLMReranker`

Opt-in, never load-bearing.

- Enabled only when `TECHJAM_LLM=1` **and** `ANTHROPIC_API_KEY` is set. Default
  off. The reported official number is the offline run.
- Wraps Branch D's `FeatureReranker` as `fallback`. Any exception, timeout, rate
  limit, malformed JSON, or missing key falls through silently to the fallback.
  Test that path explicitly with a stubbed client that always raises — it is the
  path that will actually execute if anything goes wrong on judging day.
- Only reranks when it can change the outcome: skip when the pool is under ~15
  (already converged) or when the top candidate's margin is large. Wasting
  tokens on a decided ranking is exactly the cost inefficiency the problem
  statement penalises.
- Sends the top ~30 fused candidates as compact records (asin, title, 2 features,
  price) plus Branch C's distilled state summary. Asks for a JSON array of asins,
  best first. Parse defensively; drop any asin not in the input set.
- Hard timeout ~2s per call, single retry, then fall back.
- Model: `claude-haiku-4-5-20251001` for cost, `claude-sonnet-5` for the quality
  ceiling. Benchmark both on the dev slice and report cost per session.
- Track real token counts and expose them via `usage`. The harness sums them
  into `reported_token_usage`; the number must be honest, not estimated.

**Never commit a key.** `os.environ` only, `.env` is already gitignored.

## Task 2 — `tools/`

| Tool | What it does |
|---|---|
| `slice.py` | Deterministic N-session subset preserving the 40/40/15/5 scenario mix. The team's fast dev loop — everyone uses it, so land it first. |
| `ablate.py` | Leave-one-component-out runs via env flags (`no-dense`, `no-constraint-index`, `no-policy`, `no-rerank`, `no-override-erasure`, `no-profile`). Emits a markdown table of TechnicalScore delta per component. This table is the single most persuasive artifact in the report. |
| `profile_run.py` | Per-turn latency percentiles, peak RSS, index build time, tokens per session. Gates the M-milestone budgets: < 1.5 GB resident, < 150 ms/turn offline. |
| `paraphrase_stress.py` | Rewrites the simulated customer's messages (synonym swap, clause reorder, filler insertion) and re-scores. **This is the private-set proxy.** Any component losing > 15% under paraphrase gets a fallback before it gets a tuning pass. |
| `session_trace.py` | Pretty-prints one session turn by turn — track, slots, pool size, chosen probe, top 3 with evidence. The demo video is a screen recording of this. |

`ablate.py` and `paraphrase_stress.py` matter more than the LLM layer. If you
run out of time, cut the LLM layer, not these.

## Task 3 — `team/REPORT.md`

Draft continuously, not at the end. Sections:

1. Problem framing — needle in a 50k haystack, why keyword search fails, why
   pool selection beats ranking
2. Architecture — the diagram from `00_BLUEPRINT.md` §3, mapped to the four pillars
3. Results — headline table, per-scenario table, baseline comparison
4. Ablations — the leave-one-out table
5. Robustness — paraphrase stress results, offline vs LLM configurations
6. Cost and feasibility — tokens, latency, memory, estimated cost per 1,000 sessions
7. Limitations and next steps — honest ones
8. Team contributions

## Reporting rules (these are non-negotiable)

- Every number cites its configuration and dataset. `TechnicalScore 0.73
  (offline, no LLM, full 200 public sessions)` — never a bare number.
- Tune on the dev slice, report on the full 200.
- Never report a number from a run where the evaluator or labels were modified.
- Report the offline configuration as the headline. The LLM configuration is a
  secondary line. If the organizer disables the network, the offline number is
  the only one that survives, and it must be the one we pitched.
- If a component does not help, say so in the report and delete it from the
  code. A crisp negative result reads as engineering judgment; a dead component
  reads as noise.

## Acceptance

- `slice.py` produces a scenario-balanced subset, identical across two runs.
- `ablate.py --dry-run` lists every flag and works end to end on a 20-session slice.
- `LLMReranker` with a stubbed always-raising client returns exactly the fallback ordering and reports zero tokens.
- `LLMReranker` with a stubbed client returning malformed JSON returns the fallback ordering.
- `LLMReranker` returns a permutation of its input — no invented or dropped asins, ever.
- With `TECHJAM_LLM` unset, no module under `src/` opens a socket. Verify by monkeypatching `socket.socket` to raise during a full 200-session run.
- `profile_run.py` reports p50/p95/p99 turn latency and peak RSS.

## Definition of done

See `team/01_WORKFLOW.md`. Your PR must include the current ablation table and
the offline-vs-LLM comparison, even if partial.
