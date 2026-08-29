# Brief LEAD — Orchestrator, integration, submission

**Branches:** `feat/contracts` (D0, merged first), `feat/orchestrator` (from M1)
**Files owned:** `src/contracts.py`, `src/catalog/loader.py`, all `__init__.py`,
`tests/fixtures/`, `requirements.txt`, `src/agent.py`, `starter/agent.py`, root `README.md`

## D0 — already done

`src/contracts.py`, `src/catalog/loader.py`, the package skeleton, the stub
modules, `tests/fixtures/mini_catalog.jsonl` and `requirements.txt` are written
and importable. Merge `feat/contracts` before anyone branches.

## `src/agent.py` — the orchestrator

```python
class ShoppingAgent:
    def __init__(self, catalog_path="data/catalog.jsonl"):
        products = load_catalog(catalog_path)          # once, ~5s
        self.index      = ConstraintIndex(products)     # A
        self.routes     = [ConstraintRoute, StructuredRoute, LexicalRoute, DenseRoute]
        self.machine    = StateMachine()                # C
        self.policy     = InfoGainQuestionPolicy()      # D
        self.reranker   = FeatureReranker(products)     # D
        if os.environ.get("TECHJAM_LLM") == "1":
            self.reranker = LLMReranker(fallback=self.reranker)   # E
        self.states = {}

    def reset(self, session_id, user_profile):
        self.states[session_id] = self.machine.start(session_id, user_profile)

    def respond(self, session_id, user_message, turn, top_k):
        state = self.machine.ingest(self.states[...], user_message, turn)
        pool  = self.index.candidate_pool(state)
        depth = dynamic_truncation(state, len(pool))
        fused = reciprocal_rank_fusion({r.name: r.search(state, depth) for r in routes},
                                       track_weights(state))
        ranked = self.reranker.rerank(state, fused)
        attribute, message = self.policy.choose(state, pool)
        return TurnResult(message, attribute, ranked[:top_k], *usage).to_payload()
```

Three hard requirements on top of the wiring:

- **`respond` must never raise.** The harness swallows exceptions and scores an
  empty response, so a bug that throws on 3% of sessions costs 3% of hit rate
  invisibly. Wrap the body; on failure log to stderr and return the previous
  turn's ranking (or a category-tail fallback list) rather than nothing.
- **Never return an empty recommendation list.** If every filter empties, back
  off to category tail, then to global popularity by `rating_number`. Ten
  mediocre guesses beat zero.
- **Session state keyed by `session_id`**, created in `reset`. Nothing about one
  session may leak into another's ranking — the probe bandit is the sole
  documented exception, and it learns policy only, never targets.

`starter/agent.py` becomes a thin adapter: `from src.agent import ShoppingAgent
as Agent`, keeping the `Agent(catalog_path)` positional signature the evaluator
calls. Keep the old BM25 implementation reachable as `LegacyAgent` for ablation.

## Integration cadence

- Merge in the order in `team/01_WORKFLOW.md`; run the full evaluator after each
  merge; post the score. Regressions get reverted, not debugged on `main`.
- Daily: confirm every branch has rebased. A branch stale by 24 h is the main
  risk to the final day.
- At M1, wire the naive version of every component end to end even if each is
  weak. Integration bugs found on the last day are what sink hackathon teams.

## Submission checklist

- [ ] `git clone` -> `pip install -r requirements.txt` -> download catalog ->
      one command reproduces `results.json` exactly
- [ ] Full 200-session offline run with the network disabled, passing
- [ ] README: overview, setup, reproduction, limitations, contributions
- [ ] `team/REPORT.md` finished, ablation table included
- [ ] Devpost: description, tools, APIs, libraries, datasets
- [ ] YouTube demo (public), linked from Devpost: `tools/session_trace.py` walkthrough
      of one buying and one override session, then the ablation table
- [ ] Disclosure: model, tokens, cost, latency, fallback behaviour
- [ ] No keys, no `data/catalog.jsonl`, no modified evaluator in the diff
