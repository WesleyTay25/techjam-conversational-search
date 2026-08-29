# Kickoff — first 90 minutes

## Everyone, in parallel

1. Run the D0 setup in `team/01_WORKFLOW.md`. **Do not proceed until
   `python3 -m evaluator.local_evaluator` prints `technical_score: 0.10671`.**
   That number is the control for every experiment we run.
2. Read `team/00_BLUEPRINT.md` and `team/02_EVALUATOR_MECHANICS.md` end to end.
   Twenty minutes, and it is the difference between building the right thing and
   building BM25 with extra steps.
3. Read `src/contracts.py`. It is frozen. If you think it is wrong, say so in
   the next hour — after that a change costs everyone a rebase.
4. Read your brief in `team/briefs/`.
5. Cut your branch and hand your agent the opening prompt from
   `team/01_WORKFLOW.md`.

## Assignment

| Person | Brief | Branch |
|---|---|---|
| Lead | `LEAD_orchestrator.md` | `feat/contracts`, then `feat/orchestrator` |
| A | `A_retrieval_lexical.md` | `feat/retrieval-lexical` |
| B | `B_retrieval_dense.md` | `feat/retrieval-dense` |
| C | `C_dialog_state.md` | `feat/dialog-state` |
| D | `D_policy_rerank.md` | `feat/policy-rerank` |
| E | `E_llm_evalops.md` | `feat/llm-evalops` |

Fill in real names before you start. If the team is smaller, apply the
collapsing rules in `team/00_BLUEPRINT.md` §5.

## Order of attack, if you only get one thing done

The score is roughly ordered by these, so build them in this order:

1. **Category-tail pooling** (A) — every session leaks the target's own category
   path in its opening line. This alone should roughly double hit rate.
2. **Constraint exact-key index** (A) — turns disclosed text into a hash lookup.
   Biggest single precision jump.
3. **Ship 10 recommendations on every turn** (Lead) — nearly free, and it is most
   of the gap between MTTC 9.81 and MTTC ~4.
4. **Override erasure** (C) — recovers 15% of sessions that otherwise score zero.
5. **Info-gain probes** (D) — converts turns into pool reduction instead of noise.
6. Dense route, reranker weights, LLM layer (B, D, E) — real gains, but they
   multiply the above rather than substituting for them.

Items 1–3 are roughly a day and should get us past M2 on their own. Do not let
anyone start on the LLM layer before item 3 is merged.
