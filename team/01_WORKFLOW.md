# Workflow — git, agents, merge discipline

## Day 0 setup (everyone, once)

```bash
git clone https://github.com/WesleyTay25/techjam-conversational-search
cd techjam-conversational-search

# Catalog is NOT in the repo. Get it from the official participant kit release.
# https://github.com/TechJam2026/techjam-conversational-search/releases/tag/participant-kit
gzip -dk catalog.jsonl.gz && mv catalog.jsonl data/catalog.jsonl
shasum -a 256 -c SHA256SUMS          # must pass before you trust any number

python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Baseline reproduction — this is the number we are beating.
python3 -m evaluator.local_evaluator
# expect hit_rate_at_10 0.125, mrr 0.068034, mttc 9.81, technical_score 0.10671

python3 -m unittest discover tests    # must be green before you branch
```

If the baseline does not reproduce exactly, stop and tell the lead. Every
comparison we make for the next week is against that number.

## Branch protocol

```bash
git checkout main && git pull
git checkout -b feat/<your-branch>
```

- Branch off `main` only, never off another feature branch.
- **Rebase onto `main` every morning**: `git fetch && git rebase origin/main`.
  A branch that has not rebased in 24 h is a merge conflict waiting to happen.
- Commit in small, working increments. `main` must always be green.
- Open a PR at M1 even if incomplete — early integration beats a big-bang merge.

## Working with your Claude Code agent

Each brief in `team/briefs/` is written to be handed to an agent directly.
Recommended opening prompt:

```
Read team/00_BLUEPRINT.md, team/02_EVALUATOR_MECHANICS.md, src/contracts.py,
and team/briefs/<YOUR_BRIEF>.md. Then implement only the files listed under
"Files you own" in that brief. Do not modify any other file. Run the unit tests
in your brief's "Acceptance" section before you report done.
```

Rules for your agent, restate them every session:

- It owns **only** the files listed in the brief. If it wants to change
  `src/contracts.py`, it must stop and ask you; you ask the lead.
- It must not import from `evaluator/`, and must not edit `evaluator/`,
  `data/`, or `docs/`.
- Tests must run against `tests/fixtures/mini_catalog.jsonl`, not the 50k file.
- No network calls anywhere except `src/rank/llm_reranker.py`.

## Definition of done (per branch)

A PR is mergeable when all of these hold:

1. Every stub in your owned files is implemented; no `NotImplementedError`
   remains on a code path the orchestrator reaches.
2. `python3 -m unittest discover tests` is green.
3. Your own unit tests cover the failure modes named in your brief's
   "Acceptance" section, not just the happy path.
4. `python3 -m evaluator.local_evaluator` runs 200 sessions without an
   exception and without regressing TechnicalScore.
5. Public functions have docstrings saying *why*, not *what*. Judges read this
   code; 35% of the grade is technical execution and that includes legibility.
6. No secret, no key, no `data/catalog.jsonl` in the diff.
7. PR description states: what changed, the before/after TechnicalScore, and
   any new dependency (the lead adds it to `requirements.txt` — you do not).

## Scoring runs

```bash
# full public set
python3 -m evaluator.local_evaluator --output results.json

# fast dev loop: 40-session slice, ~8x faster
python3 tools/slice.py --n 40 --out data/dev_slice.jsonl
python3 -m evaluator.local_evaluator --dataset data/dev_slice.jsonl --output results_dev.json
```

Tune on the slice, **report only on the full 200**, and never report a number
from a run where you edited the evaluator or the labels.

## Merge order at each milestone

The lead merges in dependency order so `main` is never broken:

`feat/contracts` -> `feat/retrieval-lexical` -> `feat/retrieval-dense` ->
`feat/dialog-state` -> `feat/policy-rerank` -> `feat/orchestrator` ->
`feat/llm-evalops`

After each merge the lead runs the full evaluator and posts the score to the
team channel. A merge that regresses the score gets reverted, not debugged on
`main`.
