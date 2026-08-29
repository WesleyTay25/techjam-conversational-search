# Evaluator mechanics — what the simulated customer will and will not tell us

Everything here is derived from `evaluator/local_evaluator.py`, which ships to
every participant. Reading the harness we are scored by is normal engineering
practice, and none of it depends on private data. But **the public set and the
private set are not generated the same way**, and the difference decides whether
our score transfers. Read §5 before you optimise anything.

## 1. Session loop (`evaluate`, lines 189–246)

```
reset(session_id, user_profile)
turn 1..10:
    respond(session_id, user_message, turn, top_k=10)
    ranked = first 10 valid, unique, in-catalog parent_asins
    if override_applied and target in ranked:  HIT at (turn, rank); stop
    else: user_message = next simulated reply
```

Consequences we design around:

- **A question and ten guesses cost the same one turn.** Ship recommendations
  on every single turn, including turn 1, including boundary turns. The starter
  agent's MTTC of 9.81 is almost entirely this mistake.
- **Only the first 10 valid unique IDs count.** Duplicates and unknown IDs are
  dropped silently by `normalize_recommendations`, so a malformed list costs
  slots. Validate against the catalog id set before returning.
- **An exception is a miss, not a crash.** The harness swallows it
  (`except Exception`) and scores an empty response. So a bug that throws on 3%
  of sessions costs 3% of hit rate and you will never see a traceback. Log
  defensively inside `respond`.
- **`Agent` is constructed once** and reused across all sessions. Index build
  cost is amortised; per-session state must be created in `reset`.

## 2. What the opening message contains (`initial_message`, line 158)

| Scenario | Template |
|---|---|
| buying | `I'm looking for {category}. A key requirement is: {hard_constraints[0]}.` |
| browsing | `I'm looking for {category}, but I'm still exploring.` |
| intent_override | `I'm looking for {category}. {old_value}` |
| boundary | (falls through to the browsing template) |

`{category}` is `coarse_category(target.categories)` — **the last two
comma-split segments of the target product's own category path**, with the
generic "Clothing…" nodes removed. It is present in *every* opening message of
*every* scenario.

This is the highest-value signal in the whole task. It is an exact string that
exists in the catalog, so it is a hash lookup, not a search. On a 50k catalog a
tail like `Women Dresses` typically leaves a few hundred rows. Branch A builds
that index; everything downstream ranks inside it.

## 3. What later replies contain (`customer_reply`, line 172)

```python
matches = [v for v in hard_constraints + soft_preferences
           if v not in disclosed
           and (attribute == "other" or classify_constraint(v) == attribute)][:2]
```

Four things follow.

1. **Constraints are literal catalog text.** For any session where the harness
   derives the intent card (`materialize_hidden_fields`), the constraint strings
   are `_clean_constraint()` of entries from the target's own `features` and
   `details`. `src/contracts.py` mirrors that transform, so a disclosed string
   hashes straight onto the products that contain it. This is Branch A's
   constraint index and it is our single biggest precision lever.
2. **The disclosure pool is small.** `hard_constraints = cleaned[:2]` and
   `soft_preferences = cleaned[2:4]`, so at most **four** constraint strings
   exist per session, one of which the opening line already spent (buying) or
   which is about to be superseded (override). Plan for a ceiling of three or
   four useful disclosures, not ten.
3. **`ask_attribute` is a typed request, and `other` is the open probe.** The
   `attribute == "other"` branch short-circuits the class filter, so an open
   probe returns the next two undisclosed constraints whatever their class. That
   makes it the highest-expected-yield probe while the constraint set is
   unsaturated, and the *worst* one once only a specific slot is missing. Branch
   D's policy should discover this from the information-gain arithmetic rather
   than hard-coding it — a policy that just says `"other"` forever is fragile,
   unexplainable in the pitch, and dies the moment the private simulator differs.
4. **A refusal is still information.** `I don't have an additional preference
   for {attribute}` tells us no constraint of that class exists. Record it in
   `dead_attributes`, never re-ask, and stop weighting that attribute in
   ranking.

`classify_constraint` branch order is `budget -> material -> colour -> size ->
style -> use_case -> feature`, with `feature` as the catch-all. It is mirrored in
`src/contracts.py`; Branch D inverts it to predict which probe unlocks what.

## 4. Scenario-specific traps

**Intent override (15%).** `override_applied` starts `False`, and the hit check
is `if override_applied and target in ranked`. **A correct answer before the
override turn scores nothing.** The override fires on turn 3 or 4 with
`Actually, ignore my earlier preference. What I need is: {new_value}.` where
`new_value` is `hard_constraints[0]` and the discarded `old_value` was a *soft*
preference. So on an override session, turns 1–2 are free: spend them
harvesting constraints, and hard-erase the opening preference the moment the
override lands. Do not decay it — erase it. Branch C owns this.

**Boundary (5%).** The first probe is answered with `I don't have a preference
for {attribute}; please use your judgment.` exactly once per session. That turn
is unavoidable; make it cheap by having already shipped ten recommendations
with it, and fall back to profile-driven ranking.

**Browsing (40%).** The opening line carries the category tail and nothing else.
This is where the dense route and the question policy earn their keep.

## 5. The transfer risk — read this before optimising

`materialize_hidden_fields` returns the sample's own `intent_card` and
`behavior` **if the sample has them**, and only derives them from product
metadata as a fallback. The public 200 have no intent cards, so they are
derived. The private 800 may well ship real ones, and the spec explicitly
reserves the right to add organizer paraphrasing.

So the following are **structural** and safe to build on:

- the session loop, the turn budget, and the hit rule
- `ask_attribute` semantics, including the `other` open probe
- the `override_applied` gate
- disclosure being drawn from a small constraint set, two at a time

And the following are **distributional** and must never be a single point of
failure:

- constraint strings being byte-identical to catalog text
- the exact opening-message templates
- the category tail being present verbatim

Every component therefore needs graded backoff: exact key -> token-set overlap
-> dense cosine. A route that only works on exact strings is a route that scores
0.8 on the public set and 0.2 on the private one. Branch E's
`tools/paraphrase_stress.py` exists to prove we survive that; it rewrites the
public messages and re-scores, and any component that loses more than ~15% under
paraphrase gets a fallback before it gets a tuning pass.
