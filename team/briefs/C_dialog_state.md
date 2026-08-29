# Brief C — Intent routing, slot extraction, dialog state machine

**Branch:** `feat/dialog-state`
**Pillar:** II (multi-turn scenario evolution), III (context distillation)
**Read first:** `team/00_BLUEPRINT.md`, `team/02_EVALUATOR_MECHANICS.md` §2–4, `src/contracts.py`

## Why this branch matters

You decide what the rest of the system believes. Every route reads
`DialogState`; if a superseded preference survives an override, four routes
retrieve the wrong pool and the session is lost regardless of how good the
ranking is. 15% of sessions are override sessions and they are the ones where
naive agents score zero — `override_applied` gates the hit check, so a
pre-override answer is worth literally nothing.

## Files you own

- `src/nlu/intent_router.py`
- `src/nlu/slot_extractor.py`
- `src/dialog/state.py`
- `tests/unit/test_dialog.py`

## Task 1 — `HybridIntentRouter`, two layers

**Layer 1, deterministic.** The simulator's opening templates are documented in
`team/02_EVALUATOR_MECHANICS.md` §2. `A key requirement is:` -> buying.
`but I'm still exploring` -> browsing. `Actually, ignore my earlier preference`
-> override, mid-session. `I don't have a preference for` -> boundary. Return
confidence 0.95 on a template hit.

**Layer 2, robust — and this one is the point.** Never let layer 1 be the only
path. Score the message on features that survive rewording: presence of a
concrete constraint noun phrase, hedging language ("exploring", "not sure",
"maybe"), contrastive/negation markers ("actually", "instead", "rather",
"changed my mind"), imperative specificity. Return confidence < 0.7 so the
fusion layer knows to hedge its weights.

Assume the private set is paraphrased (spec §"If natural-language paraphrasing
is added by the organizer"). Branch E's `tools/paraphrase_stress.py` will
rewrite the public messages and re-score; your router must hold up under it.
Build layer 2 first if it helps you resist the temptation to over-fit layer 1.

Override detection is the one case where a false negative is far more expensive
than a false positive: missing it costs the whole session. Bias the threshold
accordingly and say so in a comment.

## Task 2 — `SlotExtractor`

For each message emit `Slot(attribute, value, turn, confidence, hard, weight)`.

Non-negotiable: **preserve the verbatim constraint substring** as `Slot.value`.
Branch A's index keys on the exact string; if you normalise, lowercase, or
lemmatise it before storing, that index stops working. Put the parsed/typed
form in a separate slot or in `state.price_target`, and push the raw string onto
`state.raw_constraints`.

Extraction targets:

- the constraint payload after `A key requirement is:` and after
  `what matters is:` (the simulator emits `; `-separated lists — split them);
- the category tail from `I'm looking for {X}` — strip the trailing
  `, but I'm still exploring` and any following sentence;
- price from `budget around $X` -> `state.price_target` as a float;
- colour and material tokens, but **do not restrict yourself to the mirrored
  regex vocabularies** — those cover the simulator's own detector, not the
  catalog's vocabulary. "navy", "merino", "chiffon" must survive as raw
  constraints even though `contracts.COLOR_RE` will not match them.

Mark `hard=True` for opening requirements and post-override intents;
`hard=False` for anything volunteered mid-session.

## Task 3 — `StateMachine`

**Accumulation.** New slots merge by attribute. Same attribute, different
value: keep both unless the new one is `hard`, in which case the new one
dominates. Never silently drop a disclosure — it may be the only copy of that
string we get, and there are at most four per session.

**Erasure on override.** `apply_override` sets `weight = 0.0` on superseded
slots — do not delete them; the reranker uses the erased set as a *negative*
signal, and the ablation table needs to show the difference. Promote the new
value to a `hard` slot, set `state.override_applied = True`, and clear
`state.shown` so previously rejected products can resurface under the new
intent. That last detail is easy to miss and costs hits.

**Decay.** Soft slots lose weight with age: `weight *= 0.85` per turn, floored
at 0.3. Hard slots never decay. Rationale: a browsing customer's turn-1 vague
preference should not outweigh their turn-5 specific one, but a stated hard
requirement is a requirement until it is overridden.

**Dead attributes.** On `I don't have an additional preference for {X}` or
`I don't have a preference for {X}`, add `X` to `state.dead_attributes` and set
`state.boundary_seen` for the latter. Branch D reads both; never re-ask a dead
attribute.

**Context distillation (pillar III).** Maintain a compact derived summary on the
state — active hard constraints, dead attributes, inferred track, pool size —
short enough to drop into an LLM prompt without blowing the token budget, and
useful enough that Branch D's policy reads it instead of re-deriving. Keep it
under ~200 tokens and regenerate it each turn rather than appending.

Fold `user_profile` in at `start()`: `preference_tags` become low-weight soft
slots, `average_prior_rating` and `rating_style` become a ranking prior the
reranker can read. Keep the weight genuinely low — the profile is aggregate and
anonymized, and over-trusting it will hurt.

## Acceptance

`tests/unit/test_dialog.py`, no catalog needed:

- Each of the four opening templates routes to the right track at confidence ≥ 0.9.
- Hand-paraphrased versions of all four (write them yourself, 3 per scenario) still route correctly at confidence ≥ 0.5.
- `For that, what matters is: 100% Cotton; Machine Wash Cold.` yields two slots whose `.value` are exactly `100% Cotton` and `Machine Wash Cold` — byte-identical, no normalization.
- `budget around $49.99` sets `state.price_target == 49.99`.
- `I'm looking for Women Dresses, but I'm still exploring.` sets `category_tail == "Women Dresses"` with no trailing clause.
- After `apply_override`, superseded slots have `weight == 0.0`, are still present in `state.slots`, `override_applied` is True, and `shown` is empty.
- Soft slot weight after 3 turns is `0.85**3` within 1e-6; a hard slot is still 1.0.
- A dead attribute is never returned by `active_slots()` as askable and persists across turns.
- Distilled summary stays under 200 tokens on a 10-turn session with all slots filled.

## Definition of done

See `team/01_WORKFLOW.md`. Report in your PR: intent-routing accuracy on all 200
public sessions (you can derive true labels from `scenario_type` in
`data/public_set.jsonl` — read only, never edit), and the same accuracy under
Branch E's paraphrase stress once it lands.
