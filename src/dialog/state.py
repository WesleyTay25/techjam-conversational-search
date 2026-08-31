"""Conversational state machine: accumulation, decay, override erasure.
OWNER: Branch C.

This module decides what the rest of the system believes. Every retrieval route
reads `DialogState`, so a superseded preference that survives an override
re-pools four routes onto the wrong few hundred rows and the session is lost no
matter how good the ranking is. 15% of sessions are override sessions and the
harness gates its hit check on `override_applied`, which means a correct answer
delivered before the override turn is worth exactly zero.

Three invariants hold everywhere below:

1. **Nothing is deleted.** At most four constraint strings exist per session
   (02_EVALUATOR_MECHANICS.md §3), so a dropped disclosure may be the only copy
   we ever get. Superseded slots are zeroed, not removed — the reranker reads
   the erased set as a negative signal and the ablation table needs the
   difference to be visible.
2. **`raw_constraints` is the live retrieval key set; `slots` is the audit
   trail.** They diverge exactly once, at override: the superseded strings leave
   `raw_constraints` (or Branch A keeps hashing onto the wrong products) and
   stay in `slots` at weight 0.0.
3. **Hard requirements do not decay.** A stated requirement is a requirement
   until it is overridden; only volunteered soft preferences age.

Where the distilled summary lives
---------------------------------
`src/contracts.py` is frozen and has no field for a context summary, so
`distill()` is a pure function and `ingest()` caches its output into
`state.user_profile["context_summary"]` — `user_profile` is copied at `start()`,
so the harness's own dict is never mutated. Branch D should read
`distill(state)` (or that cached key) instead of re-deriving the session.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

from src.contracts import (
    ALLOWED_ATTRIBUTES,
    DialogState,
    Slot,
    TRACK_BOUNDARY,
    TRACK_OVERRIDE,
    classify_constraint,
)
from src.nlu.intent_router import HybridIntentRouter
from src.nlu.slot_extractor import (
    DERIVED_CONFIDENCE,
    PROFILE_CONFIDENCE,
    RE_WHAT_I_NEED,
    SlotExtractor,
    extract_category_tail,
    extract_dead_attribute,
    extract_price,
    looks_like_refusal,
    split_constraint_payload,
)

DECAY_RATE = 0.85
DECAY_FLOOR = 0.30
PROFILE_TAG_WEIGHT = 0.15   # aggregate and anonymized: real, but never load-bearing
SUMMARY_TOKEN_BUDGET = 200
CATEGORY_ATTRIBUTE = "category"

# Slots created at turn 0 come from the user profile rather than the
# conversation. They sit below the decay floor on purpose, so they are exempt
# from ageing — decaying them would drag them *up* to the floor.
PROFILE_TURN = 0


# ---------------------------------------------------------------------------
# Context distillation (pillar III)
# ---------------------------------------------------------------------------

def approx_tokens(text: str) -> int:
    """Deliberately pessimistic token estimate: max(words, chars/4).

    We never see the tokenizer Branch E's LLM layer will use, so overestimating
    is the only safe direction for a budget check.
    """
    if not text:
        return 0
    return max(len(text.split()), math.ceil(len(text) / 4))


def _fmt(values: Iterable[str], limit: int, width: int = 48) -> str:
    items = [v if len(v) <= width else v[: width - 1] + "…" for v in list(values)[:limit]]
    return " | ".join(items)


def distill(state: DialogState, budget: int = SUMMARY_TOKEN_BUDGET) -> str:
    """A compact, regenerated-each-turn view of the session.

    Regenerated rather than appended so it cannot grow with turn count: on a
    ten-turn session with every slot filled this stays well under `budget`
    tokens, which is what makes it safe to paste into an LLM prompt. It carries
    the four things a downstream policy actually needs — active hard
    constraints, dead attributes, inferred track, live pool size — so Branch D
    reads this instead of walking `state.slots` itself.
    """
    active = [s for s in state.active_slots() if s.attribute != CATEGORY_ATTRIBUTE]
    hard = [s.value for s in active if s.hard]
    soft = [
        f"{s.value} ({s.weight:.2f})"
        for s in sorted(active, key=lambda s: -s.weight)
        if not s.hard and s.confidence > PROFILE_CONFIDENCE
    ]
    erased = [
        s.value
        for group in state.slots.values()
        for s in group
        if s.weight <= 0.0
    ]
    profile = state.user_profile or {}
    tags = [str(t) for t in (profile.get("preference_tags") or [])]

    lines = [
        f"track={state.track} conf={state.track_confidence:.2f} turn={state.turn} "
        f"pool={state.pool_size} override={'yes' if state.override_applied else 'no'}",
        f"category: {state.category_tail or 'unknown'}",
    ]
    # Budget-ordered: the lines most likely to change the next pool come first,
    # so the trim loop below sheds the cheapest context, not the load-bearing bits.
    if hard:
        lines.append(f"must: {_fmt(hard, 6)}")
    if soft:
        lines.append(f"prefer: {_fmt(soft, 6)}")
    if state.price_target is not None:
        lines.append(f"budget<={state.price_target:.2f}")
    if state.dead_attributes:
        lines.append(f"ruled out: {', '.join(sorted(state.dead_attributes))}")
    if state.asked:
        lines.append(f"asked: {', '.join(dict.fromkeys(state.asked))}")
    if erased:
        lines.append(f"superseded (do not retrieve): {_fmt(erased, 4)}")
    if tags:
        lines.append(f"profile(weak): {', '.join(tags[:4])}")
    if state.boundary_seen:
        lines.append("customer deferred once; lean on profile prior")

    summary = "\n".join(lines)
    # Hard guarantee rather than a hope: shed optional lines from the bottom
    # until the estimate fits, so no session can blow the prompt budget.
    while approx_tokens(summary) > budget and len(lines) > 2:
        lines.pop()
        summary = "\n".join(lines)
    return summary


def askable_attributes(state: DialogState) -> list[str]:
    """Attributes still worth a probe, in `ALLOWED_ATTRIBUTES` order.

    Excludes anything the customer declined (`dead_attributes`) and anything
    already answered by a *disclosed* slot. Slots we merely inferred
    (confidence <= DERIVED_CONFIDENCE, e.g. a "navy" companion) do not count as
    answered — the simulator may still be holding a constraint of that class,
    and its classifier does not agree with our wide vocabulary about what
    "navy" is.

    `asked` is deliberately *not* excluded. The open `other` probe short-circuits
    the simulator's class filter and returns the next two undisclosed
    constraints whatever their class (02_EVALUATOR_MECHANICS.md §3), so it stays
    productive on repeat; retiring it after one use would throw away the
    highest-yield probe we have. Deciding when it stops paying is Branch D's
    information-gain arithmetic, not ours.
    """
    answered = {
        s.attribute
        for s in state.active_slots()
        if s.confidence > DERIVED_CONFIDENCE and s.attribute != CATEGORY_ATTRIBUTE
    }
    spent = set(state.dead_attributes) | answered
    return [a for a in ALLOWED_ATTRIBUTES if a not in spent and a != CATEGORY_ATTRIBUTE]


class StateMachine:
    """Owns the lifecycle of one `DialogState` per session.

    Stateless between sessions by design — the harness constructs `Agent` once
    and reuses it across all 200 sessions, so anything cached on this object
    would leak one customer's constraints into the next customer's pool.
    """

    def __init__(
        self,
        router: HybridIntentRouter | None = None,
        extractor: SlotExtractor | None = None,
    ) -> None:
        self.router = router or HybridIntentRouter()
        self.extractor = extractor or SlotExtractor()

    # -- lifecycle --------------------------------------------------------
    def start(self, session_id: str, user_profile: dict) -> DialogState:
        """Open a session and fold the anonymized profile in at low weight.

        The profile is aggregate — "prior purchases emphasize fit, comfort" is
        true of a population, not of the one product we have to find — so its
        tags enter as weight-0.15 soft slots and its ratings enter as a ranking
        prior the reranker may consult. Over-trusting it actively hurts: it
        would outvote a real disclosure, and there are only ever four of those.
        """
        profile = dict(user_profile or {})
        state = DialogState(session_id=session_id, user_profile=profile)

        for tag in profile.get("preference_tags") or []:
            text = str(tag).strip()
            if not text:
                continue
            self._merge(
                state,
                Slot(
                    attribute=classify_constraint(text),
                    value=text,
                    turn=PROFILE_TURN,
                    confidence=PROFILE_CONFIDENCE,
                    hard=False,
                    weight=PROFILE_TAG_WEIGHT,
                ),
                register_raw=False,   # aggregate tags are not catalog strings
            )

        # Reachable through the frozen contract without adding a field.
        profile["ranking_prior"] = self._ranking_prior(profile)
        profile["context_summary"] = distill(state)
        return state

    def ingest(self, state: DialogState, message: str, turn: int) -> DialogState:
        """Route the message, extract slots, apply accumulation or erasure."""
        text = message or ""
        state.turn = int(turn)
        state.history.append(text)

        track, confidence = self.router.route(text, state)

        # Age existing soft slots before merging this turn's disclosures, so a
        # brand-new slot is never decayed on the turn it arrives.
        self.decay(state)

        # 1. Override first: it rewrites everything else this turn would have done.
        if track == TRACK_OVERRIDE and not state.override_applied:
            new_value = self._override_payload(text)
            if new_value is not None:
                self.apply_override(state, new_value)
                state.track_confidence = max(confidence, state.track_confidence)
                self._finish_turn(state)
                return state
            # An override *opening* (`I'm looking for X. {old_value}`) announces
            # the session type but supersedes nothing yet — fall through and
            # accumulate normally, just with the track set.

        state.track = track
        state.track_confidence = confidence
        if track == TRACK_BOUNDARY:
            state.boundary_seen = True

        # 2. Refusals. A refusal is information: it says no constraint of that
        #    class exists, so we must never spend another turn asking for it.
        dead = extract_dead_attribute(text)
        if dead is None and looks_like_refusal(text) and state.asked:
            # We can tell it is a refusal but not of what. The simulator only
            # ever refuses the attribute it was just asked about, so kill that
            # one rather than letting the policy re-probe it every turn.
            dead = state.asked[-1]
        if dead:
            state.dead_attributes.add(dead)
            if dead not in state.asked:
                state.asked.append(dead)

        # 3. The category tail is the pool key and arrives exactly once, in the
        #    opening line. Later "I need ..." phrasings must not overwrite it.
        if state.category_tail is None:
            tail = extract_category_tail(text)
            if tail:
                state.category_tail = tail
                self._merge(
                    state,
                    Slot(CATEGORY_ATTRIBUTE, tail, state.turn, confidence, True, 1.0),
                    # `raw_constraints` is Branch A's *feature/detail* key set;
                    # the category has its own field and its own index
                    # (`by_category_tail`). Registering it in both makes
                    # `candidate_pool` re-query the tail as a feature string,
                    # which either no-ops or, worse, intersects the pool down to
                    # whichever products happen to mention it in their details.
                    register_raw=False,
                )

        # 4. Price parses into a typed field; the verbatim string still becomes
        #    a slot below so the constraint index keeps its exact key.
        price = extract_price(text)
        if price is not None:
            state.price_target = price

        # 5. Disclosed constraints.
        for slot in self.extractor.extract(text, state):
            self._merge(state, slot)

        self._finish_turn(state)
        return state

    # -- override ---------------------------------------------------------
    def apply_override(self, state: DialogState, new_value: str) -> None:
        """Zero the weight of superseded slots and promote the new intent.

        Erase, do not decay: the customer said "ignore my earlier preference",
        and a decayed-but-nonzero preference still drags the pool. Three details
        that are easy to miss and each cost hits:

        * The **category tail survives**. The override replaces a preference,
          not the product class; erasing the pool key would be fatal.
        * `shown` is **cleared**, so products we already surfaced and the
          customer implicitly passed on can resurface under the new intent —
          under the old intent they were wrong, under the new one one of them
          may be the target.
        * Superseded strings leave `raw_constraints` but stay in `slots` at
          weight 0.0, because Branch A retrieves from the former and the
          reranker reads the latter as a negative signal.
        """
        superseded: list[str] = []
        for group in state.slots.values():
            for slot in group:
                if slot.attribute == CATEGORY_ATTRIBUTE:
                    continue
                if slot.turn == PROFILE_TURN:
                    # Profile priors were never "the earlier preference"; they
                    # are aggregate background and stay at their low weight.
                    continue
                if slot.weight > 0.0:
                    superseded.append(slot.value)
                slot.weight = 0.0

        superseded_set = set(superseded)
        state.raw_constraints = [v for v in state.raw_constraints if v not in superseded_set]
        # Any budget we parsed came from a now-superseded disclosure (the
        # override opening carries no price), so drop it and let the new value
        # re-establish one if it carries a figure.
        state.price_target = None

        for value in split_constraint_payload(new_value):
            self._merge(
                state,
                Slot(
                    attribute=classify_constraint(value),
                    value=value,
                    turn=max(state.turn, 1),
                    confidence=1.0,
                    hard=True,
                    weight=1.0,
                ),
            )
            price = extract_price(value)
            if price is not None:
                state.price_target = price

        state.override_applied = True
        state.track = TRACK_OVERRIDE
        state.shown.clear()

    # -- decay ------------------------------------------------------------
    def decay(self, state: DialogState) -> None:
        """Age soft slots so stale preferences stop dominating late turns.

        A browsing customer's vague turn-1 preference should not outweigh their
        specific turn-5 one. Hard slots are exempt — a stated requirement stands
        until it is overridden — and erased slots stay at 0.0, since raising
        them back to the floor would undo the override.
        """
        for group in state.slots.values():
            for slot in group:
                if slot.hard or slot.weight <= 0.0:
                    continue
                if slot.turn == PROFILE_TURN or slot.turn >= state.turn:
                    continue
                slot.weight = max(DECAY_FLOOR, slot.weight * DECAY_RATE)

    # -- helpers ----------------------------------------------------------
    def _merge(self, state: DialogState, slot: Slot, register_raw: bool = True) -> None:
        """Accumulate one slot under its attribute.

        Same attribute + same value: refresh in place rather than duplicating.
        Same attribute + different value: keep both — never silently drop a
        disclosure — unless the newcomer is hard, in which case the incumbent
        soft values are demoted to the decay floor so the hard one dominates
        ranking while remaining on the record.
        """
        group = state.slots.setdefault(slot.attribute, [])
        for existing in group:
            if existing.value == slot.value:
                existing.turn = max(existing.turn, slot.turn)
                existing.confidence = max(existing.confidence, slot.confidence)
                existing.hard = existing.hard or slot.hard
                if existing.weight > 0.0:
                    existing.weight = max(existing.weight, slot.weight)
                if register_raw:
                    self._register_raw(state, slot.value)
                return

        if slot.hard:
            for existing in group:
                if not existing.hard and existing.weight > DECAY_FLOOR:
                    existing.weight = DECAY_FLOOR

        group.append(slot)
        if register_raw:
            self._register_raw(state, slot.value)

    @staticmethod
    def _register_raw(state: DialogState, value: str) -> None:
        """Verbatim strings, deduped, order preserved — Branch A's index keys."""
        if value and value not in state.raw_constraints:
            state.raw_constraints.append(value)

    @staticmethod
    def _override_payload(message: str) -> str | None:
        """The new intent carried by an override message, or None if there is none.

        `Actually, ignore my earlier preference. What I need is: {new_value}.`
        is the simulator's literal form. A paraphrase that keeps a colon-led
        payload still parses; an override *opening*, which carries no new value
        at all, returns None so the caller knows nothing was superseded yet.
        """
        match = RE_WHAT_I_NEED.search(message or "")
        if match:
            return match.group(1)
        # Paraphrase fallback: take the clause after the contrastive marker if
        # it reads as a fresh requirement rather than a bare apology.
        for marker in ("what i need is", "i need", "i want", "make it", "switch to"):
            index = (message or "").lower().find(marker)
            if index >= 0:
                payload = message[index + len(marker):].lstrip(" :,-")
                if payload.strip():
                    return payload
        return None

    @staticmethod
    def _ranking_prior(profile: dict) -> dict:
        """Turn the anonymized profile into a small, bounded ranking prior.

        `rating_style` tells us how to read `average_prior_rating`: a critical
        rater's 3.5 is not a lenient rater's 3.5. The reranker uses this to nudge
        ties, never to reorder a constraint match — hence the tight bounds.
        """
        style = str(profile.get("rating_style") or "").lower()
        try:
            average = float(profile.get("average_prior_rating"))
        except (TypeError, ValueError):
            average = None
        if "critical" in style:
            tolerance = 0.6      # hard rater: their 4.0 is the market's 4.5
        elif "positive" in style or "lenient" in style:
            tolerance = -0.3
        else:
            tolerance = 0.0
        return {
            "average_prior_rating": average,
            "rating_style": style or "unknown",
            # Minimum catalog rating this customer is likely to accept.
            "min_rating_hint": None if average is None else round(
                min(4.6, max(3.0, average + tolerance)), 2
            ),
            "weight": 0.15,      # ties only; see PROFILE_TAG_WEIGHT
        }

    def _finish_turn(self, state: DialogState) -> None:
        """Regenerate the distilled context. Cheap, and it keeps it never-stale."""
        summary = distill(state)
        if isinstance(state.user_profile, dict):
            state.user_profile["context_summary"] = summary

    # -- read helpers for downstream branches -----------------------------
    @staticmethod
    def summary(state: DialogState) -> str:
        """The current distilled context (pillar III)."""
        return distill(state)

    @staticmethod
    def askable(state: DialogState) -> list[str]:
        """Attributes still worth spending a probe on."""
        return askable_attributes(state)

    @staticmethod
    def erased_slots(state: DialogState) -> list[Slot]:
        """Superseded slots, kept as a negative signal for the reranker."""
        return [s for group in state.slots.values() for s in group if s.weight <= 0.0]

    @staticmethod
    def mark_shown(state: DialogState, parent_asins: Sequence[str]) -> None:
        """Record what we surfaced, so the policy can diversify next turn."""
        for asin in parent_asins:
            if asin not in state.shown:
                state.shown.append(asin)
