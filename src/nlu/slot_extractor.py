"""Message -> typed slots. OWNER: Branch C.

Must preserve the verbatim constraint substring alongside the typed value:
the raw string is what the exact-constraint index keys on, the typed value is
what the structured route and question policy reason over.

The one non-negotiable in this file
-----------------------------------
`Slot.value` is a byte-identical substring of the customer's message. Branch A's
constraint index is keyed on `normalize_key()` of catalog text, and the
simulator hands us `_clean_constraint()` output drawn from the target product's
own `features`/`details` (02_EVALUATOR_MECHANICS.md §3). That makes a disclosed
string a *hash lookup* onto the target, which is the single biggest precision
lever we have — and it evaporates the moment this file lowercases, lemmatises,
or trims a word off. Parsed/typed forms therefore live in separate places:
`DialogState.price_target` for money, low-confidence companion slots for the
wide colour/material vocabulary.

Vocabulary note
---------------
`contracts.MATERIAL_RE` / `COLOR_RE` mirror the *simulator's* detector. They are
the right tool for predicting how the simulator will classify a string, and the
wrong tool for reading the catalog: "navy", "merino" and "chiffon" are all
absent from them. So typing uses two passes — `classify_constraint` for the
attribute label (keeps Branch D's probe inversion honest) and a wide vocabulary
for extra retrieval evidence.
"""

from __future__ import annotations

import re

from src.contracts import (
    ALLOWED_ATTRIBUTES,
    DialogState,
    PRICE_RE,
    Slot,
    classify_constraint,
)
from src.nlu.intent_router import RE_COLOR_WIDE, RE_MATERIAL_WIDE

# Confidence bands. Anything at or below DERIVED_CONFIDENCE was inferred by us
# rather than disclosed by the customer; `src.dialog.state.askable_attributes`
# treats those as ranking evidence only, never as "this attribute is answered".
DISCLOSED_CONFIDENCE = 1.0
DERIVED_CONFIDENCE = 0.55
PROFILE_CONFIDENCE = 0.2

# Weight given to a companion slot derived from the wide vocabulary.
DERIVED_WEIGHT = 0.6

# --- payload carriers -------------------------------------------------------
# Each pattern captures the constraint payload that follows a disclosure lead-in.
# The simulator emits `"; "`-separated lists, so every payload is split on `;`.
RE_KEY_REQUIREMENT = re.compile(r"a key requirement is\s*:\s*(.+)", re.I | re.S)
RE_WHAT_MATTERS = re.compile(r"what matters is\s*:\s*(.+)", re.I | re.S)
RE_WHAT_I_NEED = re.compile(r"what i (?:really )?need is\s*:\s*(.+)", re.I | re.S)
# Paraphrase-tolerant fallbacks, tried only when none of the literals hit.
RE_SOFT_CARRIERS = (
    re.compile(r"(?:it|what)'?s? (?:important|essential) that\s*:?\s*(.+)", re.I | re.S),
    re.compile(r"(?:i (?:need|want|require)|must be|has to be|should be)\s*:\s*(.+)", re.I | re.S),
    re.compile(r"(?:my )?requirements? (?:are|is)\s*:\s*(.+)", re.I | re.S),
    re.compile(r"(?:i care about|i'?m after|priorit(?:y|ies) (?:is|are))\s*:?\s*(.+)", re.I | re.S),
)

# --- category tail ----------------------------------------------------------
RE_CATEGORY = re.compile(
    r"(?:i'?m looking for|i am looking for|looking for|shopping for|searching for|"
    r"i'?m after|i need|i want|i'?d like|help me find|in the market for|show me)\s+(.+)",
    re.I | re.S,
)
# Everything that can trail a category tail in the simulator's opening lines.
RE_CATEGORY_TRAILERS = re.compile(
    r"\s*(?:,\s*)?\b(?:but i'?m still exploring|but i am still exploring|"
    r"but i'?m just (?:browsing|looking)|though i'?m still exploring|"
    r"but still exploring|and i'?m still exploring)\b.*$",
    re.I | re.S,
)

# --- price ------------------------------------------------------------------
RE_BUDGET = re.compile(
    r"(?:budget(?:\s+is)?(?:\s+(?:around|about|of|near|approximately|roughly))?|"
    r"under|below|less than|at most|up to|no more than|max(?:imum)?|cheaper than|"
    r"within)\s*\$?\s*([0-9]+(?:\.[0-9]{1,2})?)",
    re.I,
)

# --- refusals ---------------------------------------------------------------
RE_NO_PREFERENCE = re.compile(
    r"i don'?t have (?:an additional |a |any )?preference for\s+([a-z_]+)", re.I
)
RE_NO_PREFERENCE_LOOSE = re.compile(
    r"(?:no preference|no strong preference|don'?t care|doesn'?t matter|"
    r"not fussed|don'?t mind|up to you|your (?:judgment|judgement|call))"
    r"(?:\s+(?:about|for|on|regarding|when it comes to))?\s*(?:the\s+)?([a-z_]+)?",
    re.I,
)
# Any phrasing that means "I have nothing to add on that". A refusal is
# information — it says no constraint of that class exists — but only if we
# recognise it: an unrecognised refusal makes the policy re-ask the same dead
# attribute every turn, which is how a session burns nine turns for nothing.
REFUSAL_MARKERS = (
    "no preference", "no strong preference", "no particular preference",
    "don't have a preference", "dont have a preference",
    "don't have an additional preference", "dont have an additional preference",
    "nothing else comes to mind", "nothing more comes to mind",
    "nothing else springs to mind", "nothing further", "nothing else",
    "nothing more", "can't think of anything", "cant think of anything",
    "couldn't say", "no other", "not important to me", "no opinion",
    "don't care", "dont care", "doesn't matter", "doesnt matter",
    "does not matter", "not fussed", "don't mind", "dont mind",
    "up to you", "your judgment", "your judgement", "your call",
    "you decide", "surprise me", "not picky", "anything is fine",
    "anything works", "either is fine", "either way", "no idea on",
)


def looks_like_refusal(message: str) -> bool:
    """True if the customer declined to constrain something, however phrased.

    Used as a backstop: when we can tell it is a refusal but cannot tell *of
    what*, the caller kills the attribute it just probed. That is always the
    right target — the simulator only ever refuses the attribute it was asked
    about (02_EVALUATOR_MECHANICS.md §3).
    """
    lowered = (message or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def split_constraint_payload(payload: str) -> list[str]:
    """Split a `"; "`-joined disclosure back into its verbatim members.

    Only the outer wrapping is removed: one trailing sentence period (the
    simulator appends it and `_clean_constraint` guarantees no constraint ends
    in one) and surrounding whitespace from the `"; "` join. Nothing inside a
    member is touched, which is what keeps the strings hashable against the
    catalog.
    """
    text = payload.strip()
    if text.endswith("."):
        text = text[:-1]
    return [part.strip() for part in text.split(";") if part.strip()]


def extract_category_tail(message: str) -> str | None:
    """Pull `{category}` out of an opening line, trailing clauses removed.

    The coarse category is present in every opening message of every scenario
    and is an exact catalog string, so it collapses 50k rows to a few hundred
    before a single score is computed (00_BLUEPRINT.md §2). Getting the trailing
    `, but I'm still exploring` off it is the whole job.
    """
    match = RE_CATEGORY.search(message or "")
    if not match:
        return None
    tail = match.group(1)
    tail = RE_CATEGORY_TRAILERS.sub("", tail)
    # The override opening is `I'm looking for {category}. {old_value}` and the
    # buying opening continues into another sentence, so cut at the first
    # sentence break. `coarse_category` never emits a comma (it splits on them)
    # or a period, so this cannot truncate a real tail.
    tail = re.split(r"(?<=[A-Za-z0-9\)\]])\.(?=\s|$)|[,;]", tail, maxsplit=1)[0]
    tail = tail.strip().strip(".,;: \t\n")
    return tail or None


def extract_price(message: str) -> float | None:
    """Return the budget figure as a float, or None.

    Kept out of `Slot.value` on purpose: the verbatim string (`budget around
    $49.99`) still goes to the constraint index, while the parsed number goes to
    `state.price_target` for the structured route's range filter.
    """
    match = RE_BUDGET.search(message or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    match = PRICE_RE.search(message or "")
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def extract_dead_attribute(message: str) -> str | None:
    """Return the attribute the customer just declined to constrain."""
    match = RE_NO_PREFERENCE.search(message or "")
    if match:
        candidate = match.group(1).lower()
        return candidate if candidate in ALLOWED_ATTRIBUTES else None
    loose = RE_NO_PREFERENCE_LOOSE.search(message or "")
    if loose and loose.group(1):
        candidate = loose.group(1).lower()
        if candidate in ALLOWED_ATTRIBUTES:
            return candidate
    # Paraphrase backstop: a refusal marker plus an attribute named anywhere in
    # the message. Take the last one — "nothing else comes to mind for color"
    # puts the object at the end.
    if looks_like_refusal(message):
        named = [
            word
            for word in re.findall(r"[a-z_]+", (message or "").lower())
            if word in ALLOWED_ATTRIBUTES
        ]
        if named:
            return named[-1]
    return None


class SlotExtractor:
    """Turn one customer message into `Slot` records.

    `hard=True` marks an opening requirement or a post-override intent — the
    things the customer has committed to. Anything volunteered mid-session in
    answer to a probe is soft and therefore decays; see
    `src/dialog/state.py::StateMachine.decay` for why that asymmetry matters.
    """

    def extract(self, message: str, state: DialogState) -> list[Slot]:
        text = message or ""
        turn = max(state.turn, 1)
        slots: list[Slot] = []
        seen_values: set[str] = set()

        def add(value: str, hard: bool, confidence: float = DISCLOSED_CONFIDENCE) -> None:
            value = value.strip()
            if not value or value in seen_values:
                return
            seen_values.add(value)
            # Attribute label comes from the simulator's own classifier so that
            # Branch D's inversion ("which probe unlocks this class?") stays
            # valid. Enrichment happens in companion slots below, not here.
            slots.append(
                Slot(
                    attribute=classify_constraint(value),
                    value=value,
                    turn=turn,
                    confidence=confidence,
                    hard=hard,
                    weight=1.0,
                )
            )
            slots.extend(self._companion_slots(value, turn))

        # 1. Opening requirement — hard by definition.
        match = RE_KEY_REQUIREMENT.search(text)
        if match:
            for value in split_constraint_payload(match.group(1)):
                add(value, hard=True)

        # 2. Post-override intent — hard, and it supersedes everything before it.
        match = RE_WHAT_I_NEED.search(text)
        if match:
            for value in split_constraint_payload(match.group(1)):
                add(value, hard=True)

        # 3. Probe answer — volunteered mid-session, so soft and decaying.
        match = RE_WHAT_MATTERS.search(text)
        if match:
            for value in split_constraint_payload(match.group(1)):
                add(value, hard=False)

        # 4. Paraphrase fallbacks, only if nothing literal fired.
        if not slots:
            for pattern in RE_SOFT_CARRIERS:
                match = pattern.search(text)
                if match:
                    for value in split_constraint_payload(match.group(1)):
                        add(value, hard=False, confidence=DERIVED_CONFIDENCE)
                    break

        # 5. Bare colour/material tokens in an otherwise unstructured reply.
        #    Only when no carrier fired, so we never shadow a verbatim payload.
        if not slots:
            slots.extend(self._bare_token_slots(text, turn))

        return slots

    # -- helpers ----------------------------------------------------------
    def _companion_slots(self, value: str, turn: int) -> list[Slot]:
        """Low-confidence typed companions from the wide colour/material lists.

        `classify_constraint("navy floral midi")` returns "feature" because the
        simulator's COLOR_RE has no "navy". The verbatim slot above keeps that
        label so probe prediction stays truthful; these companions give the
        structured route something to filter on anyway. They are marked
        `confidence <= DERIVED_CONFIDENCE`, which is the signal to the question
        policy that the attribute is *evidenced*, not *answered* — the customer
        may still be holding a colour constraint the simulator would disclose.
        """
        companions: list[Slot] = []
        lowered_attr = classify_constraint(value)
        color = RE_COLOR_WIDE.search(value)
        if color and lowered_attr != "color":
            companions.append(
                Slot("color", color.group(1), turn, DERIVED_CONFIDENCE, False, DERIVED_WEIGHT)
            )
        material = RE_MATERIAL_WIDE.search(value)
        if material and lowered_attr != "material":
            companions.append(
                Slot("material", material.group(1), turn, DERIVED_CONFIDENCE, False, DERIVED_WEIGHT)
            )
        return companions

    def _bare_token_slots(self, text: str, turn: int) -> list[Slot]:
        """Catch a free-text reply that names a colour or material and nothing else."""
        found: list[Slot] = []
        color = RE_COLOR_WIDE.search(text)
        if color:
            found.append(
                Slot("color", color.group(1), turn, DERIVED_CONFIDENCE, False, DERIVED_WEIGHT)
            )
        material = RE_MATERIAL_WIDE.search(text)
        if material:
            found.append(
                Slot("material", material.group(1), turn, DERIVED_CONFIDENCE, False, DERIVED_WEIGHT)
            )
        return found
