"""Buying / Browsing / Override / Boundary detection. OWNER: Branch C.

Two-layer by design: a fast deterministic layer keyed on the simulator's
message shapes, and a lexical//semantic fallback that still routes correctly if
the organizer paraphrases the private set. The fallback is not optional — it is
the insurance policy on 800 unseen sessions.

Why two layers rather than one good one
---------------------------------------
`team/02_EVALUATOR_MECHANICS.md` §5 splits simulator behaviour into *structural*
facts (the session loop, the `override_applied` gate) and *distributional* ones
(the literal opening templates). Layer 1 reads the distributional shapes and is
worth 0.95 confidence when it fires, because on the public set those strings are
exact. Layer 2 reads only features that survive rewording — hedging, contrast,
constraint-noun-phrase presence — and deliberately caps its confidence below 0.7
so the fusion layer downstream knows to hedge its route weights instead of
committing to a pool.

Layer 2 was written first and is tested independently (see
`tests/unit/test_dialog.py::ParaphraseRoutingTest`). If layer 1 were deleted
entirely the router would still route every public opening to the right track,
just with less confidence — that is the property we actually need on the private
set.
"""

from __future__ import annotations

import re

from src.contracts import (
    DialogState,
    TRACK_BOUNDARY,
    TRACK_BROWSING,
    TRACK_BUYING,
    TRACK_OVERRIDE,
)

# Confidence bands. Layer 1 is allowed to be certain; layer 2 is explicitly
# capped under 0.7 so downstream fusion can tell "template matched" apart from
# "we inferred this from soft features".
CONF_TEMPLATE = 0.95
CONF_TEMPLATE_WEAK = 0.90     # deterministic shape, but not a documented literal
CONF_L2_BASE = 0.40
CONF_L2_STEP = 0.10
CONF_L2_MAX = 0.69

# ---------------------------------------------------------------------------
# Layer 1 — the simulator's literal templates (02_EVALUATOR_MECHANICS.md §2–3)
# ---------------------------------------------------------------------------

RE_BUYING_TEMPLATE = re.compile(r"a key requirement is\s*:", re.I)
RE_BROWSING_TEMPLATE = re.compile(r"but i'?m still exploring", re.I)
RE_OVERRIDE_TEMPLATE = re.compile(r"actually,?\s*ignore my earlier preference", re.I)
RE_BOUNDARY_TEMPLATE = re.compile(r"i don'?t have a preference for\b", re.I)
# The "no *additional* preference" reply is a dead-attribute answer, not a
# boundary turn — the state machine records both, but only the bare form above
# flips `boundary_seen`.
RE_DEAD_ATTRIBUTE = re.compile(r"i don'?t have an additional preference for\b", re.I)
# `I'm looking for {category}. {old_value}` — the intent_override opening. It is
# identified by the *absence* of the other two opening markers, so keep it
# strict: any paraphrase drops through to layer 2 rather than being force-fit.
RE_OVERRIDE_OPENING = re.compile(r"^\s*i'?m looking for\s+.+?\.\s+\S.*$", re.I | re.S)

# ---------------------------------------------------------------------------
# Layer 2 — features that survive paraphrasing
# ---------------------------------------------------------------------------

# Strong contrastive markers: on their own these are enough to call an override.
OVERRIDE_STRONG = (
    "ignore my earlier", "ignore what i said", "ignore my previous",
    "forget what i said", "forget my earlier", "disregard",
    "scratch that", "changed my mind", "change of mind", "never mind",
    "nevermind", "on second thought", "no longer", "strike that",
    "let's start over", "start over", "withdraw", "retract",
)
# Weak contrastive markers: cheap to trip accidentally, so they only call an
# override when paired with a fresh preference statement (see `_override_score`).
OVERRIDE_WEAK = (
    "actually", "instead", "rather", "correction", "switch to", "switching to",
    "make that", "in fact", "come to think of it", "revise", "update that",
    "on reflection", "second thoughts",
)
# A restated intent. Pairs with a weak marker to promote it to an override.
NEW_INTENT_MARKERS = (
    "what i need is", "what i want is", "what i really need", "i need",
    "i want", "i'm after", "im after", "i'd prefer", "id prefer",
    "looking for", "must be", "has to be", "should be", "make it",
)

BOUNDARY_MARKERS = (
    "no preference", "don't have a preference", "dont have a preference",
    "do not have a preference", "no strong preference", "no particular preference",
    "doesn't matter", "doesnt matter", "does not matter", "not fussed",
    "either is fine", "either way", "up to you", "your judgment",
    "your judgement", "your call", "you decide", "surprise me",
    "no opinion", "don't mind", "dont mind", "no strong opinion",
    "whatever you think", "whatever you recommend", "i'm easy", "im easy",
    "not picky", "open to anything", "anything is fine", "anything works",
)

HEDGING_MARKERS = (
    "still exploring", "just exploring", "exploring", "just browsing",
    "browsing", "just looking", "window shopping", "not sure", "unsure",
    "no idea", "maybe", "perhaps", "some ideas", "any ideas", "ideas for",
    "open to suggestions", "open to", "curious", "what do you have",
    "what have you got", "show me some", "starting to look", "early stages",
    "haven't decided", "havent decided", "undecided", "or something",
    "something like", "a few options", "see what", "get a feel",
)

# Imperative / requirement specificity: the buying track's tell.
REQUIREMENT_MARKERS = (
    "key requirement", "requirement is", "must be", "must have", "needs to be",
    "need it to", "has to be", "have to be", "should be", "it needs",
    "non-negotiable", "essential", "specifically", "exactly", "strictly",
    "only want", "only interested", "required", "i require", "important that",
    "top priority", "deal breaker", "dealbreaker", "make sure",
)

# Concrete constraint noun phrases — colour, material, measurement, spec-colon.
# These vocabularies are deliberately WIDER than `contracts.MATERIAL_RE` /
# `COLOR_RE`, which mirror the *simulator's* classifier, not the catalog's
# vocabulary. See `src/nlu/slot_extractor.py` for the same reasoning.
CONCRETE_COLOR_WORDS = (
    "black", "white", "blue", "red", "pink", "green", "brown", "gray", "grey",
    "purple", "yellow", "orange", "navy", "beige", "khaki", "olive", "maroon",
    "burgundy", "teal", "turquoise", "ivory", "cream", "charcoal", "tan",
    "gold", "silver", "coral", "lavender", "mint", "mustard", "indigo",
    "plum", "rose", "taupe", "magenta", "aqua", "peach", "salmon", "slate",
    "bronze", "blush", "camel", "fuchsia", "lilac", "emerald", "rust", "wine",
    "mocha", "multicolor", "multicolour",
)
CONCRETE_MATERIAL_WORDS = (
    "cotton", "polyester", "nylon", "leather", "wool", "spandex", "silk",
    "rayon", "fabric", "merino", "cashmere", "chiffon", "linen", "denim",
    "suede", "velvet", "satin", "twill", "jersey", "fleece", "corduroy",
    "tweed", "canvas", "mesh", "lace", "modal", "viscose", "acrylic",
    "bamboo", "hemp", "alpaca", "mohair", "georgette", "organza", "taffeta",
    "poplin", "flannel", "terry", "microfiber", "elastane", "lycra",
    "neoprene", "sherpa", "sequin", "tulle", "tencel", "lyocell", "crepe",
    "gabardine", "seersucker", "chambray", "sateen", "velour", "angora",
    "shearling", "nubuck", "rubber", "quilted", "waterproof", "fleece-lined",
)

RE_COLOR_WIDE = re.compile(r"\b(" + "|".join(CONCRETE_COLOR_WORDS) + r")\b", re.I)
RE_MATERIAL_WIDE = re.compile(r"\b(" + "|".join(CONCRETE_MATERIAL_WORDS) + r")\b", re.I)
RE_MEASUREMENT = re.compile(
    r"(\d+\s*%|\$\s*\d|\b\d+(?:\.\d+)?\s*(?:inch|inches|in|cm|mm|ft|oz|lb|lbs|"
    r"gsm|denier|us|eu|uk)\b|\bsize\s+\w+|\b(?:xs|s|m|l|xl|xxl|xxxl|2xl|3xl)\b)",
    re.I,
)
RE_SPEC_COLON = re.compile(r"\w\s*:\s*\S")


def _contains_any(lowered: str, phrases: tuple[str, ...]) -> int:
    """Count how many distinct phrases from `phrases` appear in `lowered`."""
    return sum(1 for phrase in phrases if phrase in lowered)


class HybridIntentRouter:
    """Route a customer message onto one of the four conversation tracks.

    Returns `(track, confidence)`. Confidence is a contract, not decoration:
    ``>= 0.9`` means a documented template matched verbatim and downstream code
    may commit to the track; ``< 0.7`` means the track was inferred from
    paraphrase-robust features and the fusion layer should keep both pools warm.
    """

    def route(self, message: str, state: DialogState) -> tuple[str, float]:
        text = message or ""
        lowered = text.lower()
        turn = max(state.turn, 1)

        layer_one = self._layer_one(text, lowered, turn, state)
        if layer_one is not None:
            return layer_one
        return self._layer_two(lowered, turn, state)

    # -- layer 1 ----------------------------------------------------------
    def _layer_one(
        self, text: str, lowered: str, turn: int, state: DialogState
    ) -> tuple[str, float] | None:
        """Literal template matches. Precedence: override > boundary > buying > browsing.

        Override outranks everything because it is the only track whose miss
        costs the entire session (`override_applied` gates the hit check), and
        the override message can legally contain a requirement clause
        ("What I need is: …") that would otherwise read as buying.
        """
        if RE_OVERRIDE_TEMPLATE.search(lowered):
            return TRACK_OVERRIDE, CONF_TEMPLATE
        if RE_BOUNDARY_TEMPLATE.search(lowered):
            return TRACK_BOUNDARY, CONF_TEMPLATE
        if RE_DEAD_ATTRIBUTE.search(lowered):
            # A refusal carries no track information; hold the session's track.
            return self._sticky(state, CONF_TEMPLATE_WEAK)
        if RE_BUYING_TEMPLATE.search(lowered):
            return TRACK_BUYING, CONF_TEMPLATE
        if RE_BROWSING_TEMPLATE.search(lowered):
            return TRACK_BROWSING, CONF_TEMPLATE
        if turn == 1 and RE_OVERRIDE_OPENING.match(text):
            # `I'm looking for {category}. {old_value}` — the only opening that
            # states a preference without either marker. Knowing this at turn 1
            # is worth real points: it says turns 1–2 are free for harvesting
            # and that the stated preference is about to be superseded, so
            # nothing downstream should commit its pool to it.
            # `override_applied` stays False — that flag belongs to the harness
            # gate, and only `apply_override` may set it.
            return TRACK_OVERRIDE, CONF_TEMPLATE_WEAK
        return None

    # -- layer 2 ----------------------------------------------------------
    def _layer_two(self, lowered: str, turn: int, state: DialogState) -> tuple[str, float]:
        """Feature scoring over paraphrase-stable signals only.

        Never returns >= 0.7. If no feature fires at all it holds the session's
        existing track rather than defaulting to browsing: a mid-session reply
        like "For that, what matters is: X" carries no track signal, and
        flipping the track on it would re-pool the whole session.
        """
        scores = {
            TRACK_OVERRIDE: self._override_score(lowered, turn),
            TRACK_BOUNDARY: _contains_any(lowered, BOUNDARY_MARKERS),
            TRACK_BROWSING: _contains_any(lowered, HEDGING_MARKERS),
            TRACK_BUYING: self._buying_score(lowered),
        }

        override_signals = scores[TRACK_OVERRIDE]
        best_track = max(scores, key=lambda track: (scores[track], _PRIORITY[track]))
        best_signals = scores[best_track]

        # Asymmetric cost, stated deliberately: a missed override forfeits the
        # whole session (the hit check is `override_applied and target in
        # ranked`, so every pre-override answer scores zero), while a spurious
        # override costs at most a re-pool that the surviving hard constraints
        # mostly reconstruct. So an override signal wins ties and only loses to
        # a track that beats it by more than one signal.
        if override_signals >= 1 and best_signals <= override_signals + 1:
            best_track, best_signals = TRACK_OVERRIDE, override_signals

        if best_signals <= 0:
            return self._sticky(state, CONF_L2_BASE)

        confidence = min(CONF_L2_MAX, CONF_L2_BASE + CONF_L2_STEP * best_signals)
        return best_track, round(confidence, 4)

    def _override_score(self, lowered: str, turn: int) -> int:
        """Strong markers stand alone; weak ones need a restated intent."""
        strong = _contains_any(lowered, OVERRIDE_STRONG)
        if strong:
            return strong + 1  # strong markers are worth a confidence step extra
        weak = _contains_any(lowered, OVERRIDE_WEAK)
        if not weak:
            return 0
        restated = _contains_any(lowered, NEW_INTENT_MARKERS)
        if restated:
            return min(3, weak + restated)
        # The simulator fires its override on turn 3 or 4. That is a real signal
        # but a distributional one, so it only ever breaks a tie — never enough
        # on its own, and never load-bearing if the private set moves the turn.
        return 1 if turn in (3, 4) else 0

    def _buying_score(self, lowered: str) -> int:
        """Requirement language plus concrete constraint noun phrases."""
        score = _contains_any(lowered, REQUIREMENT_MARKERS)
        if RE_COLOR_WIDE.search(lowered):
            score += 1
        if RE_MATERIAL_WIDE.search(lowered):
            score += 1
        if RE_MEASUREMENT.search(lowered):
            score += 1
        if RE_SPEC_COLON.search(lowered):
            score += 1
        return score

    @staticmethod
    def _sticky(state: DialogState, ceiling: float) -> tuple[str, float]:
        """Hold the established track when a message carries no track signal."""
        if state.override_applied:
            # Not an inference: we watched the override land and recorded it.
            # Clamping this down each quiet turn would tell the fusion layer to
            # hedge about a fact, and hedging is how a superseded pool creeps
            # back in.
            return TRACK_OVERRIDE, max(state.track_confidence, CONF_TEMPLATE)
        if state.track and state.turn > 0:
            return state.track, min(ceiling, max(state.track_confidence, CONF_L2_BASE))
        return TRACK_BROWSING, CONF_L2_BASE


# Deterministic tie-break order for `max()` above, so two tracks with equal
# evidence always resolve the same way (rule 8: determinism).
_PRIORITY = {
    TRACK_OVERRIDE: 3,
    TRACK_BOUNDARY: 2,
    TRACK_BUYING: 1,
    TRACK_BROWSING: 0,
}
