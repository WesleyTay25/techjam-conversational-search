"""Frozen data contracts for the conversational shopping agent.

FROZEN AFTER D0. Every feature branch compiles against this file, so any change
requires lead sign-off called out in the PR description. Add new fields with
defaults; never rename or remove one.

Nothing in `src/` may import from `evaluator/` — the submission must stand alone
when the organizer swaps in the private harness. Where we need to reason about
simulator behaviour we MIRROR the logic here and note the source function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Mapping, Protocol, Sequence

MAX_TURNS = 10
TOP_K = 10

ALLOWED_ATTRIBUTES: tuple[str, ...] = (
    "category", "material", "color", "size", "style",
    "brand", "budget", "feature", "use_case", "other",
)

# Conversation tracks produced by the intent router.
TRACK_BUYING = "buying"
TRACK_BROWSING = "browsing"
TRACK_OVERRIDE = "intent_override"
TRACK_BOUNDARY = "boundary"
TRACKS = (TRACK_BUYING, TRACK_BROWSING, TRACK_OVERRIDE, TRACK_BOUNDARY)

MATERIAL_WORDS = (
    "cotton", "polyester", "nylon", "leather", "wool",
    "spandex", "silk", "rayon", "fabric",
)
COLOR_WORDS = (
    "black", "white", "blue", "red", "pink", "green",
    "brown", "gray", "grey", "purple", "yellow", "orange",
)

MATERIAL_RE = re.compile(r"\b(" + "|".join(MATERIAL_WORDS) + r")\b", re.I)
COLOR_RE = re.compile(r"\b(" + "|".join(COLOR_WORDS) + r")\b", re.I)
PRICE_RE = re.compile(r"\$\s*([0-9]+(?:\.[0-9]{1,2})?)")
_WS_RE = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Simulator mirrors. Keep byte-identical to the referenced evaluator function.
# ---------------------------------------------------------------------------

def clean_constraint(value: str, limit: int = 180) -> str:
    """Mirror of `evaluator.local_evaluator._clean_constraint`.

    The simulator emits constraint strings through this filter, so replicating
    it lets us build an index whose keys are exactly the strings we will be
    handed at conversation time.
    """
    return _WS_RE.sub(" ", value).strip(" -;,.\t\n")[:limit].rstrip()


def flatten_values(value: object) -> list[str]:
    """Mirror of `evaluator.local_evaluator._flatten_values`."""
    if isinstance(value, dict):
        return [f"{key}: {item}" for key, item in value.items() if item not in (None, "", [])]
    if isinstance(value, list):
        return [str(item) for item in value if item not in (None, "")]
    return [str(value)] if value not in (None, "") else []


def coarse_category(values: Sequence[str]) -> str:
    """Mirror of `evaluator.local_evaluator.coarse_category`.

    The customer's opening line embeds this string verbatim for every scenario,
    which makes it the single highest-recall prior we ever receive.
    """
    excluded = {"clothing", "clothing shoes & jewelry", "clothing, shoes & jewelry"}
    cleaned: list[str] = []
    for value in values:
        for part in str(value).split(","):
            part = part.strip()
            if part and part.lower() not in excluded:
                cleaned.append(part)
    return " ".join(cleaned[-2:]) if cleaned else "clothing item"


def classify_constraint(value: str) -> str:
    """Mirror of `evaluator.local_evaluator.classify_constraint`.

    Inverting this router is how the question policy predicts which probe will
    unlock which residual constraint. Branch order is significant.
    """
    lowered = value.lower()
    if "budget" in lowered or re.search(r"(?:\$|<=|under)\s*\d", lowered):
        return "budget"
    if any(material in lowered for material in MATERIAL_WORDS):
        return "material"
    if any(word in lowered for word in ("color", "black", "white", "blue", "red", "pink", "green")):
        return "color"
    if any(word in lowered for word in ("size", "sizing", "width", "wide", "narrow")):
        return "size"
    if any(word in lowered for word in ("department", "style", "fit", "sleeve", "neck")):
        return "style"
    if any(word in lowered for word in ("hiking", "running", "gym", "winter", "outdoor", "work")):
        return "use_case"
    return "feature"


def normalize_key(value: str) -> str:
    """Case/whitespace-insensitive form used as the constraint-index key."""
    return _WS_RE.sub(" ", str(value)).strip(" -;,.\t\n").lower()


# ---------------------------------------------------------------------------
# Core records
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Product:
    """One catalog row, normalized once at start-up and never mutated.

    The catalog is read-only per the rules; every derived structure here is an
    index over it, not a modification of it.
    """

    parent_asin: str
    title: str
    features: tuple[str, ...]
    description: tuple[str, ...]
    categories: tuple[str, ...]
    details: Mapping[str, str]
    store: str
    price: float | None
    average_rating: float | None
    rating_number: int
    # Derived --------------------------------------------------------------
    text_blob: str                      # lowercased concatenation of all text fields
    constraint_keys: frozenset[str]     # normalize_key() over flattened features+details
    category_tail: str                  # coarse_category(categories)
    material: str | None                # first MATERIAL_RE hit in text_blob
    color: str | None                   # first COLOR_RE hit in text_blob


@dataclass(slots=True)
class Candidate:
    """A scored catalog row on its way through the pipeline."""

    parent_asin: str
    score: float
    route: str = ""
    # Per-route or per-feature contributions, kept for the reranker and for the
    # customer-facing explanation string.
    evidence: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class Slot:
    """One piece of disclosed customer intent."""

    attribute: str          # member of ALLOWED_ATTRIBUTES, or "category"
    value: str              # verbatim text as disclosed
    turn: int
    confidence: float = 1.0
    hard: bool = False      # opening requirement or post-override intent
    weight: float = 1.0     # decayed by the state machine; 0.0 == erased


@dataclass(slots=True)
class DialogState:
    """Per-session working memory. Owned by `src/dialog/state.py`."""

    session_id: str
    user_profile: dict = field(default_factory=dict)
    track: str = TRACK_BROWSING
    track_confidence: float = 0.0
    turn: int = 0

    slots: dict[str, list[Slot]] = field(default_factory=dict)
    raw_constraints: list[str] = field(default_factory=list)   # verbatim disclosed strings
    category_tail: str | None = None
    price_target: float | None = None

    asked: list[str] = field(default_factory=list)             # probes already spent
    dead_attributes: set[str] = field(default_factory=set)     # answered "no preference"
    boundary_seen: bool = False
    override_applied: bool = False

    shown: list[str] = field(default_factory=list)             # asins already surfaced
    pool_size: int = 0                                         # candidates after filtering
    history: list[str] = field(default_factory=list)           # raw customer messages

    def active_slots(self) -> list[Slot]:
        return [s for group in self.slots.values() for s in group if s.weight > 0.0]


@dataclass(slots=True)
class TurnResult:
    """What the orchestrator hands back to the harness adapter."""

    message: str
    ask_attribute: str | None
    candidates: list[Candidate]
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def to_payload(self) -> dict:
        """Render the exact `respond()` shape from docs/agent_api_contract.json."""
        attribute = self.ask_attribute if self.ask_attribute in ALLOWED_ATTRIBUTES else None
        return {
            "message": str(self.message),
            "ask_attribute": attribute,
            "recommendations": [
                {"parent_asin": str(c.parent_asin)} for c in self.candidates[:TOP_K]
            ],
            "usage": {
                "prompt_tokens": max(0, int(self.prompt_tokens)),
                "completion_tokens": max(0, int(self.completion_tokens)),
            },
        }


# ---------------------------------------------------------------------------
# Component protocols. Each feature branch implements one of these.
# ---------------------------------------------------------------------------

class Route(Protocol):
    """A retrieval route. Pure: reads state, returns scored candidates."""

    name: str

    def search(self, state: DialogState, limit: int) -> list[Candidate]: ...


class IntentRouter(Protocol):
    def route(self, message: str, state: DialogState) -> tuple[str, float]:
        """Return (track, confidence) for the current message."""


class SlotExtractor(Protocol):
    def extract(self, message: str, state: DialogState) -> list[Slot]: ...


class QuestionPolicy(Protocol):
    def choose(
        self, state: DialogState, pool: Sequence[Product]
    ) -> tuple[str | None, str]:
        """Return (ask_attribute, customer-facing question text)."""


class Reranker(Protocol):
    def rerank(
        self, state: DialogState, candidates: Sequence[Candidate]
    ) -> list[Candidate]: ...
