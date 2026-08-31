"""Optional LLM semantic reranking layer. OWNER: Branch E.

Strictly opt-in (`TECHJAM_LLM=1`). Hard timeout, bounded token budget, and a
silent fall-through to `FeatureReranker` on any failure. Reports real token
counts so the harness `usage` field is honest.

Integration note (lead, at merge time)
--------------------------------------
Branch E landed this class at `rank/llm_reranker.py` — outside `src/`, so
nothing imported it — with the signature `rerank(candidates, user_context)`,
which does not satisfy `contracts.Reranker`. It has been moved to the path the
blueprint assigns it and adapted to the frozen protocol so the orchestrator can
hold it. Behaviour is unchanged from what Branch E wrote: the model call is
still a TODO, and until it lands this layer is a pass-through that defers to the
deterministic reranker. Branch E owns filling in `_call_model`.

The offline path never reaches that TODO. `TECHJAM_LLM` is unset by default, so
the official score is produced entirely by `FeatureReranker` and is reproducible
with the network down (blueprint §7 rule 3).
"""

from __future__ import annotations

import logging
import os
from typing import Sequence

from src.contracts import Candidate, DialogState

LOGGER = logging.getLogger(__name__)

# Default model for the opt-in path. Cheap and fast enough for a per-turn
# budget; the deterministic reranker is what actually produces the reported
# score, so this only ever reorders a shortlist.
DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_TIMEOUT = 3.0
# Only the head of the list is worth a model call — anything past this is not
# going to be pulled into the top 10 by a reorder.
RERANK_DEPTH = 20


def llm_enabled() -> bool:
    """True only when the operator explicitly opted in.

    Checked at call time rather than import time so a test can toggle it, and
    so an agent built in an offline harness never trips over a missing SDK.
    """
    return os.environ.get("TECHJAM_LLM") == "1"


class LLMReranker:
    """Wraps the deterministic reranker; upgrades it only when opted in.

    Composition rather than substitution: `fallback` is not an error path, it is
    the *default* path. Every failure mode — disabled, missing SDK, timeout,
    malformed response, exception — lands on exactly the same deterministic
    ordering, so enabling the flag can reorder results but can never crash a
    session or make one non-reproducible.
    """

    def __init__(self, fallback, model: str = DEFAULT_MODEL, timeout: float = DEFAULT_TIMEOUT) -> None:
        self.fallback = fallback
        self.model = model
        self.timeout = timeout
        self._prompt_tokens = 0
        self._completion_tokens = 0

    def rerank(self, state: DialogState, candidates: Sequence[Candidate]) -> list[Candidate]:
        ordered = self.fallback.rerank(state, candidates)
        if not ordered or not llm_enabled():
            return ordered
        try:
            head = ordered[:RERANK_DEPTH]
            reordered = self._call_model(state, head)
            if not reordered:
                return ordered
            # Trust the model only about the order of the shortlist it was
            # given: rebuild from our own objects and re-append the tail, so a
            # hallucinated or dropped ASIN cannot cost us a recommendation slot.
            by_asin = {c.parent_asin: c for c in head}
            promoted = [by_asin[a] for a in reordered if a in by_asin]
            promoted += [c for c in head if c.parent_asin not in set(reordered)]
            return promoted + ordered[RERANK_DEPTH:]
        except Exception as error:  # noqa: BLE001 - never let the layer break a turn
            LOGGER.warning("LLM rerank failed (%s); using deterministic order", error)
            return ordered

    def _call_model(self, state: DialogState, candidates: Sequence[Candidate]) -> list[str]:
        """Return re-ordered `parent_asin`s, or [] to keep the deterministic order.

        TODO(Branch E): issue the bounded Anthropic call here, honouring
        `self.timeout`, and add the real counts to `self._prompt_tokens` /
        `self._completion_tokens`. `state.user_profile["context_summary"]` is
        the distilled prompt context (under 200 tokens) built for exactly this.
        Returning [] is the correct no-op and is what ships today.
        """
        return []

    @property
    def usage(self) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) consumed since the last reset."""
        return self._prompt_tokens, self._completion_tokens

    def reset_usage(self) -> None:
        self._prompt_tokens = 0
        self._completion_tokens = 0
