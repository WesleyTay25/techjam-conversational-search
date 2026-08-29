"""Optional LLM semantic reranking layer. OWNER: Branch E.

Strictly opt-in (`TECHJAM_LLM=1`). Hard timeout, bounded token budget, and a
silent fall-through to `FeatureReranker` on any failure. Reports real token
counts so the harness `usage` field is honest.
"""

from __future__ import annotations

from typing import Sequence

from src.contracts import Candidate, DialogState


class LLMReranker:
    def __init__(self, fallback, model: str = "claude-haiku-4-5-20251001") -> None:
        raise NotImplementedError("Branch E")

    def rerank(self, state: DialogState, candidates: Sequence[Candidate]) -> list[Candidate]:
        raise NotImplementedError("Branch E")

    @property
    def usage(self) -> tuple[int, int]:
        """(prompt_tokens, completion_tokens) consumed since the last reset."""
        raise NotImplementedError("Branch E")
