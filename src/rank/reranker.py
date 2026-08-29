"""Deterministic offline feature reranker. OWNER: Branch D.

This is the scoring layer that must carry the official run when the organizer
disables network access, so it may not depend on any remote service.
"""

from __future__ import annotations

from typing import Sequence

from src.contracts import Candidate, DialogState, Product


class FeatureReranker:
    def __init__(self, products: dict[str, Product], weights: dict[str, float] | None = None) -> None:
        raise NotImplementedError("Branch D")

    def features(self, state: DialogState, candidate: Candidate) -> dict[str, float]:
        raise NotImplementedError("Branch D")

    def rerank(self, state: DialogState, candidates: Sequence[Candidate]) -> list[Candidate]:
        raise NotImplementedError("Branch D")
