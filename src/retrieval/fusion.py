"""Multi-route fusion with track-dependent weights and dynamic truncation.
OWNER: Branch B.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from src.contracts import Candidate, DialogState


def reciprocal_rank_fusion(
    route_results: Mapping[str, Sequence[Candidate]],
    weights: Mapping[str, float],
    k: int = 60,
    limit: int = 100,
) -> list[Candidate]:
    raise NotImplementedError("Branch B")


def track_weights(state: DialogState) -> dict[str, float]:
    """Route weights for the current track and turn (the 'dual-track' split)."""
    raise NotImplementedError("Branch B")


def dynamic_truncation(state: DialogState, pool_size: int) -> int:
    """How deep to retrieve this turn, given how converged the session is."""
    raise NotImplementedError("Branch B")
