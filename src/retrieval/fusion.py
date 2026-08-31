"""Multi-route fusion with track-dependent weights and dynamic truncation.
OWNER: Branch B.
"""

from __future__ import annotations

from typing import Mapping, Sequence

from src.contracts import (
    TRACK_BOUNDARY,
    TRACK_BROWSING,
    TRACK_BUYING,
    TRACK_OVERRIDE,
    Candidate,
    DialogState,
)

ROUTE_NAMES: tuple[str, ...] = ("constraint", "structured", "lexical", "dense")

# Starting weights per track (§Task 3 of the brief). Tune, do not treat as given.
_BASE_WEIGHTS: dict[str, dict[str, float]] = {
    TRACK_BUYING: {"constraint": 0.45, "structured": 0.25, "lexical": 0.20, "dense": 0.10},
    TRACK_BROWSING: {"constraint": 0.10, "structured": 0.10, "lexical": 0.25, "dense": 0.55},
    TRACK_BOUNDARY: {"constraint": 0.15, "structured": 0.10, "lexical": 0.25, "dense": 0.50},
}
_OVERRIDE_PRE = {"constraint": 0.20, "structured": 0.15, "lexical": 0.25, "dense": 0.40}
_OVERRIDE_POST = {"constraint": 0.50, "structured": 0.25, "lexical": 0.15, "dense": 0.10}


def reciprocal_rank_fusion(
    route_results: Mapping[str, Sequence[Candidate]],
    weights: Mapping[str, float],
    k: int = 60,
    limit: int = 100,
) -> list[Candidate]:
    """Weighted RRF: score(d) = sum_r w_r / (k + rank_r(d)).

    Rank fusion, not score blending, because routes produce incomparable
    scales (BM25 unbounded, cosine in [-1, 1], constraint is set membership).
    """
    fused: dict[str, Candidate] = {}
    for route_name, candidates in route_results.items():
        weight = weights.get(route_name, 0.0)
        if weight <= 0:
            continue
        for rank, candidate in enumerate(candidates, start=1):
            contribution = weight / (k + rank)
            entry = fused.get(candidate.parent_asin)
            if entry is None:
                entry = Candidate(parent_asin=candidate.parent_asin, score=0.0, route="fusion")
                fused[candidate.parent_asin] = entry
            entry.score += contribution
            entry.evidence[f"{route_name}_rank"] = rank
            entry.evidence[f"{route_name}_contribution"] = contribution

    ranked = sorted(fused.values(), key=lambda c: c.score, reverse=True)
    return ranked[:limit]


def _constraint_shift(hard_slot_count: int) -> float:
    """Weight migrated from `dense` to `constraint` per active hard slot.

    A session that starts browsing and accumulates hard constraints should end
    up scored like a buying session -- a continuous function of the live pool,
    not a fixed track->weights table lookup.
    """
    return min(0.08 * hard_slot_count, 0.45)


def track_weights(state: DialogState) -> dict[str, float]:
    """Route weights for the current track and turn (the 'dual-track' split)."""
    if state.track == TRACK_OVERRIDE:
        base = dict(_OVERRIDE_POST if state.override_applied else _OVERRIDE_PRE)
    else:
        base = dict(_BASE_WEIGHTS.get(state.track, _BASE_WEIGHTS[TRACK_BROWSING]))

    hard_slots = sum(1 for slot in state.active_slots() if slot.hard)
    shift = _constraint_shift(hard_slots)
    move = min(base["dense"], shift)
    base["dense"] -= move
    base["constraint"] += move

    total = sum(base.values())
    return {name: value / total for name, value in base.items()}


def dynamic_truncation(state: DialogState, pool_size: int) -> dict[str, int]:
    """Retrieval depth per route for this turn, given how converged the pool is."""
    state.pool_size = pool_size

    if pool_size > 5000:
        return {name: 50 for name in ROUTE_NAMES}
    if pool_size > 200:
        return {name: 200 for name in ROUTE_NAMES}

    depths = {name: pool_size for name in ROUTE_NAMES}
    depths["dense"] = 0
    return depths
