"""Exact-constraint, category, price and attribute indices. OWNER: Branch A.

The core insight this module exists to exploit: every constraint the simulated
customer utters is a `clean_constraint()` of a string that lives verbatim in the
target product's own `features`/`details`. Matching those as hashed keys instead
of as bag-of-words is the difference between a 3,000-row candidate pool and a
30-row one.
"""

from __future__ import annotations

from src.contracts import Candidate, DialogState, Product


class ConstraintIndex:
    def __init__(self, products: dict[str, Product]) -> None:
        raise NotImplementedError("Branch A")

    def by_constraint(self, constraint: str) -> set[str]:
        """ASINs whose flattened features/details contain this exact constraint."""
        raise NotImplementedError("Branch A")

    def by_category_tail(self, tail: str) -> set[str]:
        """ASINs whose `coarse_category()` equals `tail` (with graded backoff)."""
        raise NotImplementedError("Branch A")

    def by_price(self, price: float, tolerance: float = 0.01) -> set[str]:
        raise NotImplementedError("Branch A")

    def candidate_pool(self, state: DialogState) -> set[str]:
        """Intersect every hard signal, backing off until the pool is non-empty."""
        raise NotImplementedError("Branch A")
