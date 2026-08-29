"""Hard structured filters: price, colour, material, department, size. OWNER: Branch A.

Implements `src.contracts.Route`. This is the high-precision arm of the
dual-track router: when the customer has disclosed a hard constraint we filter
rather than score.
"""

from __future__ import annotations

from src.contracts import Candidate, DialogState, Product


class StructuredRoute:
    name = "structured"

    def __init__(self, products: dict[str, Product]) -> None:
        raise NotImplementedError("Branch A")

    def search(self, state: DialogState, limit: int) -> list[Candidate]:
        raise NotImplementedError("Branch A")
