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
        self._products = products
        self._all_asins: frozenset[str] = frozenset(products)

    def search(self, state: DialogState, limit: int) -> list[Candidate]:
        limit = max(0, int(limit))
        if limit == 0:
            return []

        pool = set(self._all_asins)
        evidence: dict[str, float] = {}

        if state.price_target is not None:
            tolerance = max(0.01, 0.02 * state.price_target)
            filtered = {
                asin
                for asin in pool
                if self._products[asin].price is not None
                and abs(self._products[asin].price - state.price_target) <= tolerance
            }
            pool = self._soft_apply(pool, filtered, evidence, "filter_dropped_price")

        for attribute, field in (("material", "material"), ("color", "color")):
            values = self._slot_values(state, attribute)
            if not values:
                continue
            filtered = {asin for asin in pool if self._matches_attribute(self._products[asin], field, values)}
            pool = self._soft_apply(pool, filtered, evidence, f"filter_dropped_{attribute}")

        size_values = self._slot_values(state, "size")
        if size_values:
            filtered = {asin for asin in pool if self._matches_details(self._products[asin], size_values)}
            pool = self._soft_apply(pool, filtered, evidence, "filter_dropped_size")

        ranked = sorted(pool, key=lambda asin: -(self._products[asin].rating_number or 0))
        return [
            Candidate(parent_asin=asin, score=1.0, route=self.name, evidence=dict(evidence))
            for asin in ranked[:limit]
        ]

    @staticmethod
    def _soft_apply(
        pool: set[str], filtered: set[str], evidence: dict[str, float], flag: str
    ) -> set[str]:
        """Never let a filter empty the pool — drop it and flag it instead."""
        if filtered:
            return filtered
        evidence[flag] = 1.0
        return pool

    @staticmethod
    def _slot_values(state: DialogState, attribute: str) -> list[str]:
        return [
            s.value.strip().lower()
            for s in state.active_slots()
            if s.attribute == attribute and s.value and s.value.strip()
        ]

    @staticmethod
    def _matches_attribute(product: Product, field: str, values: list[str]) -> bool:
        attr_value = getattr(product, field)
        if attr_value and attr_value.lower() in values:
            return True
        # Values outside the mirrored regex vocabularies (e.g. "navy", "merino")
        # still need to match — fall back to a substring check on the blob.
        return any(value in product.text_blob for value in values)

    @staticmethod
    def _matches_details(product: Product, values: list[str]) -> bool:
        blob = " ".join(f"{k} {v}" for k, v in product.details.items()).lower()
        return any(value in blob for value in values)
