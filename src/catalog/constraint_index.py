"""Exact-constraint, category, price and attribute indices. OWNER: Branch A.

The core insight this module exists to exploit: every constraint the simulated
customer utters is a `clean_constraint()` of a string that lives verbatim in the
target product's own `features`/`details`. Matching those as hashed keys instead
of as bag-of-words is the difference between a 3,000-row candidate pool and a
30-row one.
"""

from __future__ import annotations

import re
from collections import defaultdict

from src.contracts import DialogState, Product, clean_constraint, normalize_key

_TOKEN_RE = re.compile(r"\w+")
_JACCARD_THRESHOLD = 0.6


def _tokenize(key: str) -> frozenset[str]:
    return frozenset(_TOKEN_RE.findall(key))


class ConstraintIndex:
    def __init__(self, products: dict[str, Product]) -> None:
        self._products = products
        self._all_asins: frozenset[str] = frozenset(products)

        by_constraint: dict[str, set[str]] = defaultdict(set)
        by_tail: dict[str, set[str]] = defaultdict(set)
        by_price: dict[int, set[str]] = defaultdict(set)
        by_material: dict[str, set[str]] = defaultdict(set)
        by_color: dict[str, set[str]] = defaultdict(set)
        by_token: dict[str, set[str]] = defaultdict(set)

        for asin, product in products.items():
            for key in product.constraint_keys:
                by_constraint[key].add(asin)
                for token in _tokenize(key):
                    by_token[token].add(asin)

            by_tail[product.category_tail].add(asin)

            if product.price is not None:
                by_price[round(product.price * 100)].add(asin)

            if product.material:
                by_material[product.material].add(asin)
            if product.color:
                by_color[product.color].add(asin)

        self._by_constraint: dict[str, set[str]] = dict(by_constraint)
        self._by_tail: dict[str, set[str]] = dict(by_tail)
        self._by_price: dict[int, set[str]] = dict(by_price)
        self._by_material: dict[str, set[str]] = dict(by_material)
        self._by_color: dict[str, set[str]] = dict(by_color)
        self._by_token: dict[str, set[str]] = dict(by_token)

    def by_constraint(self, constraint: str) -> set[str]:
        """ASINs whose flattened features/details contain this exact constraint.

        Graded, not brittle: an exact `normalize_key` hit wins outright; failing
        that, we fall back to token-set Jaccard against the constraint keys of
        products sharing the query's rarest token (via `_by_token`, so we never
        scan the full catalog). This is what survives paraphrase and reordering
        on the private set, per `team/02_EVALUATOR_MECHANICS.md` §5.
        """
        key = normalize_key(clean_constraint(constraint))
        if not key:
            return set()

        exact = self._by_constraint.get(key)
        if exact:
            return set(exact)

        query_tokens = _tokenize(key)
        if not query_tokens:
            return set()

        candidate_tokens = [t for t in query_tokens if t in self._by_token]
        if not candidate_tokens:
            return set()
        rare_token = min(candidate_tokens, key=lambda t: len(self._by_token[t]))

        matched: set[str] = set()
        for asin in self._by_token[rare_token]:
            for candidate_key in self._products[asin].constraint_keys:
                candidate_key_tokens = _tokenize(candidate_key)
                union = query_tokens | candidate_key_tokens
                if not union:
                    continue
                jaccard = len(query_tokens & candidate_key_tokens) / len(union)
                if jaccard >= _JACCARD_THRESHOLD:
                    matched.add(asin)
                    break
        return matched

    def by_category_tail(self, tail: str) -> set[str]:
        """ASINs whose `coarse_category()` equals `tail` (with graded backoff).

        Backs off `"Women Dresses"` -> `"Dresses"` -> unrestricted, because
        category paths in the catalog are messy enough that an exact-only tail
        lookup silently returns nothing on some sessions.
        """
        cleaned = clean_constraint(tail) if tail else ""
        if not cleaned:
            return set(self._all_asins)

        exact = self._by_tail.get(cleaned)
        if exact:
            return set(exact)

        words = cleaned.split()
        last_word = words[-1].lower() if words else ""
        if last_word:
            backoff: set[str] = set()
            for indexed_tail, asins in self._by_tail.items():
                indexed_words = indexed_tail.split()
                if indexed_words and indexed_words[-1].lower() == last_word:
                    backoff |= asins
            if backoff:
                return backoff

        return set(self._all_asins)

    def by_price(self, price: float, tolerance: float = 0.01) -> set[str]:
        try:
            price = float(price)
        except (TypeError, ValueError):
            return set()

        key = round(price * 100)
        exact = self._by_price.get(key)
        if exact:
            return set(exact)

        tolerance_cents = max(0, round(tolerance * 100))
        if tolerance_cents == 0:
            return set()

        matched: set[str] = set()
        for indexed_key, asins in self._by_price.items():
            if abs(indexed_key - key) <= tolerance_cents:
                matched |= asins
        return matched

    def candidate_pool(self, state: DialogState) -> set[str]:
        """Intersect every hard signal, backing off until the pool is non-empty."""
        if state.category_tail:
            pool = self.by_category_tail(state.category_tail)
        else:
            pool = set(self._all_asins)

        constraints = [c for c in state.raw_constraints if c and c.strip()]
        constraints.sort(key=lambda c: len(self.by_constraint(c)))

        for constraint in constraints:
            matched = self.by_constraint(constraint)
            if not matched:
                continue
            intersected = pool & matched
            if intersected:
                pool = intersected
            # else: this constraint contradicts the pool so far — skip it
            # rather than collapsing to zero candidates on a wasted turn.

        state.pool_size = len(pool)
        return pool
