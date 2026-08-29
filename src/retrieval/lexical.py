"""BM25 lexical route over SQLite FTS5. OWNER: Branch A.

Implements `src.contracts.Route`. Upgrades the starter's flat OR-of-unigrams
into phrase-aware, field-weighted, pool-restricted retrieval.
"""

from __future__ import annotations

from src.contracts import Candidate, DialogState, Product


class LexicalRoute:
    name = "lexical"

    def __init__(self, products: dict[str, Product]) -> None:
        raise NotImplementedError("Branch A")

    def search(self, state: DialogState, limit: int) -> list[Candidate]:
        raise NotImplementedError("Branch A")
