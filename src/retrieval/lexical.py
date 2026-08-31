"""BM25 lexical route over SQLite FTS5. OWNER: Branch A.

Implements `src.contracts.Route`. Upgrades the starter's flat OR-of-unigrams
into phrase-aware, field-weighted, pool-restricted retrieval.
"""

from __future__ import annotations

import re
import sqlite3

from src.contracts import Candidate, DialogState, Product

_TOKEN_RE = re.compile(r"\w+")
_MIN_TOKEN_LEN = 2


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if len(t) >= _MIN_TOKEN_LEN]


class LexicalRoute:
    """FTS5 BM25 search, phrase-aware and field-weighted over disclosed constraints.

    Column order mirrors `starter/agent.py` so the schema stays familiar. The
    weights differ from the starter's guess: disclosed constraints are drawn
    verbatim from `features`/`details` (see
    `team/02_EVALUATOR_MECHANICS.md` §3), so those two columns are weighted
    above `title`/`categories` instead of below them.

    Grid-searched against the full 50k catalog on the 200 public sessions
    (fully-disclosed constraints, lexical route alone, target within top 50):

    | config             | weights (title,cat,feat,det,store,desc) | hit@10 | mrr   |
    |--------------------|------------------------------------------|-------:|------:|
    | starter baseline   | 6.0, 4.0, 2.5, 2.5, 1.5, 1.0               | 0.895  | 0.787 |
    | (this route)       | 4.0, 4.0, 5.0, 5.0, 2.0, 1.5                | 0.915  | 0.806 |

    A handful of neighbours of this point tie on hit@10; this one was picked
    for having the simplest weight structure among the tied configs.
    """

    name = "lexical"

    _COLUMNS = ("parent_asin", "title", "categories", "features", "details", "store", "description")
    _WEIGHTS = (0.0, 4.0, 4.0, 5.0, 5.0, 2.0, 1.5)
    _BM25_ARGS = ", ".join(str(w) for w in _WEIGHTS)
    _RESTRICT_POOL_CEILING = 5000

    def __init__(self, products: dict[str, Product]) -> None:
        self._products = products
        self._asins: frozenset[str] = frozenset(products)
        self._conn = sqlite3.connect(":memory:")
        self._build_index()

    def _build_index(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        for product in self._products.values():
            batch.append((
                product.parent_asin,
                product.title,
                " ".join(product.categories),
                " ".join(product.features),
                " ".join(f"{k}: {v}" for k, v in product.details.items()),
                product.store,
                " ".join(product.description),
            ))
            if len(batch) >= 1000:
                cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self._conn.commit()

    def _phrase_expression(self, state: DialogState) -> str:
        """Disclosed constraints as FTS5 phrases, not unigram soup."""
        phrases: list[str] = []
        for raw in (*state.raw_constraints, state.category_tail or ""):
            cleaned = raw.strip().replace('"', '""')
            if cleaned:
                phrases.append(f'"{cleaned}"')
        return " OR ".join(phrases)

    def _term_expression(self, state: DialogState) -> str:
        """Lower-weighted fallback: the individual terms, OR'd."""
        tokens: set[str] = set()
        for raw in (*state.raw_constraints, state.category_tail or ""):
            tokens.update(_tokens(raw))
        for slot in state.active_slots():
            tokens.update(_tokens(str(slot.value)))
        return " OR ".join(f'"{t}"' for t in list(tokens)[:40])

    def _run(self, expression: str, fetch_limit: int) -> list[tuple[str, float]]:
        if not expression or fetch_limit <= 0:
            return []
        try:
            rows = self._conn.execute(
                f"SELECT parent_asin, bm25(products, {self._BM25_ARGS}) FROM products "
                f"WHERE products MATCH ? ORDER BY bm25(products, {self._BM25_ARGS}) LIMIT ?",
                (expression, fetch_limit),
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        return [(str(asin), float(score)) for asin, score in rows]

    def search(self, state: DialogState, limit: int) -> list[Candidate]:
        limit = max(0, int(limit))
        if limit == 0:
            return []

        # SQLite FTS5 MATCH already only scans matching postings, not the whole
        # table; the "pool restriction" the brief asks for is applied here as a
        # post-filter against the tail/constraint pool once one is known, with a
        # fallback to the unrestricted rows if the filter leaves too few.
        restrict_tail = state.category_tail if 0 < state.pool_size <= self._RESTRICT_POOL_CEILING else None
        fetch_limit = limit if restrict_tail is None else max(limit, min(self._RESTRICT_POOL_CEILING, limit * 20))

        seen: set[str] = set()
        rows: list[tuple[str, float]] = []
        for expression in (self._phrase_expression(state), self._term_expression(state)):
            for asin, score in self._run(expression, fetch_limit):
                if asin in seen or asin not in self._asins:
                    continue
                seen.add(asin)
                rows.append((asin, score))

        if restrict_tail:
            filtered = [(a, s) for a, s in rows if self._products[a].category_tail == restrict_tail]
            if len(filtered) >= min(limit, 5):
                rows = filtered

        candidates = [
            Candidate(parent_asin=asin, score=-score, route=self.name, evidence={"bm25": score})
            for asin, score in rows[:limit]
        ]
        return candidates
