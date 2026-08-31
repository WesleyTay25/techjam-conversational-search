"""Harness entry point. OWNER: Lead.

`evaluator/local_evaluator.py` imports `Agent` from this module, and the
organizer's private harness does the same, so this file is the seam between our
pipeline and whatever runs it. It holds no logic of its own: `Agent` is the
orchestrator in `src/agent.py`.

Two escape hatches, both deliberate:

* `TECHJAM_BASELINE=1` routes back to `BaselineAgent`, the original BM25
  starter, so the M0 control number (technical_score 0.10671) stays
  reproducible from the same command.
* If the pipeline cannot be constructed at all -- a missing wheel, a corrupt
  index -- we log and fall back to the baseline rather than failing to build.
  A broken agent that raises in `__init__` scores zero on all 200 sessions;
  the baseline scores 0.10671. Never trade the second for the first silently.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from pathlib import Path

LOGGER = logging.getLogger(__name__)


TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from",
    "i", "in", "is", "it", "me", "my", "of", "on", "or", "please", "some",
    "that", "the", "this", "to", "want", "with", "would", "you", "looking",
}


def _text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(f"{key} {item}" for key, item in value.items())
    if isinstance(value, list):
        return " ".join(str(item) for item in value)
    return str(value)


def _terms(text: str) -> list[str]:
    return [
        token.lower()
        for token in TOKEN_RE.findall(text)
        if len(token) > 1 and token.lower() not in STOPWORDS
    ]


class BaselineAgent:
    """The original weak baseline: stateless BM25, no state, no LLM.

    Kept verbatim and reachable so M0's control number stays reproducible --
    every comparison the team makes is against the 0.10671 this scores. Set
    `TECHJAM_BASELINE=1` to route the harness back to it.
    """

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.catalog_path = Path(catalog_path)
        self.connection = sqlite3.connect(":memory:")
        self._sessions: set[str] = set()
        self._build_index()

    def _build_index(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            "CREATE VIRTUAL TABLE products USING fts5("
            "parent_asin UNINDEXED, title, categories, features, details, store, description, "
            "tokenize='unicode61 remove_diacritics 2')"
        )
        batch: list[tuple[str, str, str, str, str, str, str]] = []
        with self.catalog_path.open(encoding="utf-8") as handle:
            for line in handle:
                product = json.loads(line)
                batch.append(
                    (
                        str(product["parent_asin"]),
                        _text(product.get("title")),
                        _text(product.get("categories")),
                        _text(product.get("features")),
                        _text(product.get("details")),
                        _text(product.get("store")),
                        _text(product.get("description")),
                    )
                )
                if len(batch) >= 1000:
                    cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
                    batch.clear()
        if batch:
            cursor.executemany("INSERT INTO products VALUES (?, ?, ?, ?, ?, ?, ?)", batch)
        self.connection.commit()

    def reset(self, session_id: str, user_profile: dict) -> None:
        # The profile is anonymized and may be used for personalization.
        self._sessions.add(session_id)

    def respond(
        self,
        session_id: str,
        user_message: str,
        turn: int,
        top_k: int,
    ) -> dict:
        if session_id not in self._sessions:
            raise RuntimeError("reset must be called before respond")
        unique_terms = list(dict.fromkeys(_terms(user_message)))[:40]
        expression = " OR ".join(f'"{term}"' for term in unique_terms)
        if not expression:
            recommendations: list[dict] = []
        else:
            rows = self.connection.execute(
                "SELECT parent_asin FROM products WHERE products MATCH ? "
                "ORDER BY bm25(products, 0.0, 6.0, 4.0, 2.5, 2.5, 1.5, 1.0) LIMIT ?",
                (expression, top_k),
            ).fetchall()
            recommendations = [{"parent_asin": str(row[0])} for row in rows]
        return {
            "message": "Here are the closest matches I found.",
            "ask_attribute": None,
            "recommendations": recommendations,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }


def _use_baseline() -> bool:
    return os.environ.get("TECHJAM_BASELINE") == "1"


class Agent:
    """Delegates to the real pipeline; falls back to BM25 only if it cannot build."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self._impl = None
        if not _use_baseline():
            try:
                from src.agent import Agent as PipelineAgent

                self._impl = PipelineAgent(catalog_path)
            except Exception as error:  # noqa: BLE001
                LOGGER.error(
                    "pipeline failed to build (%s); falling back to the BM25 baseline", error
                )
        if self._impl is None:
            self._impl = BaselineAgent(catalog_path)

    @property
    def implementation(self) -> str:
        """Which agent actually answered -- printed by the ablation tooling."""
        return type(self._impl).__name__

    def reset(self, session_id: str, user_profile: dict) -> None:
        self._impl.reset(session_id, user_profile)

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int = 10) -> dict:
        return self._impl.respond(session_id, user_message, turn, top_k)
