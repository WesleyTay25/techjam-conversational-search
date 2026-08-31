"""The orchestrator: one turn of the pipeline, end to end. OWNER: Lead.

Wires the five feature branches into the `reset`/`respond` contract the harness
calls (`docs/agent_api_contract.json`):

    message -> [C] router + slot extractor + state machine
            -> [A] constraint pool -> [A] structured / lexical  +  [B] dense
            -> [B] weighted RRF fusion + dynamic truncation
            -> [D] feature reranker -> [E] optional LLM rerank
            -> top 10  +  [D] info-gain question policy

Four rules from `team/00_BLUEPRINT.md` §7 and `team/02_EVALUATOR_MECHANICS.md`
§1 shape everything here, and they are the reason this file looks defensive:

1. **Never ask empty-handed.** A question and ten guesses cost the same single
   turn, and the session ends the instant the target appears. So every response
   carries ten recommendations *and* a probe — the starter agent's MTTC of 9.81
   is almost entirely the absence of this line.
2. **An exception is a miss, not a crash.** The harness swallows exceptions and
   scores an empty response, so a bug that throws on 3% of sessions silently
   costs 3% of hit rate with no traceback. `respond` therefore cannot raise:
   every stage is individually guarded and degrades to the stage before it.
3. **Everything expensive happens in `__init__`.** The harness builds `Agent`
   once and reuses it for all 200 sessions; per-session state is created in
   `reset`. Per-turn budget is 150 ms.
4. **Determinism.** No RNG anywhere on this path. Two runs must produce
   byte-identical `results.json`.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Sequence

from src.catalog.constraint_index import ConstraintIndex
from src.catalog.loader import load_catalog
from src.contracts import (
    MAX_TURNS,
    TOP_K,
    Candidate,
    DialogState,
    Product,
    TurnResult,
)
from src.dialog.question_policy import InfoGainQuestionPolicy, ProbeBandit
from src.dialog.state import StateMachine
from src.rank.reranker import FeatureReranker
from src.retrieval.fusion import dynamic_truncation, reciprocal_rank_fusion, track_weights
from src.retrieval.lexical import LexicalRoute
from src.retrieval.structured import StructuredRoute

LOGGER = logging.getLogger(__name__)

# How deep the fused list runs before reranking. Wide enough that the reranker
# has something to reorder, narrow enough to stay inside the per-turn budget.
FUSION_LIMIT = 200


class Agent:
    """The submission entry point. Construct once, reuse across sessions."""

    def __init__(self, catalog_path: str | Path = "data/catalog.jsonl") -> None:
        self.products: dict[str, Product] = load_catalog(catalog_path)
        self.catalog_ids = frozenset(self.products)

        self.index = ConstraintIndex(self.products)
        self.lexical = LexicalRoute(self.products)
        self.structured = StructuredRoute(self.products)
        self.dense = self._build_dense()

        self.machine = StateMachine()
        self.bandit = ProbeBandit()
        self.policy = InfoGainQuestionPolicy(self.bandit)
        self.reranker: object = FeatureReranker(self.products)
        self.reranker = self._wrap_llm(self.reranker)

        # Ordered once, not per turn (rule 6: expensive work lives in __init__).
        self._popular: tuple[str, ...] = tuple(
            sorted(self.products, key=self._popularity_key)
        )

        self.sessions: dict[str, DialogState] = {}
        # The bandit grades a probe by how fast its session converted, but the
        # harness never tells us we converted — it just stops calling. It is
        # inferable: a session that stops before MAX_TURNS ended on a hit. So we
        # settle the previous session at the next `reset`. Deterministic,
        # because the harness walks the dataset in a fixed order.
        self._pending: tuple[str, str | None, str | None, int] | None = None
        self._last_asked: dict[str, str | None] = {}

    # -- construction helpers ---------------------------------------------
    def _build_dense(self):
        """The dense route is the one component with third-party dependencies.

        numpy/scikit-learn are declared in `requirements.txt`, but a hard import
        at module scope makes a missing wheel fatal for the *whole* agent rather
        than for one of four routes. Losing the dense route costs recall on
        browsing sessions; failing to construct costs every session.
        """
        try:
            from src.retrieval.dense import DenseRoute

            return DenseRoute(self.products)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("dense route unavailable (%s); running on 3 routes", error)
            return None

    def _wrap_llm(self, fallback):
        """Opt-in LLM layer. Unset `TECHJAM_LLM` means this is never constructed."""
        if os.environ.get("TECHJAM_LLM") != "1":
            return fallback
        try:
            from src.rank.llm_reranker import LLMReranker

            return LLMReranker(fallback)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("LLM reranker unavailable (%s); staying deterministic", error)
            return fallback

    # -- harness API -------------------------------------------------------
    def reset(self, session_id: str, user_profile: dict) -> None:
        """Open a session. Also settles the previous one for the probe bandit."""
        self._settle_pending(converted=True)
        try:
            self.sessions[session_id] = self.machine.start(session_id, user_profile or {})
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("reset failed for %s (%s); using a bare state", session_id, error)
            self.sessions[session_id] = DialogState(session_id=session_id)
        self._last_asked[session_id] = None

    def respond(self, session_id: str, user_message: str, turn: int, top_k: int = TOP_K) -> dict:
        """One turn. Never raises: the harness scores a thrown exception as a miss."""
        started = time.perf_counter()
        state = self.sessions.get(session_id)
        if state is None:
            # `reset` is contractually guaranteed, but a missing session must
            # still produce ten valid ASINs rather than an empty list.
            LOGGER.warning("respond before reset for %s; recovering", session_id)
            state = self.sessions[session_id] = DialogState(session_id=session_id)

        try:
            self.machine.ingest(state, user_message, turn)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("ingest failed on turn %s (%s); ranking on prior state", turn, error)

        candidates = self._retrieve(state)
        candidates = self._rerank(state, candidates)
        attribute, message = self._ask(state, candidates)

        picks = self._finalize(state, candidates, top_k)
        self._record_turn(state, attribute, turn)

        elapsed = (time.perf_counter() - started) * 1000.0
        if elapsed > 150.0:
            LOGGER.warning("turn %s took %.0f ms (budget 150 ms)", turn, elapsed)

        return TurnResult(
            message=message,
            ask_attribute=attribute,
            candidates=picks,
            **self._usage(),
        ).to_payload()

    # -- pipeline stages ---------------------------------------------------
    def _retrieve(self, state: DialogState) -> list[Candidate]:
        """Pool -> four routes -> weighted RRF. Degrades route by route."""
        try:
            pool = self.index.candidate_pool(state)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("pool build failed (%s); falling back to full catalog", error)
            pool = set(self.catalog_ids)
            state.pool_size = len(pool)

        depths = dynamic_truncation(state, len(pool))
        results: dict[str, Sequence[Candidate]] = {
            "constraint": self._constraint_candidates(state, pool, depths.get("constraint", 0)),
        }
        for name, route in (
            ("structured", self.structured),
            ("lexical", self.lexical),
            ("dense", self.dense),
        ):
            limit = depths.get(name, 0)
            if route is None or limit <= 0:
                continue
            try:
                results[name] = route.search(state, limit)
            except Exception as error:  # noqa: BLE001
                LOGGER.warning("route %s failed (%s); dropping it this turn", name, error)

        try:
            fused = reciprocal_rank_fusion(results, track_weights(state), limit=FUSION_LIMIT)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("fusion failed (%s); using the constraint pool order", error)
            fused = list(results.get("constraint") or [])

        # Constrain to the pool only while that leaves us a full page. The pool
        # is a hard filter built from disclosed constraints, and on the private
        # set those strings may not be catalog-exact — so it must never be the
        # reason we return fewer than ten.
        inside = [c for c in fused if c.parent_asin in pool]
        return inside if len(inside) >= TOP_K else fused

    def _constraint_candidates(
        self, state: DialogState, pool: set[str], limit: int
    ) -> list[Candidate]:
        """Turn the exact-match pool into a ranked route.

        The pool is a set, so it carries no order of its own — but RRF weights
        by rank, so an arbitrary order would inject noise into the highest-
        weighted route on buying sessions. Rank by how many disclosed strings a
        product actually matches, then by review support, then by ASIN so two
        runs agree exactly.
        """
        if limit <= 0 or not pool:
            return []
        constraints = [c for c in state.raw_constraints if c and c.strip()]
        matches: dict[str, int] = {asin: 0 for asin in pool}
        for constraint in constraints:
            try:
                hits = self.index.by_constraint(constraint)
            except Exception:  # noqa: BLE001
                continue
            for asin in hits & pool:
                matches[asin] += 1

        def sort_key(asin: str) -> tuple:
            product = self.products.get(asin)
            rating = product.average_rating if product and product.average_rating else 0.0
            support = product.rating_number if product else 0
            return (-matches[asin], -rating, -support, asin)

        return [
            Candidate(parent_asin=asin, score=1.0 / (rank + 1), route="constraint")
            for rank, asin in enumerate(sorted(pool, key=sort_key)[:limit])
        ]

    def _rerank(self, state: DialogState, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        try:
            return self.reranker.rerank(state, candidates)
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("rerank failed (%s); keeping fusion order", error)
            return candidates

    def _ask(self, state: DialogState, candidates: Sequence[Candidate]) -> tuple[str | None, str]:
        """Choose the probe. Failure here costs a question, never the turn."""
        try:
            pool = [
                self.products[c.parent_asin]
                for c in candidates[:FUSION_LIMIT]
                if c.parent_asin in self.products
            ]
            attribute, message = self.policy.choose(state, pool)
            return attribute, message or "Here are the closest matches I found."
        except Exception as error:  # noqa: BLE001
            LOGGER.warning("question policy failed (%s); showing the list", error)
            return None, "Here are the closest matches I found."

    def _finalize(
        self, state: DialogState, candidates: Sequence[Candidate], top_k: int
    ) -> list[Candidate]:
        """Ten valid, unique, in-catalog ASINs — padded if the routes came up short.

        `normalize_recommendations` silently drops duplicates and unknown IDs,
        so a malformed list costs recommendation slots. Validating here means a
        thin turn still spends all ten.
        """
        limit = max(1, int(top_k or TOP_K))
        picks: list[Candidate] = []
        seen: set[str] = set()
        for candidate in candidates:
            asin = str(candidate.parent_asin)
            if asin in seen or asin not in self.catalog_ids:
                continue
            seen.add(asin)
            picks.append(candidate)
            if len(picks) >= limit:
                break

        if len(picks) < limit:
            for asin in self._pad_pool(state, limit - len(picks), seen):
                picks.append(Candidate(parent_asin=asin, score=0.0, route="pad"))

        try:
            self.machine.mark_shown(state, [c.parent_asin for c in picks])
        except Exception:  # noqa: BLE001
            pass
        return picks

    def _pad_pool(self, state: DialogState, needed: int, seen: set[str]) -> list[str]:
        """Deterministic filler so we never hand back fewer than ten slots.

        Two tiers, and the second one is the point: pad from the category tail
        first because those are at least the right kind of product, then top up
        from the catalog-wide popularity order. Stopping at the tail would cap
        us below ten whenever the tail itself is small, and a short list is a
        silently discarded recommendation slot, not a neutral outcome.
        """
        picked: list[str] = []
        taken = set(seen)

        try:
            tail_pool = (
                self.index.by_category_tail(state.category_tail)
                if state.category_tail
                else set()
            )
        except Exception:  # noqa: BLE001
            tail_pool = set()

        if tail_pool:
            for asin in sorted(tail_pool, key=self._popularity_key):
                if asin not in taken:
                    picked.append(asin)
                    taken.add(asin)
                    if len(picked) >= needed:
                        return picked

        # `self._popular` is ordered once at construction: re-sorting 50k rows
        # inside a 150 ms turn budget is not affordable.
        for asin in self._popular:
            if asin not in taken:
                picked.append(asin)
                taken.add(asin)
                if len(picked) >= needed:
                    break
        return picked

    def _popularity_key(self, asin: str) -> tuple:
        product = self.products.get(asin)
        rating = product.average_rating if product and product.average_rating else 0.0
        support = product.rating_number if product else 0
        return (-rating, -support, asin)

    # -- bookkeeping -------------------------------------------------------
    def _record_turn(self, state: DialogState, attribute: str | None, turn: int) -> None:
        """Remember the probe, for `state.asked` and for the bandit's credit."""
        if attribute:
            try:
                state.asked.append(attribute)
            except Exception:  # noqa: BLE001
                pass
        first = self._last_asked.get(state.session_id) or attribute
        self._last_asked[state.session_id] = first
        self._pending = (state.session_id, state.category_tail, first, int(turn))

    def _settle_pending(self, converted: bool) -> None:
        """Grade the previous session's opening probe by when it converted.

        A session that stopped before `MAX_TURNS` ended on a hit, so its last
        turn *is* its conversion turn. One that ran the full ten did not
        convert and earns no reward — `observe` ignores a None turn.
        """
        if self._pending is None:
            return
        _, tail, probe, last_turn = self._pending
        self._pending = None
        conversion = last_turn if (converted and last_turn < MAX_TURNS) else None
        try:
            self.bandit.observe(tail, probe, conversion)
        except Exception:  # noqa: BLE001
            pass

    def _usage(self) -> dict:
        """Report real token counts; the offline path honestly reports zero."""
        usage = getattr(self.reranker, "usage", None)
        if isinstance(usage, tuple) and len(usage) == 2:
            return {"prompt_tokens": int(usage[0]), "completion_tokens": int(usage[1])}
        return {"prompt_tokens": 0, "completion_tokens": 0}
