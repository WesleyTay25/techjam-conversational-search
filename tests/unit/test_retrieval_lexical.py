"""Acceptance tests for Branch A: constraint index, lexical route, structured route."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.catalog.constraint_index import ConstraintIndex
from src.catalog.loader import load_catalog
from src.contracts import DialogState, Slot
from src.retrieval.lexical import LexicalRoute
from src.retrieval.structured import StructuredRoute

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "mini_catalog.jsonl"

COTTON_ASINS = {"B0MINI0001", "B0MINI0006", "B0MINI0011"}
DRESS_ASINS = {"B0MINI0001", "B0MINI0002", "B0MINI0003", "B0MINI0009", "B0MINI0012"}


class ConstraintIndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.products = load_catalog(FIXTURE_PATH)
        cls.index = ConstraintIndex(cls.products)

    def test_by_constraint_exact_match(self) -> None:
        self.assertEqual(self.index.by_constraint("100% Cotton"), COTTON_ASINS)

    def test_by_constraint_normalizes_whitespace_and_case(self) -> None:
        self.assertEqual(self.index.by_constraint("100 % cotton  "), COTTON_ASINS)

    def test_by_constraint_jaccard_backoff_on_reorder(self) -> None:
        self.assertEqual(self.index.by_constraint("cotton 100%"), COTTON_ASINS)

    def test_by_tail_exact(self) -> None:
        self.assertEqual(self.index.by_category_tail("Women Dresses"), DRESS_ASINS)

    def test_by_tail_backs_off_to_shorter_tail(self) -> None:
        self.assertEqual(self.index.by_category_tail("Dresses"), DRESS_ASINS)

    def test_by_price(self) -> None:
        self.assertEqual(self.index.by_price(49.99), {"B0MINI0001", "B0MINI0009"})

    def test_candidate_pool_intersects_tail_and_constraint(self) -> None:
        state = DialogState(
            session_id="s1",
            category_tail="Women Dresses",
            raw_constraints=["100% Cotton"],
        )
        self.assertEqual(self.index.candidate_pool(state), {"B0MINI0001"})

    def test_candidate_pool_skips_contradictory_constraint(self) -> None:
        state = DialogState(
            session_id="s2",
            category_tail="Women Dresses",
            raw_constraints=["Genuine Leather"],
        )
        self.assertEqual(self.index.candidate_pool(state), DRESS_ASINS)


class LexicalRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.products = load_catalog(FIXTURE_PATH)
        cls.route = LexicalRoute(cls.products)

    def test_no_duplicates_and_no_unknown_asins(self) -> None:
        state = DialogState(
            session_id="s1",
            category_tail="Women Dresses",
            raw_constraints=["100% Cotton", "Machine Wash Cold"],
        )
        results = self.route.search(state, 10)
        asins = [c.parent_asin for c in results]
        self.assertEqual(len(asins), len(set(asins)))
        self.assertTrue(all(asin in self.products for asin in asins))

    def test_empty_state_returns_no_results_without_raising(self) -> None:
        state = DialogState(session_id="s2")
        self.assertEqual(self.route.search(state, 5), [])

    def test_zero_limit_returns_empty(self) -> None:
        state = DialogState(session_id="s3", category_tail="Women Dresses")
        self.assertEqual(self.route.search(state, 0), [])

    def test_full_run_over_all_scenarios_never_raises(self) -> None:
        for tail in ("Women Dresses", "Men Socks", "Shoes Boots", None):
            for constraints in ([], ["Genuine Leather"], ["100% Cotton", "Machine Wash Cold"]):
                state = DialogState(session_id="s", category_tail=tail, raw_constraints=list(constraints))
                results = self.route.search(state, 10)
                self.assertLessEqual(len(results), 10)


class StructuredRouteTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.products = load_catalog(FIXTURE_PATH)
        cls.route = StructuredRoute(cls.products)

    def test_impossible_price_falls_back_to_unfiltered_pool(self) -> None:
        state = DialogState(session_id="s1", price_target=999999.0)
        results = self.route.search(state, 20)
        self.assertEqual(len(results), len(self.products))
        self.assertTrue(all(c.evidence.get("filter_dropped_price") == 1.0 for c in results))

    def test_price_filter_matches_within_tolerance(self) -> None:
        state = DialogState(session_id="s2", price_target=49.99)
        results = self.route.search(state, 20)
        asins = {c.parent_asin for c in results}
        self.assertEqual(asins, {"B0MINI0001", "B0MINI0009"})

    def test_material_and_color_filter_outside_mirrored_vocab(self) -> None:
        state = DialogState(session_id="s3")
        state.slots["material"] = [Slot(attribute="material", value="navy", turn=1, hard=True)]
        results = self.route.search(state, 20)
        self.assertEqual({c.parent_asin for c in results}, {"B0MINI0001"})

    def test_never_returns_empty_when_a_non_empty_pool_exists(self) -> None:
        state = DialogState(session_id="s4")
        state.slots["color"] = [Slot(attribute="color", value="chartreuse", turn=1, hard=True)]
        results = self.route.search(state, 20)
        self.assertGreater(len(results), 0)
        self.assertTrue(any(c.evidence.get("filter_dropped_color") == 1.0 for c in results))


if __name__ == "__main__":
    unittest.main()
