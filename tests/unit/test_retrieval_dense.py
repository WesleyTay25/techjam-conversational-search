from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from src.catalog.loader import load_catalog
from src.contracts import (
    TRACK_BOUNDARY,
    TRACK_BROWSING,
    TRACK_BUYING,
    TRACK_OVERRIDE,
    Candidate,
    DialogState,
    Slot,
)
from src.retrieval.dense import DenseRoute
from src.retrieval.fusion import dynamic_truncation, reciprocal_rank_fusion, track_weights

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "mini_catalog.jsonl"


def _make_route() -> DenseRoute:
    return DenseRoute(load_catalog(FIXTURE))


class DenseRouteTest(unittest.TestCase):
    def test_matrix_shape_and_normalization(self) -> None:
        route = _make_route()
        self.assertEqual(route.matrix.dtype, np.float32)
        self.assertEqual(route.matrix.shape, (12, route.dim))
        self.assertLessEqual(route.dim, 256)
        norms = np.linalg.norm(route.matrix, axis=1)
        self.assertTrue(np.allclose(norms, 1.0, atol=1e-5))

    def test_construction_is_deterministic(self) -> None:
        a = _make_route()
        b = _make_route()
        self.assertTrue(np.array_equal(a.matrix, b.matrix))
        vec_a = a.encode_query("dress for a wedding")
        vec_b = b.encode_query("dress for a wedding")
        self.assertTrue(np.array_equal(vec_a, vec_b))

    def test_wedding_query_favours_formal_dresses(self) -> None:
        route = _make_route()
        scores = route.matrix @ route.encode_query("dress for a wedding")
        by_asin = dict(zip(route.asins, scores))
        self.assertGreater(by_asin["B0MINI0001"], by_asin["B0MINI0002"])
        self.assertGreater(by_asin["B0MINI0003"], by_asin["B0MINI0002"])

    def test_gym_query_inverts_the_ordering(self) -> None:
        route = _make_route()
        scores = route.matrix @ route.encode_query("dress for the gym")
        by_asin = dict(zip(route.asins, scores))
        self.assertGreater(by_asin["B0MINI0002"], by_asin["B0MINI0001"])
        self.assertGreater(by_asin["B0MINI0002"], by_asin["B0MINI0003"])

    def test_search_uses_active_slots_and_ignores_erased_ones(self) -> None:
        route = _make_route()
        state = DialogState(session_id="s1", category_tail="Women Dresses")
        state.slots["use_case"] = [Slot(attribute="use_case", value="wedding", turn=1, hard=True, weight=1.0)]
        # An erased slot (weight 0.0) must not leak into the query text.
        state.slots["dead"] = [Slot(attribute="feature", value="gym", turn=1, hard=False, weight=0.0)]

        results = route.search(state, limit=3)
        self.assertIn(results[0].parent_asin, {"B0MINI0001", "B0MINI0003"})
        self.assertEqual(results[0].route, "dense")


class ReciprocalRankFusionTest(unittest.TestCase):
    def test_single_route_reproduces_its_own_ordering(self) -> None:
        route_results = {
            "lexical": [
                Candidate(parent_asin="A", score=9.0),
                Candidate(parent_asin="B", score=5.0),
                Candidate(parent_asin="C", score=1.0),
            ]
        }
        fused = reciprocal_rank_fusion(route_results, weights={"lexical": 1.0})
        self.assertEqual([c.parent_asin for c in fused], ["A", "B", "C"])

    def test_double_agreement_beats_single_higher_weighted_route(self) -> None:
        route_results = {
            "dense": [Candidate(parent_asin="A", score=0.9)],
            "lexical": [Candidate(parent_asin="A", score=8.0)],
            "constraint": [Candidate(parent_asin="B", score=1.0)],
        }
        weights = {"constraint": 0.45, "lexical": 0.30, "dense": 0.25}
        fused = reciprocal_rank_fusion(route_results, weights=weights)
        fused_by_asin = {c.parent_asin: c for c in fused}
        # B is ranked first by the single highest-weighted route (constraint);
        # A is ranked first by two lower-weighted routes. Rank fusion favours A.
        self.assertGreater(fused_by_asin["A"].score, fused_by_asin["B"].score)


class TrackWeightsTest(unittest.TestCase):
    def test_weights_sum_to_one_for_every_track(self) -> None:
        for track in (TRACK_BUYING, TRACK_BROWSING, TRACK_BOUNDARY, TRACK_OVERRIDE):
            for override_applied in (False, True):
                state = DialogState(session_id="s", track=track, override_applied=override_applied)
                weights = track_weights(state)
                self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)

    def test_constraint_weight_is_monotonic_in_hard_slot_count(self) -> None:
        state = DialogState(session_id="s", track=TRACK_BROWSING)
        previous = track_weights(state)["constraint"]
        for count in range(1, 6):
            state.slots["hard"] = [
                Slot(attribute="feature", value=f"c{i}", turn=1, hard=True, weight=1.0) for i in range(count)
            ]
            current = track_weights(state)["constraint"]
            self.assertGreaterEqual(current, previous)
            previous = current


class DynamicTruncationTest(unittest.TestCase):
    def test_dense_is_skipped_once_pool_is_small(self) -> None:
        state = DialogState(session_id="s")
        depths = dynamic_truncation(state, pool_size=150)
        self.assertEqual(depths["dense"], 0)
        self.assertEqual(state.pool_size, 150)

    def test_shallow_and_mid_pool_depths(self) -> None:
        state = DialogState(session_id="s")
        self.assertEqual(dynamic_truncation(state, pool_size=10_000)["lexical"], 50)
        self.assertEqual(dynamic_truncation(state, pool_size=1_000)["lexical"], 200)


if __name__ == "__main__":
    unittest.main()
