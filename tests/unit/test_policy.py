"""Branch D acceptance tests: information-gain question policy + offline reranker.

Everything runs on hand-built rows pushed through the real `build_product`
normaliser, so the fixtures exercise the same derived fields (`constraint_keys`,
`category_tail`, `color`, `material`) the 50k catalog produces.
"""

from __future__ import annotations

import random
import unittest

from src.catalog.loader import build_product
from src.contracts import Candidate, DialogState, Slot, TRACK_BROWSING
from src.dialog.question_policy import ASKABLE, InfoGainQuestionPolicy, ProbeBandit
from src.rank.reranker import FeatureReranker, fit_weights

DRESS_CATEGORIES = ["Clothing, Shoes & Jewelry", "Women", "Clothing, Dresses"]


def make_product(asin: str, **overrides):
    row = {
        "parent_asin": asin,
        "title": overrides.get("title", "Womens Dress"),
        "features": overrides.get("features", []),
        "description": overrides.get("description", []),
        "categories": overrides.get("categories", DRESS_CATEGORIES),
        "details": overrides.get("details", {}),
        "store": overrides.get("store", "Acme"),
        "price": overrides.get("price", 40.0),
        "average_rating": overrides.get("average_rating", 4.0),
        "rating_number": overrides.get("rating_number", 100),
    }
    return build_product(row)


def make_state(**overrides) -> DialogState:
    state = DialogState(session_id=overrides.pop("session_id", "test"))
    state.category_tail = overrides.pop("category_tail", "Women Dresses")
    for name, value in overrides.items():
        setattr(state, name, value)
    return state


class QuestionPolicyTest(unittest.TestCase):
    def setUp(self) -> None:
        self.policy = InfoGainQuestionPolicy()

    def test_distinct_colour_beats_shared_brand(self) -> None:
        pool = [
            make_product(f"c{i}", title=f"Womens {colour} Dress")
            for i, colour in enumerate(["Black", "White", "Blue", "Red", "Green", "Pink"])
        ]
        state = make_state(pool_size=400)

        attribute, _ = self.policy.choose(state, pool)

        self.assertEqual(attribute, "color")
        # brand has one value across the whole pool, so its entropy -- and its
        # gain -- is exactly zero. No special-casing needed.
        self.assertEqual(self.policy.expected_gain("brand", state, pool), 0.0)

    def test_dead_attribute_scores_zero_and_is_never_returned(self) -> None:
        pool = [
            make_product(f"c{i}", title=f"Womens {colour} Dress")
            for i, colour in enumerate(["Black", "White", "Blue", "Red", "Green"])
        ]
        state = make_state(pool_size=400, dead_attributes={"color"})

        self.assertEqual(self.policy.expected_gain("color", state, pool), 0.0)
        attribute, _ = self.policy.choose(state, pool)
        self.assertNotEqual(attribute, "color")

    def test_open_probe_wins_while_constraint_set_is_unsaturated(self) -> None:
        pool = [
            make_product("p1", title="Womens Black Cotton Midi Dress", price=20.0, description=["for work"]),
            make_product("p2", title="Womens White Silk Maxi Dress", price=120.0, description=["for a wedding"]),
            make_product("p3", title="Womens Blue Polyester Wrap Dress", price=45.0, description=["everyday"]),
            make_product("p4", title="Womens Red Wool Mini Dress", price=80.0, description=["for winter"]),
            make_product("p5", title="Womens Green Nylon Sleeveless Dress", price=30.0, description=["for the gym"]),
            make_product("p6", title="Womens Pink Rayon Bodycon Dress", price=60.0, description=["cocktail"]),
        ]
        state = make_state(pool_size=800, raw_constraints=[])

        best = max(ASKABLE, key=lambda a: self.policy.expected_gain(a, state, pool))
        self.assertEqual(best, "other")

    def test_specific_slot_beats_open_probe_once_only_colour_is_missing(self) -> None:
        pool = [
            make_product(f"c{i}", title=f"Womens {colour} Dress", price=30.0)
            for i, colour in enumerate(["Black", "White", "Blue", "Red", "Green"])
        ]
        # material / use_case / budget are already disclosed; only colour is left.
        state = make_state(
            pool_size=400,
            raw_constraints=["100% Cotton", "for the gym", "budget around $30"],
        )

        colour_gain = self.policy.expected_gain("color", state, pool)
        open_gain = self.policy.expected_gain("other", state, pool)
        self.assertGreater(colour_gain, open_gain)

    def test_over_general_pool_forces_a_structured_question(self) -> None:
        pool = [
            make_product(f"c{i}", title=f"Womens {colour} Dress")
            for i, colour in enumerate(["Black", "Black", "White", "White", "Blue", "Red"])
        ]
        state = make_state(pool_size=20_000)

        attribute, message = self.policy.choose(state, pool)

        self.assertEqual(attribute, "color")
        self.assertIn("?", message)
        self.assertIn(" or ", message)
        self.assertTrue(any(colour in message for colour in ("black", "white", "blue")))

    def test_tight_pool_stops_asking(self) -> None:
        pool = [make_product(f"c{i}", title=f"Womens {c} Dress") for i, c in enumerate("abcdefgh")]
        state = make_state(pool_size=8)

        attribute, message = self.policy.choose(state, pool)

        self.assertIsNone(attribute)
        self.assertTrue(message)

    def test_choice_is_deterministic_including_under_the_bandit(self) -> None:
        pool = [
            make_product(f"c{i}", title=f"Womens {colour} Dress")
            for i, colour in enumerate(["Black", "White", "Blue", "Red", "Green"])
        ]
        state = make_state(pool_size=400)

        bandit = ProbeBandit()
        bandit.observe("Women Dresses", "color", 3)
        bandit.observe("Women Dresses", "color", 2)
        policy = InfoGainQuestionPolicy(bandit=bandit)

        self.assertEqual(policy.choose(state, pool), policy.choose(state, pool))


class FeatureRerankerTest(unittest.TestCase):
    def _rerank(self, products, state, candidates):
        return FeatureReranker(products).rerank(state, candidates)

    def test_more_exact_constraint_hits_ranks_higher(self) -> None:
        two = make_product("X", title="Womens Blue Dress", features=["100% Cotton", "Waterproof membrane"])
        one = make_product("Y", title="Womens Blue Dress", features=["100% Cotton", "Breathable panel"])
        products = {"X": two, "Y": one}
        state = make_state(raw_constraints=["100% Cotton", "Waterproof membrane"])

        ordered = self._rerank(products, state, [Candidate("Y", 1.0), Candidate("X", 1.0)])

        self.assertEqual([c.parent_asin for c in ordered], ["X", "Y"])

    def test_matching_an_erased_slot_ranks_lower(self) -> None:
        matches = make_product("E", title="Womens Green Wrap Dress", features=["Wrap front"])
        clean = make_product("F", title="Womens Green Shift Dress", features=["Shift silhouette"])
        products = {"E": matches, "F": clean}
        state = make_state(slots={"style": [Slot(attribute="style", value="wrap", turn=1, weight=0.0)]})

        ordered = self._rerank(products, state, [Candidate("E", 1.0), Candidate("F", 1.0)])

        self.assertEqual([c.parent_asin for c in ordered], ["F", "E"])

    def test_output_is_a_permutation_of_the_input(self) -> None:
        products = {a: make_product(a, title=f"Womens {a} Dress") for a in ("a", "b", "c", "d", "e")}
        state = make_state()
        candidates = [Candidate(a, float(i)) for i, a in enumerate(products)]

        ordered = self._rerank(products, state, candidates)

        self.assertEqual(sorted(c.parent_asin for c in ordered), sorted(products))
        self.assertEqual(len(ordered), len(candidates))

    def test_reranking_is_deterministic_across_runs(self) -> None:
        products = {
            a: make_product(a, title=f"Womens {colour} Dress")
            for a, colour in zip("abcdef", ["Black", "Black", "Blue", "Blue", "Red", "Green"])
        }
        state = make_state(track=TRACK_BROWSING)
        candidates = [Candidate(a, 1.0 - 0.1 * i, evidence={"bm25": float(i)}) for i, a in enumerate(products)]
        reranker = FeatureReranker(products)

        first = [c.parent_asin for c in reranker.rerank(state, candidates)]
        second = [c.parent_asin for c in reranker.rerank(state, candidates)]

        self.assertEqual(first, second)

    def test_browsing_track_diversifies_near_duplicates(self) -> None:
        # Five identical black dresses and one blue: MMR should lift the blue one
        # off the bottom so the top of the list is not a single guess repeated.
        products = {
            "b1": make_product("b1", title="Womens Black Dress"),
            "b2": make_product("b2", title="Womens Black Dress"),
            "b3": make_product("b3", title="Womens Black Dress"),
            "b4": make_product("b4", title="Womens Black Dress"),
            "blue": make_product("blue", title="Womens Blue Dress"),
        }
        state = make_state(track=TRACK_BROWSING)
        # Equal relevance, so diversity is the only thing left to rank on.
        candidates = [Candidate(asin, 1.0) for asin in ("b1", "b2", "b3", "b4", "blue")]

        ordered = [c.parent_asin for c in FeatureReranker(products).rerank(state, candidates)]
        buying = make_state(track="buying")
        without_mmr = [c.parent_asin for c in FeatureReranker(products).rerank(buying, candidates)]

        self.assertLessEqual(ordered.index("blue"), 1)
        self.assertGreater(without_mmr.index("blue"), ordered.index("blue"))


class FitWeightsTest(unittest.TestCase):
    def test_logistic_fit_reports_cv_spread_and_caps_weights(self) -> None:
        try:
            import sklearn  # noqa: F401
        except ImportError:
            self.skipTest("scikit-learn not installed")

        rng = random.Random(0)
        rows: list[dict[str, float]] = []
        labels: list[int] = []
        for _ in range(240):
            target = rng.random() < 0.3
            rows.append(
                {
                    "constraint_exact_hits": (2.0 if target else 0.0) + rng.random(),
                    "price_delta": (0.05 if target else 0.6) + rng.random() * 0.1,
                }
            )
            labels.append(int(target))

        weights, cv_mean, cv_sd = fit_weights(rows, labels, cap=10.0, folds=5)

        self.assertEqual(set(weights), {"constraint_exact_hits", "price_delta"})
        self.assertTrue(all(abs(w) <= 10.0 for w in weights.values()))
        self.assertGreater(cv_mean, 0.5)
        self.assertGreaterEqual(cv_sd, 0.0)


if __name__ == "__main__":
    unittest.main()
