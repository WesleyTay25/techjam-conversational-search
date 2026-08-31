"""End-to-end orchestrator tests. OWNER: Lead.

These are the tests that would have caught the two integration defects found at
merge time: a response that returned five recommendations instead of ten, and a
component wired to a signature it did not implement. Unit tests per branch
cannot see either — both only exist once the pieces are joined.

Runs against `tests/fixtures/mini_catalog.jsonl`, so CI never needs the 50k
catalog (blueprint §7 rule 7).
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from src.agent import Agent
from src.contracts import ALLOWED_ATTRIBUTES, TOP_K

FIXTURE = Path(__file__).parent / "fixtures" / "mini_catalog.jsonl"
PROFILE = {
    "purchase_frequency": "3-4 prior purchases",
    "average_prior_rating": 4.0,
    "rating_style": "usually positive",
    "preference_tags": ["fit", "comfort"],
    "summary": "Prior purchases emphasize fit and comfort.",
}

BUYING = "I'm looking for Women Dresses. A key requirement is: 100% Cotton."
BROWSING = "I'm looking for Women Dresses, but I'm still exploring."
OVERRIDE = "Actually, ignore my earlier preference. What I need is: 100% Cotton."
BOUNDARY = "I don't have a preference for material; please use your judgment."
DISCLOSURE = "For that, what matters is: Machine Wash Cold; Imported."


class AgentContractTest(unittest.TestCase):
    """The payload shape the harness validates, and the slots it silently drops."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.agent = Agent(FIXTURE)
        cls.catalog_size = len(cls.agent.products)

    def _turn(self, session: str, message: str, turn: int) -> dict:
        return self.agent.respond(session, message, turn, TOP_K)

    def test_every_turn_ships_a_full_page_of_valid_unique_asins(self) -> None:
        """A short list is a discarded slot, not a neutral outcome.

        `normalize_recommendations` drops duplicates and unknown IDs silently,
        so anything less than ten valid unique in-catalog ASINs is lost scoring
        surface we never get a warning about.
        """
        self.agent.reset("s-full", PROFILE)
        expected = min(TOP_K, self.catalog_size)
        for turn, message in enumerate([BUYING, DISCLOSURE, OVERRIDE, BOUNDARY], start=1):
            with self.subTest(turn=turn):
                payload = self._turn("s-full", message, turn)
                asins = [r["parent_asin"] for r in payload["recommendations"]]
                self.assertEqual(len(asins), expected)
                self.assertEqual(len(set(asins)), expected, "duplicates cost slots")
                for asin in asins:
                    self.assertIn(asin, self.agent.catalog_ids)

    def test_payload_matches_the_api_contract(self) -> None:
        self.agent.reset("s-shape", PROFILE)
        payload = self._turn("s-shape", BROWSING, 1)
        self.assertEqual(
            set(payload), {"message", "ask_attribute", "recommendations", "usage"}
        )
        self.assertIsInstance(payload["message"], str)
        self.assertTrue(payload["message"].strip(), "message is graded for tone")
        if payload["ask_attribute"] is not None:
            self.assertIn(payload["ask_attribute"], ALLOWED_ATTRIBUTES)
        self.assertGreaterEqual(payload["usage"]["prompt_tokens"], 0)
        self.assertGreaterEqual(payload["usage"]["completion_tokens"], 0)
        # The offline path must report honestly rather than inventing usage.
        self.assertEqual(payload["usage"]["prompt_tokens"], 0)

    def test_respond_never_raises(self) -> None:
        """The harness scores a thrown exception as a miss, with no traceback."""
        self.agent.reset("s-hostile", PROFILE)
        for turn, message in enumerate(["", "   ", "!!!", "\x00\x01", "a" * 5000], start=1):
            with self.subTest(message=message[:12]):
                payload = self.agent.respond("s-hostile", message, turn, TOP_K)
                self.assertIsInstance(payload["message"], str)
                self.assertTrue(payload["recommendations"])

    def test_respond_without_reset_still_answers(self) -> None:
        payload = self.agent.respond("never-reset", BUYING, 1, TOP_K)
        self.assertTrue(payload["recommendations"])

    def test_top_k_is_honoured(self) -> None:
        self.agent.reset("s-topk", PROFILE)
        payload = self.agent.respond("s-topk", BROWSING, 1, 3)
        self.assertEqual(len(payload["recommendations"]), 3)

    def test_sessions_do_not_leak_into_each_other(self) -> None:
        """`Agent` is built once and reused for all 200 sessions."""
        self.agent.reset("s-a", PROFILE)
        self.agent.respond("s-a", BUYING, 1, TOP_K)
        self.agent.reset("s-b", PROFILE)
        self.agent.respond("s-b", BROWSING, 1, TOP_K)
        self.assertNotIn("100% Cotton", self.agent.sessions["s-b"].raw_constraints)
        self.assertEqual(self.agent.sessions["s-b"].track, "browsing")

    def test_override_erasure_survives_the_full_pipeline(self) -> None:
        """15% of sessions score zero if the superseded pool is not dropped."""
        self.agent.reset("s-ovr", PROFILE)
        self.agent.respond("s-ovr", "I'm looking for Women Dresses. Polyester lining", 1, TOP_K)
        self.agent.respond("s-ovr", DISCLOSURE, 2, TOP_K)
        self.agent.respond("s-ovr", OVERRIDE, 3, TOP_K)
        state = self.agent.sessions["s-ovr"]
        self.assertTrue(state.override_applied)
        self.assertIn("100% Cotton", state.raw_constraints)
        self.assertNotIn("Machine Wash Cold", state.raw_constraints)

    def test_run_is_deterministic(self) -> None:
        """Two runs must produce byte-identical results (blueprint rule 8)."""
        def transcript() -> str:
            agent = Agent(FIXTURE)
            out = []
            for session in ("d1", "d2"):
                agent.reset(session, PROFILE)
                for turn, message in enumerate([BUYING, DISCLOSURE, OVERRIDE], start=1):
                    out.append(agent.respond(session, message, turn, TOP_K))
            return json.dumps(out, sort_keys=True)

        self.assertEqual(transcript(), transcript())


class DegradationTest(unittest.TestCase):
    """Any single component may fail; the turn must still produce a full page."""

    def setUp(self) -> None:
        self.agent = Agent(FIXTURE)
        self.agent.reset("s-degraded", PROFILE)

    def _assert_still_answers(self) -> None:
        payload = self.agent.respond("s-degraded", BUYING, 1, TOP_K)
        self.assertEqual(
            len(payload["recommendations"]), min(TOP_K, len(self.agent.products))
        )

    def test_survives_a_dead_retrieval_route(self) -> None:
        class Broken:
            def search(self, state, limit):
                raise RuntimeError("route down")

        self.agent.lexical = Broken()
        self.agent.structured = Broken()
        self._assert_still_answers()

    def test_survives_a_missing_dense_route(self) -> None:
        self.agent.dense = None
        self._assert_still_answers()

    def test_survives_a_broken_reranker_and_policy(self) -> None:
        class Broken:
            def rerank(self, state, candidates):
                raise RuntimeError("rerank down")

            def choose(self, state, pool):
                raise RuntimeError("policy down")

        self.agent.reranker = Broken()
        self.agent.policy = Broken()
        self._assert_still_answers()


class BanditFeedbackTest(unittest.TestCase):
    """Pillar III: the probe ordering re-tunes itself from observed conversions."""

    def test_previous_session_is_graded_at_the_next_reset(self) -> None:
        agent = Agent(FIXTURE)
        agent.reset("b1", PROFILE)
        agent.respond("b1", BUYING, 1, TOP_K)
        agent.sessions["b1"].category_tail = "Women Dresses"
        agent._pending = ("b1", "Women Dresses", "material", 2)
        agent.reset("b2", PROFILE)
        # A session that stopped at turn 2 converted at turn 2, so its opening
        # probe earned reward; the bias must move off the neutral 1.0.
        self.assertNotEqual(agent.bandit.bias("Women Dresses", "material"), 1.0)

    def test_a_session_that_ran_the_full_ten_turns_earns_no_reward(self) -> None:
        agent = Agent(FIXTURE)
        agent._pending = ("b3", "Women Dresses", "color", 10)
        agent.reset("b4", PROFILE)
        self.assertEqual(agent.bandit.bias("Women Dresses", "color"), 1.0)


if __name__ == "__main__":
    unittest.main()
