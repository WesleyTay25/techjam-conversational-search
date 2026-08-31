"""Acceptance tests for Branch C — intent routing, slot extraction, dialog state.

No catalog needed: every case here is about what the agent *believes*, not what
it retrieves. The paraphrase cases exist because the private 800 sessions may be
reworded by the organizer (02_EVALUATOR_MECHANICS.md §5) — if layer 1 of the
router were the only path, those tests would be the ones that fail first.
"""

from __future__ import annotations

import unittest

from src.contracts import (
    DialogState,
    TRACK_BOUNDARY,
    TRACK_BROWSING,
    TRACK_BUYING,
    TRACK_OVERRIDE,
    Slot,
)
from src.dialog.state import (
    DECAY_RATE,
    StateMachine,
    approx_tokens,
    askable_attributes,
    distill,
)
from src.nlu.intent_router import HybridIntentRouter
from src.nlu.slot_extractor import (
    SlotExtractor,
    extract_category_tail,
    extract_dead_attribute,
    extract_price,
)

PROFILE = {
    "average_prior_rating": 4.0,
    "preference_tags": ["fit", "comfort", "durability"],
    "purchase_frequency": "3-4 prior purchases",
    "rating_style": "usually positive",
    "summary": "Prior purchases emphasize fit, comfort, durability.",
}

# The simulator's literal openings (02_EVALUATOR_MECHANICS.md §2). Boundary
# sessions reuse the browsing opening verbatim and only reveal themselves on the
# first probe reply, so that scenario is asserted on its reply line instead.
BUYING_OPEN = "I'm looking for Women Dresses. A key requirement is: 100% Cotton."
BROWSING_OPEN = "I'm looking for Women Dresses, but I'm still exploring."
OVERRIDE_OPEN = "I'm looking for Women Dresses. color: black"
OVERRIDE_MSG = "Actually, ignore my earlier preference. What I need is: 100% Cotton."
BOUNDARY_MSG = "I don't have a preference for material; please use your judgment."


def _state(turn: int = 1, **kwargs) -> DialogState:
    state = DialogState(session_id="t", **kwargs)
    state.turn = turn
    return state


class TemplateRoutingTest(unittest.TestCase):
    """Layer 1: the four documented message shapes, at high confidence."""

    def setUp(self) -> None:
        self.router = HybridIntentRouter()

    def test_four_templates_route_at_high_confidence(self) -> None:
        cases = [
            (BUYING_OPEN, TRACK_BUYING, 1),
            (BROWSING_OPEN, TRACK_BROWSING, 1),
            (OVERRIDE_MSG, TRACK_OVERRIDE, 3),
            (BOUNDARY_MSG, TRACK_BOUNDARY, 2),
        ]
        for message, expected, turn in cases:
            with self.subTest(message=message):
                track, confidence = self.router.route(message, _state(turn))
                self.assertEqual(track, expected)
                self.assertGreaterEqual(confidence, 0.9)

    def test_override_opening_is_flagged_before_the_override_lands(self) -> None:
        """`I'm looking for X. {old_value}` announces an override session at turn 1.

        Worth calling out: it tells the pipeline not to commit its pool to the
        stated preference, which is about to be superseded.
        """
        track, confidence = self.router.route(OVERRIDE_OPEN, _state(1))
        self.assertEqual(track, TRACK_OVERRIDE)
        self.assertGreaterEqual(confidence, 0.9)

    def test_override_outranks_a_requirement_clause_in_the_same_message(self) -> None:
        # The override message legally contains "What I need is:", which reads
        # as buying. Missing the override costs the whole session, so it wins.
        track, _ = self.router.route(OVERRIDE_MSG, _state(3))
        self.assertEqual(track, TRACK_OVERRIDE)

    def test_recorded_override_keeps_full_confidence_on_quiet_turns(self) -> None:
        machine = StateMachine()
        state = machine.start("s", PROFILE)
        machine.ingest(state, OVERRIDE_OPEN, 1)
        machine.ingest(state, OVERRIDE_MSG, 3)
        machine.ingest(state, "I don't have an additional preference for brand.", 4)
        self.assertEqual(state.track, TRACK_OVERRIDE)
        self.assertGreaterEqual(state.track_confidence, 0.95)

    def test_refusal_reply_holds_the_established_track(self) -> None:
        state = _state(4)
        state.track = TRACK_BUYING
        state.track_confidence = 0.95
        track, _ = self.router.route(
            "I don't have an additional preference for material.", state
        )
        self.assertEqual(track, TRACK_BUYING)


class ParaphraseRoutingTest(unittest.TestCase):
    """Layer 2: hand-written rewordings, none of which contain a template."""

    PARAPHRASES = {
        TRACK_BUYING: [
            "Hi — I want a women's dress, and it absolutely must be 100% cotton.",
            "Shopping for a dress. Non-negotiable: merino wool, size medium.",
            "Need a dress; the essential thing is that it's machine washable navy fabric.",
        ],
        TRACK_BROWSING: [
            "Just browsing for dresses at the moment, not sure what I want yet.",
            "I'm in the early stages of looking at women's dresses — any ideas?",
            "Show me some dresses, I'm still exploring options and open to suggestions.",
        ],
        TRACK_OVERRIDE: [
            "Hold on, I've changed my mind. What I need is a linen dress.",
            "Please disregard what I said earlier — I want something in chiffon instead.",
            "Scratch that, forget my earlier preference; make it a wool coat.",
        ],
        TRACK_BOUNDARY: [
            "I have no preference for color, use your judgment.",
            "Honestly that doesn't matter to me — your call on the size.",
            "No strong preference either way; whatever you recommend is fine.",
        ],
    }

    def setUp(self) -> None:
        self.router = HybridIntentRouter()

    def test_paraphrases_route_correctly_below_template_confidence(self) -> None:
        for expected, messages in self.PARAPHRASES.items():
            for message in messages:
                with self.subTest(track=expected, message=message):
                    track, confidence = self.router.route(message, _state(3))
                    self.assertEqual(track, expected)
                    self.assertGreaterEqual(confidence, 0.5)
                    # Layer 2 must never claim template certainty: the fusion
                    # layer keys off this to decide whether to hedge its pools.
                    self.assertLess(confidence, 0.7)

    def test_unmarked_reply_does_not_flip_the_track(self) -> None:
        state = _state(5)
        state.track = TRACK_BUYING
        state.track_confidence = 0.95
        track, confidence = self.router.route("Those look alright I suppose.", state)
        self.assertEqual(track, TRACK_BUYING)
        self.assertLess(confidence, 0.95)


class SlotExtractionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.extractor = SlotExtractor()

    def test_semicolon_payload_is_preserved_byte_for_byte(self) -> None:
        slots = self.extractor.extract(
            "For that, what matters is: 100% Cotton; Machine Wash Cold.", _state(2)
        )
        self.assertEqual([s.value for s in slots], ["100% Cotton", "Machine Wash Cold"])
        # Byte-identical is the contract: Branch A's index is keyed on the exact
        # catalog string, so any normalization here silently breaks that route.
        self.assertIn("100% Cotton", [s.value for s in slots])
        self.assertTrue(all(s.value == s.value.strip() for s in slots))
        self.assertFalse(any(s.hard for s in slots))

    def test_opening_requirement_is_hard(self) -> None:
        slots = self.extractor.extract(BUYING_OPEN, _state(1))
        self.assertEqual([s.value for s in slots], ["100% Cotton"])
        self.assertTrue(slots[0].hard)

    def test_post_override_intent_is_hard(self) -> None:
        slots = self.extractor.extract(OVERRIDE_MSG, _state(3))
        self.assertEqual([s.value for s in slots], ["100% Cotton"])
        self.assertTrue(slots[0].hard)

    def test_price_parses_to_a_float(self) -> None:
        self.assertEqual(extract_price("budget around $49.99"), 49.99)
        machine = StateMachine()
        state = machine.start("s", PROFILE)
        machine.ingest(state, "For that, what matters is: budget around $49.99.", 2)
        self.assertEqual(state.price_target, 49.99)
        # The verbatim string still reaches the constraint index.
        self.assertIn("budget around $49.99", state.raw_constraints)

    def test_category_tail_drops_the_trailing_clause(self) -> None:
        self.assertEqual(extract_category_tail(BROWSING_OPEN), "Women Dresses")
        machine = StateMachine()
        state = machine.start("s", PROFILE)
        machine.ingest(state, BROWSING_OPEN, 1)
        self.assertEqual(state.category_tail, "Women Dresses")

    def test_catalog_vocabulary_survives_the_mirrored_regexes(self) -> None:
        """"navy", "merino", "chiffon" are absent from contracts.COLOR_RE/MATERIAL_RE.

        They must still reach the index verbatim — those regexes mirror the
        simulator's detector, not the catalog's vocabulary.
        """
        machine = StateMachine()
        state = machine.start("s", PROFILE)
        machine.ingest(
            state, "For that, what matters is: navy chiffon overlay; merino lining.", 2
        )
        self.assertIn("navy chiffon overlay", state.raw_constraints)
        self.assertIn("merino lining", state.raw_constraints)


class StateMachineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.machine = StateMachine()

    def test_accumulation_keeps_both_values_of_one_attribute(self) -> None:
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, BROWSING_OPEN, 1)
        self.machine.ingest(state, "For that, what matters is: 100% Cotton.", 2)
        self.machine.ingest(state, "For that, what matters is: Machine Wash Cold.", 3)
        values = [s.value for s in state.active_slots()]
        self.assertIn("100% Cotton", values)
        self.assertIn("Machine Wash Cold", values)

    def test_override_erases_without_deleting(self) -> None:
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, OVERRIDE_OPEN, 1)
        self.machine.ingest(state, "For that, what matters is: Polyester lining.", 2)
        self.machine.mark_shown(state, ["B0MINI0001", "B0MINI0002"])
        self.machine.ingest(state, OVERRIDE_MSG, 3)

        superseded = [
            s for group in state.slots.values() for s in group
            if s.value == "Polyester lining"
        ]
        self.assertEqual(len(superseded), 1, "the disclosure must not be deleted")
        self.assertEqual(superseded[0].weight, 0.0)
        self.assertTrue(state.override_applied)
        self.assertEqual(state.shown, [])
        # The superseded string must leave the live retrieval key set, or four
        # routes keep hashing onto the wrong products.
        self.assertNotIn("Polyester lining", state.raw_constraints)
        self.assertIn("100% Cotton", state.raw_constraints)

    def test_override_preserves_the_category_pool_key(self) -> None:
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, OVERRIDE_OPEN, 1)
        self.machine.ingest(state, OVERRIDE_MSG, 3)
        self.assertEqual(state.category_tail, "Women Dresses")
        self.assertIn("Women Dresses", [s.value for s in state.active_slots()])

    def test_soft_slots_decay_and_hard_slots_do_not(self) -> None:
        state = self.machine.start("s", PROFILE)
        state.turn = 1
        self.machine._merge(state, Slot("feature", "soft value", 1, 1.0, False, 1.0))
        self.machine._merge(state, Slot("material", "hard value", 1, 1.0, True, 1.0))
        for turn in (2, 3, 4):
            state.turn = turn
            self.machine.decay(state)
        soft = next(s for s in state.active_slots() if s.value == "soft value")
        hard = next(s for s in state.active_slots() if s.value == "hard value")
        self.assertAlmostEqual(soft.weight, DECAY_RATE ** 3, delta=1e-6)
        self.assertEqual(hard.weight, 1.0)

    def test_decay_never_resurrects_an_erased_slot(self) -> None:
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, OVERRIDE_OPEN, 1)
        self.machine.ingest(state, "For that, what matters is: Polyester lining.", 2)
        self.machine.ingest(state, OVERRIDE_MSG, 3)
        for turn in (4, 5, 6):
            state.turn = turn
            self.machine.decay(state)
        erased = [s for s in self.machine.erased_slots(state) if s.value == "Polyester lining"]
        self.assertEqual(erased[0].weight, 0.0)

    def test_dead_attribute_is_never_askable_and_persists(self) -> None:
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, BROWSING_OPEN, 1)
        self.machine.ingest(state, "I don't have an additional preference for brand.", 2)
        self.assertIn("brand", state.dead_attributes)
        self.assertNotIn("brand", askable_attributes(state))
        for turn in (3, 4, 5):
            self.machine.ingest(state, "For that, what matters is: 100% Cotton.", turn)
            self.assertIn("brand", state.dead_attributes)
            self.assertNotIn("brand", askable_attributes(state))

    def test_boundary_reply_sets_boundary_seen_and_kills_the_attribute(self) -> None:
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, BROWSING_OPEN, 1)
        self.machine.ingest(state, BOUNDARY_MSG, 2)
        self.assertTrue(state.boundary_seen)
        self.assertIn("material", state.dead_attributes)

    def test_paraphrased_refusal_still_kills_the_attribute(self) -> None:
        """An unrecognised refusal is expensive: the policy re-probes it forever.

        Found by running the 200 public sessions under paraphrase stress — one
        boundary session spent nine turns re-asking "color" because "Nothing
        else comes to mind for color." did not match the literal template.
        """
        self.assertEqual(
            extract_dead_attribute("Nothing else comes to mind for color."), "color"
        )
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, BROWSING_OPEN, 1)
        state.asked.append("color")
        self.machine.ingest(state, "Nothing else comes to mind for color.", 2)
        self.assertIn("color", state.dead_attributes)
        self.assertNotIn("color", askable_attributes(state))

    def test_refusal_backstop_kills_the_attribute_we_just_probed(self) -> None:
        """When the refusal names no attribute, the last probe is the target."""
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, BROWSING_OPEN, 1)
        state.asked.append("style")
        self.machine.ingest(state, "Honestly, I couldn't say — up to you.", 2)
        self.assertIn("style", state.dead_attributes)

    def test_a_normal_disclosure_is_not_read_as_a_refusal(self) -> None:
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, BROWSING_OPEN, 1)
        state.asked.append("color")
        self.machine.ingest(state, "For that, what matters is: color: black.", 2)
        self.assertEqual(state.dead_attributes, set())
        self.assertIn("color: black", state.raw_constraints)

    def test_open_probe_stays_askable_after_being_spent(self) -> None:
        """`other` short-circuits the class filter, so it keeps paying on repeat."""
        state = self.machine.start("s", PROFILE)
        self.machine.ingest(state, BROWSING_OPEN, 1)
        state.asked.append("other")
        self.assertIn("other", askable_attributes(state))

    def test_profile_folds_in_at_low_weight_and_does_not_decay(self) -> None:
        state = self.machine.start("s", PROFILE)
        tags = [s for s in state.active_slots() if s.value in PROFILE["preference_tags"]]
        self.assertEqual(len(tags), len(PROFILE["preference_tags"]))
        self.assertTrue(all(s.weight <= 0.2 for s in tags))
        # Aggregate tags are not catalog strings; they must not pollute the
        # exact-match index.
        self.assertNotIn("comfort", state.raw_constraints)
        for turn in (2, 3, 4):
            state.turn = turn
            self.machine.decay(state)
        self.assertTrue(all(s.weight <= 0.2 for s in tags))

    def test_ranking_prior_is_derived_from_the_profile(self) -> None:
        prior = self.machine.start("s", PROFILE).user_profile["ranking_prior"]
        self.assertEqual(prior["average_prior_rating"], 4.0)
        self.assertLessEqual(prior["weight"], 0.2)
        critical = self.machine.start(
            "s2", {**PROFILE, "rating_style": "critical"}
        ).user_profile["ranking_prior"]
        self.assertGreater(critical["min_rating_hint"], prior["min_rating_hint"])

    def test_start_does_not_mutate_the_harness_profile(self) -> None:
        original = dict(PROFILE)
        self.machine.start("s", PROFILE)
        self.assertEqual(PROFILE, original)


class DistillationTest(unittest.TestCase):
    def test_summary_stays_under_budget_on_a_full_ten_turn_session(self) -> None:
        machine = StateMachine()
        state = machine.start("s", PROFILE)
        machine.ingest(state, BUYING_OPEN, 1)
        disclosures = [
            "For that, what matters is: Machine Wash Cold; Imported.",
            "For that, what matters is: color: black; budget around $49.99.",
            "For that, what matters is: Zipper closure; Flutter sleeve detail.",
            "I don't have an additional preference for brand.",
            "For that, what matters is: navy chiffon overlay; merino lining.",
            "For that, what matters is: Department: Womens; Midi length.",
            "I don't have a preference for size; please use your judgment.",
            "For that, what matters is: Machine wash warm; Do not bleach.",
            "For that, what matters is: Regular fit; Crew neck.",
        ]
        for offset, message in enumerate(disclosures, start=2):
            state.pool_size = 400 - offset * 30
            machine.ingest(state, message, offset)
        summary = distill(state)
        self.assertLess(approx_tokens(summary), 200, summary)
        # Still useful, not just short: the pool key and the refusals survive.
        self.assertIn("Women Dresses", summary)
        self.assertIn("brand", summary)

    def test_summary_is_regenerated_not_appended(self) -> None:
        machine = StateMachine()
        state = machine.start("s", PROFILE)
        machine.ingest(state, BUYING_OPEN, 1)
        first = len(distill(state))
        for turn in range(2, 11):
            machine.ingest(state, "Those options are not quite right yet.", turn)
        self.assertLessEqual(len(distill(state)), first * 2)

    def test_summary_is_cached_on_state_for_downstream_branches(self) -> None:
        machine = StateMachine()
        state = machine.start("s", PROFILE)
        machine.ingest(state, BUYING_OPEN, 1)
        self.assertEqual(state.user_profile["context_summary"], distill(state))


if __name__ == "__main__":
    unittest.main()
