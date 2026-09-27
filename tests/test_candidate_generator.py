"""
tests/test_candidate_generator.py
=================================
Automated unit tests for Strategy Engine Candidate Strategy Generator.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.models import ParticipantState, RaceState
from strategy_engine.candidate_generator import CandidateStrategyGenerator
from strategy_engine.types import Strategy, StrategyEngineConfig


class TestCandidateStrategyGenerator(unittest.TestCase):
    def setUp(self):
        self.config = StrategyEngineConfig(
            minimum_stint_length=4,
            max_tyre_age_soft=24,
            max_tyre_age_medium=34,
            max_tyre_age_hard=46,
            enforce_f1_two_compound_rule=True,
        )
        self.generator = CandidateStrategyGenerator(config=self.config)

    def _create_standard_state(
        self,
        current_lap: int = 15,
        total_laps: int = 57,
        tyre_life: float = 15.0,
        compound: str = "MEDIUM",
        pit_count: int = 0,
    ) -> RaceState:
        state = RaceState(
            current_lap=current_lap,
            total_laps_expected=total_laps,
            grand_prix="British Grand Prix",
            location="Silverstone",
        )
        state.participants["VER"] = ParticipantState(
            driver="VER",
            team="Red Bull Racing",
            position=1,
            compound=compound,
            tyre_life=tyre_life,
            stint=1,
            pit_count=pit_count,
            is_active=True,
        )
        return state

    def test_1stop_candidate_generation(self):
        """Test generation of 1-stop strategy candidates across nominal and undercut windows."""
        state = self._create_standard_state(current_lap=18, tyre_life=18.0, compound="MEDIUM")
        candidates = self.generator.generate_candidates(state, "VER", horizon_laps=10)

        self.assertGreater(len(candidates), 0)
        one_stops = [c for c in candidates if c.num_stops == 1]
        self.assertGreater(len(one_stops), 0)

        # Verify immediate pit and future pit options exist
        pit_laps = [c.pit_laps[0] for c in one_stops]
        self.assertIn(18, pit_laps, "Undercut/pit now (lap 18) should be generated.")
        self.assertTrue(any(l > 18 for l in pit_laps), "Future nominal pit laps should be generated.")

        # Verify compound transitions switch from Medium
        for c in one_stops:
            self.assertIn(c.compounds[0], ["HARD", "SOFT"])
            self.assertNotEqual(c.compounds[0], "MEDIUM")

    def test_2stop_candidate_generation(self):
        """Test generation of 2-stop strategies with proper stint spacing."""
        state = self._create_standard_state(current_lap=10, tyre_life=10.0, compound="SOFT")
        candidates = self.generator.generate_candidates(state, "VER", horizon_laps=10, max_stops=2)

        two_stops = [c for c in candidates if c.num_stops == 2]
        self.assertGreater(len(two_stops), 0)

        for c in two_stops:
            self.assertEqual(len(c.pit_laps), 2)
            p1, p2 = c.pit_laps
            self.assertGreaterEqual(p2 - p1, self.config.minimum_stint_length)
            self.assertLess(p2, 57)

    def test_2stop_same_compound_repeats(self):
        """Test that 2-stop generation allows same-compound repeats while satisfying F1 2-compound rule."""
        state = self._create_standard_state(current_lap=15, tyre_life=15.0, compound="MEDIUM")
        candidates = self.generator.generate_candidates(state, "VER", max_stops=2)

        two_stops = [c for c in candidates if c.num_stops == 2]
        self.assertGreater(len(two_stops), 0)

        # 1. Verify same-compound repeat where comp1 == curr_compound (e.g. M -> M -> H)
        m_repeat_first = [c for c in two_stops if c.compounds[0] == "MEDIUM"]
        self.assertGreater(len(m_repeat_first), 0, "Should generate 2-stop strategies repeating starting compound in stint 2 (M->M->X)")

        # 2. Verify same-compound repeat where comp1 == comp2 (e.g. M -> H -> H or M -> S -> S)
        same_subsequent = [c for c in two_stops if c.compounds[0] == c.compounds[1]]
        self.assertGreater(len(same_subsequent), 0, "Should generate 2-stop strategies repeating compound in subsequent stints (e.g. M->H->H)")

        # 3. Verify that all 2-stops strictly adhere to the 2-compound rule (no M->M->M)
        for c in two_stops:
            all_compounds = {"MEDIUM"} | set(c.compounds)
            self.assertGreaterEqual(
                len(all_compounds), 2,
                f"Strategy {c.strategy_id} ({c.name}) violated F1 2-compound rule with only 1 compound: {all_compounds}"
            )


    def test_f1_two_compound_rule_enforcement(self):
        """Validate rejection of strategies that only use 1 distinct dry compound."""
        state = self._create_standard_state(current_lap=20, tyre_life=20.0, compound="MEDIUM")

        # Illegal: Medium -> Medium with no prior stops
        illegal_strat = Strategy(
            strategy_id="ILLEGAL",
            name="Illegal 1-Stop",
            num_stops=1,
            pit_laps=[25],
            compounds=["MEDIUM"],
        )
        is_valid, reason = self.generator.validate_strategy(illegal_strat, state, "VER")
        self.assertFalse(is_valid)
        self.assertIn("F1 Regulation violation", reason)

        # Legal: Medium -> Hard
        legal_strat = Strategy(
            strategy_id="LEGAL",
            name="Legal 1-Stop",
            num_stops=1,
            pit_laps=[25],
            compounds=["HARD"],
        )
        is_valid, reason = self.generator.validate_strategy(legal_strat, state, "VER")
        self.assertTrue(is_valid)

    def test_tyre_age_limit_pruning(self):
        """Strategies projecting tyre age past cliff reference must be rejected (hybrid or static gate)."""
        # Current tyre age is 32 on Medium (cliff reference 34). Pitting at lap 25 (+10 laps)
        # -> tyre reaches age 42, which is 8 laps beyond cliff -> significant cliff-excess pace loss
        state = self._create_standard_state(current_lap=15, tyre_life=32.0, compound="MEDIUM")

        infeasible_strat = Strategy(
            strategy_id="TOO_LATE",
            name="Late Pit Stop",
            num_stops=1,
            pit_laps=[25],  # 10 laps later -> age 42 (8 beyond cliff at 34)
            compounds=["HARD"],
        )
        is_valid, reason = self.generator.validate_strategy(infeasible_strat, state, "VER")
        self.assertFalse(is_valid)
        # Hybrid mode: reason mentions cliff-excess; static mode: mentions exceeding
        self.assertTrue(
            "cliff" in reason.lower() or "exceeding" in reason.lower(),
            f"Expected rejection reason to mention cliff or exceeding; got: {reason}",
        )

    def test_zero_stop_only_if_already_pitted(self):
        """A 0-stop strategy can only be valid if the car has already completed a mandatory stop."""
        # Case 1: 0 prior pits -> 0-stop is illegal
        state_unpitted = self._create_standard_state(current_lap=30, pit_count=0, compound="HARD")
        candidates_unpitted = self.generator.generate_candidates(state_unpitted, "VER")
        self.assertFalse(any(c.num_stops == 0 for c in candidates_unpitted))

        # Case 2: 1 prior pit -> 0-stop is legal if tyres can survive
        state_pitted = self._create_standard_state(current_lap=45, total_laps=53, tyre_life=10.0, pit_count=1, compound="HARD")
        candidates_pitted = self.generator.generate_candidates(state_pitted, "VER")
        zero_stops = [c for c in candidates_pitted if c.num_stops == 0]
        self.assertGreater(len(zero_stops), 0)
        self.assertEqual(zero_stops[0].num_stops, 0)
        self.assertEqual(zero_stops[0].pit_laps, [])

    def test_late_race_boundary_edge_case(self):
        """Verify behavior when called at race finish or 1 lap from end."""
        state = self._create_standard_state(current_lap=56, total_laps=57, tyre_life=20.0, pit_count=1)
        candidates = self.generator.generate_candidates(state, "VER")
        self.assertGreater(len(candidates), 0)
        # Should not schedule stops at or past lap 57
        for c in candidates:
            for p in c.pit_laps:
                self.assertLess(p, 57)


if __name__ == "__main__":
    unittest.main()
