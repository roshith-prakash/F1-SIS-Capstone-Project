import sys
import unittest
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lap_time.pace_tracker import DynamicTeamPaceTracker, DEFAULT_TEAM_PACE_PRIORS
from lap_time.adapter import LapTimeAdapter
from tyre_deg.adapter import TyreDegAdapter
from strategy_engine.simulator import RaceScenarioSimulator
from strategy_engine.types import Strategy, StintPlan
from race_state.models import CurrentConditions, ParticipantState, RaceState


class TeamPaceTrackerTests(unittest.TestCase):
    def setUp(self):
        self.tracker = DynamicTeamPaceTracker(alpha=0.20)

    def test_canonicalization_and_priors(self):
        self.assertEqual(self.tracker.canonicalize_team("McLaren F1 Team"), "McLaren")
        self.assertEqual(self.tracker.canonicalize_team("mclaren"), "McLaren")
        self.assertEqual(self.tracker.canonicalize_team("Red Bull"), "Red Bull Racing")
        self.assertEqual(self.tracker.canonicalize_team("Scuderia Ferrari"), "Ferrari")
        self.assertEqual(self.tracker.canonicalize_team("VCARB"), "Racing Bulls")
        self.assertEqual(self.tracker.canonicalize_team("Kick Sauber"), "Kick Sauber")

        # McLaren 2024/2025 prior is faster than field baseline (< 0)
        mcl_offset = self.tracker.get_team_offset("McLaren")
        self.assertLess(mcl_offset, -0.5)

        # Kick Sauber is slower (> 0)
        sau_offset = self.tracker.get_team_offset("Kick Sauber")
        self.assertGreater(sau_offset, 0.5)

    def test_ema_adaptation_clean_laps(self):
        tracker = DynamicTeamPaceTracker(alpha=0.25)
        initial = tracker.get_team_offset("McLaren")

        # Simulate McLaren bringing an upgrade package performing -0.4s faster than expected
        updated_1 = tracker.observe_lap_residual("McLaren", initial - 0.40)
        self.assertLess(updated_1, initial)

        updated_2 = tracker.observe_lap_residual("McLaren", initial - 0.40)
        self.assertLess(updated_2, updated_1)

        # After several consistent laps, tracker smoothly tracks towards new pace
        for _ in range(5):
            tracker.observe_lap_residual("McLaren", initial - 0.40)

        self.assertAlmostEqual(tracker.get_team_offset("McLaren"), initial - 0.40, delta=0.10)

    def test_outlier_residual_clipping(self):
        tracker = DynamicTeamPaceTracker(alpha=0.20, residual_clip=(-1.5, 1.5))
        init_offset = tracker.get_team_offset("Ferrari")

        # Driver has puncture or traffic lockup causing +8.0s spike
        updated = tracker.observe_lap_residual("Ferrari", 8.0)
        
        # Max update should be constrained by +1.5s clip: (1 - 0.2)*init + 0.2*1.5
        expected = 0.8 * init_offset + 0.2 * 1.5
        self.assertAlmostEqual(updated, expected, places=4)

    def test_caution_freezes_state_updates(self):
        tracker = DynamicTeamPaceTracker()
        before_offsets = tracker.get_all_offsets()

        cond = CurrentConditions(has_safety_car=True)
        state = RaceState(current_lap=12, total_laps_expected=50, current_conditions=cond)
        state.participants["NOR"] = ParticipantState(
            driver="NOR", team="McLaren", last_lap_time_seconds=120.0, is_active=True
        )

        after_offsets = tracker.update_from_race_state(state, circuit_base=85.0)
        self.assertEqual(before_offsets, after_offsets)

    def test_lap_time_adapter_integration(self):
        adapter = LapTimeAdapter()
        state = RaceState(location="Monza", grand_prix="Italian Grand Prix", current_lap=5, total_laps_expected=53)
        state.participants["NOR"] = ParticipantState(driver="NOR", team="McLaren", position=1, compound="MEDIUM", tyre_life=5.0, is_active=True)
        state.participants["BOT"] = ParticipantState(driver="BOT", team="Kick Sauber", position=18, compound="MEDIUM", tyre_life=5.0, is_active=True)

        pred_nor = adapter.predict_lap_time(state, "NOR")
        pred_bot = adapter.predict_lap_time(state, "BOT")

        self.assertIsNotNone(pred_nor)
        self.assertIsNotNone(pred_bot)
        # McLaren pace should be significantly faster than Kick Sauber on identical tyres
        self.assertLess(pred_nor, pred_bot)

    def test_tyre_deg_adapter_integration(self):
        tracker = DynamicTeamPaceTracker()
        adapter = TyreDegAdapter(pace_tracker=tracker, circuit_base_pace=85.0)

        cond = CurrentConditions(track_temp=30.0, air_temp=25.0)
        state = RaceState(current_lap=10, total_laps_expected=50, current_conditions=cond)
        p = ParticipantState(driver="PIA", team="McLaren", compound="HARD", tyre_life=10.0, is_active=True)

        feats = adapter.build_features(state, p)
        # Team_Median_Pace_Lag1 should reflect dynamic tracker offset
        self.assertAlmostEqual(feats["Team_Median_Pace_Lag1"], tracker.get_team_offset("McLaren"))

    def test_simulator_rollout_with_team_pace(self):
        sim = RaceScenarioSimulator(default_rollouts=20, random_seed=42)
        state = RaceState(location="Bahrain", grand_prix="Bahrain Grand Prix", current_lap=1, total_laps_expected=57)
        state.participants["NOR"] = ParticipantState(
            driver="NOR", team="McLaren", position=1, compound="MEDIUM", tyre_life=1.0, is_active=True, last_lap_time_seconds=92.0
        )
        state.participants["VER"] = ParticipantState(
            driver="VER", team="Red Bull Racing", position=2, compound="MEDIUM", tyre_life=1.0, is_active=True, last_lap_time_seconds=92.2
        )
        state.participants["BOT"] = ParticipantState(
            driver="BOT", team="Kick Sauber", position=15, compound="MEDIUM", tyre_life=1.0, is_active=True, last_lap_time_seconds=94.5
        )

        strat = Strategy(
            strategy_id="1S_M_H_L20",
            name="M-H",
            num_stops=1,
            compounds=["MEDIUM", "HARD"],
            pit_laps=[20],
            stints=[
                StintPlan(stint_number=1, compound="MEDIUM", target_laps=19, start_lap=1, end_lap=20),
                StintPlan(stint_number=2, compound="HARD", target_laps=37, start_lap=21, end_lap=57),
            ],
        )

        res = sim.simulate_strategy(state, ego_driver="NOR", strategy=strat, horizon_laps=10, n_rollouts=25)
        # Starting P1 with the dominant McLaren in Bahrain, expected finish should be top tier
        self.assertLessEqual(res.expected_position, 3.0)


if __name__ == "__main__":
    unittest.main()
