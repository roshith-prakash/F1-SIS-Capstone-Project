import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.manager import RaceStateManager
from race_state.models import parse_lap_number
from sc_risk.adapter import SCRiskAdapter

PRIOR_PATH = ROOT / "models" / "SC Estimation" / "sc_vsc_historical_prior.csv"
CSV_PATH = ROOT / "data_fastf1_v1" / "laps" / "2024" / "Italian_Grand_Prix.csv"


class SCRiskAdapterTests(unittest.TestCase):
    def setUp(self):
        self.adapter = SCRiskAdapter(str(PRIOR_PATH))

    def test_feature_dict_structure_and_types(self):
        manager = RaceStateManager({"total_laps_expected": 53, "location": "Monza"})
        state = manager.flush()
        h_t = self.adapter.build_H_t(state)

        expected_keys = {
            "logit_p_sc_hist",
            "logit_p_vsc_hist",
            "close_battles",
            "field_spread",
            "sector_anomaly_score",
            "recent_retirements",
            "yellow_last3",
            "sc_count_cumul",
        }
        self.assertEqual(set(h_t.keys()), expected_keys)
        for k, v in h_t.items():
            self.assertIsInstance(v, float, f"{k} should be a float")

    def test_field_spread_excludes_stale_retired_drivers(self):
        manager = RaceStateManager({"total_laps_expected": 53, "location": "Monza"})
        # Simulate Lap 1 with 3 drivers
        lap1_rows = [
            {"LapNumber": 1, "Driver": "VER", "Team": "Red Bull", "Position": 1, "GapToLeaderSeconds": 0.0},
            {"LapNumber": 1, "Driver": "NOR", "Team": "McLaren", "Position": 2, "GapToLeaderSeconds": 2.0},
            {"LapNumber": 1, "Driver": "TSU", "Team": "RB", "Position": 3, "GapToLeaderSeconds": 10.0},
        ]
        manager.update_from_rows(lap1_rows)
        state1 = manager.commit_lap(1)
        h_t_1 = self.adapter.build_H_t(state1)
        self.assertGreater(h_t_1["field_spread"], 0)

        # Lap 2: TSU retires (has no row)
        lap2_rows = [
            {"LapNumber": 2, "Driver": "VER", "Team": "Red Bull", "Position": 1, "GapToLeaderSeconds": 0.0},
            {"LapNumber": 2, "Driver": "NOR", "Team": "McLaren", "Position": 2, "GapToLeaderSeconds": 4.0},
        ]
        manager.update_from_rows(lap2_rows)
        state2 = manager.commit_lap(2)
        h_t_2 = self.adapter.build_H_t(state2)

        # Active gaps on Lap 2 are [0.0, 4.0], std = sqrt(8) ~ 2.8284
        # If TSU's old 10.0 was leaked, std would be different
        self.assertAlmostEqual(h_t_2["field_spread"], 2.8284, places=3)

    def test_recent_retirements_guards_early_laps(self):
        manager = RaceStateManager({"total_laps_expected": 53, "location": "Monza"})
        # 3 cars on lap 1, 2 cars on lap 2
        manager.update_from_rows([
            {"LapNumber": 1, "Driver": "VER", "Team": "Red Bull", "Position": 1},
            {"LapNumber": 1, "Driver": "NOR", "Team": "McLaren", "Position": 2},
            {"LapNumber": 1, "Driver": "TSU", "Team": "RB", "Position": 3},
        ])
        manager.commit_lap(1)

        manager.update_from_rows([
            {"LapNumber": 2, "Driver": "VER", "Team": "Red Bull", "Position": 1},
            {"LapNumber": 2, "Driver": "NOR", "Team": "McLaren", "Position": 2},
        ])
        state2 = manager.commit_lap(2)
        h_t_2 = self.adapter.build_H_t(state2)

        # On lap 2, len(history) = 2 <= retirement_window (3), so recent_retirements must be 0
        self.assertEqual(h_t_2["recent_retirements"], 0.0)

    def test_location_accent_normalization(self):
        manager = RaceStateManager({"total_laps_expected": 70, "location": "Montréal"})
        state = manager.flush()
        # Should look up 'Montreal' or 'Montréal' without KeyError
        h_t = self.adapter.build_H_t(state)
        self.assertIn("logit_p_sc_hist", h_t)

    def test_lap_hazard_multiplier_phases(self):
        from sc_risk.adapter import get_lap_hazard_multiplier
        # Lap 1 Turn 1 incident spike
        self.assertEqual(get_lap_hazard_multiplier(1, is_sc_restart=False), 5.5)
        # Safety Car restart spike
        self.assertEqual(get_lap_hazard_multiplier(15, is_sc_restart=True), 2.2)
        # Cold tyre pack racing (Laps 2-3)
        self.assertEqual(get_lap_hazard_multiplier(2, is_sc_restart=False), 1.8)
        self.assertEqual(get_lap_hazard_multiplier(3, is_sc_restart=False), 1.8)
        # Mid-race field stasis (Lap 25 of 57)
        self.assertEqual(get_lap_hazard_multiplier(25, is_sc_restart=False, total_laps=57), 0.60)
        # Late-race tyre degradation punctures (Lap 55 of 57)
        self.assertEqual(get_lap_hazard_multiplier(55, is_sc_restart=False, total_laps=57), 1.1)

    def test_compute_modulated_hazard_prob(self):
        from sc_risk.adapter import compute_modulated_hazard_prob
        base_hazard = 0.028
        # Lap 1: 0.028 * 5.5 = 0.154
        p_lap1 = compute_modulated_hazard_prob(base_hazard, lap_number=1)
        self.assertAlmostEqual(p_lap1, 0.154, places=3)

        # Mid-race: 0.028 * 0.60 = 0.0168
        p_mid = compute_modulated_hazard_prob(base_hazard, lap_number=30, total_laps=57)
        self.assertAlmostEqual(p_mid, 0.0168, places=4)

        # Extreme clamping test
        self.assertLessEqual(compute_modulated_hazard_prob(0.50, lap_number=1), 0.35)
        self.assertGreaterEqual(compute_modulated_hazard_prob(0.00001, lap_number=30), 0.002)

    def test_adapter_instance_methods(self):
        self.assertEqual(self.adapter.get_lap_hazard_multiplier(1), 5.5)
        p = self.adapter.compute_modulated_hazard_prob(0.028, lap_number=1)
        self.assertAlmostEqual(p, 0.154, places=3)


if __name__ == "__main__":
    unittest.main()

