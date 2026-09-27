import sys
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tyre_deg.adapter import TyreDegAdapter, COMPOUND_MAP, FEATURE_COLS
from race_state.models import ParticipantState, CurrentConditions, RaceState


class DummyModel:
    """Mock XGBRegressor for isolated deterministic unit tests."""
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        # Predict linear degradation: 0.05s per tyre age lap
        return 0.05 * df["TyreAge"].values


class TyreDegAdapterTests(unittest.TestCase):
    def setUp(self):
        self.mock_model = DummyModel()
        self.adapter = TyreDegAdapter(model=self.mock_model, circuit_base_pace=80.0)

    def test_feature_columns_schema(self):
        self.assertEqual(len(FEATURE_COLS), 12)
        expected_cols = [
            "TyreAge",
            "Log_TyreAge",
            "StintLap",
            "Compound_Encoded",
            "RaceProgressFraction",
            "TrackTemp",
            "AirTemp",
            "is_safety_car",
            "is_vsc",
            "Lag1_Pace_Residual",
            "Rolling3_Degradation_Trend",
            "Team_Median_Pace_Lag1",
        ]
        self.assertEqual(FEATURE_COLS, expected_cols)

    def test_build_features_clean(self):
        cond = CurrentConditions(air_temp=26.5, track_temp=38.0, has_safety_car=False, has_vsc=False)
        state = RaceState(
            current_lap=20,
            total_laps_expected=50,
            current_conditions=cond,
        )
        participant = ParticipantState(
            driver="VER",
            team="Red Bull",
            compound="HARD",
            tyre_life=15.0,
            laps_since_last_pit=14,
            stint=1,
            last_lap_time_seconds=82.5,
        )

        features = self.adapter.build_features(state, participant)
        for col in FEATURE_COLS:
            self.assertIn(col, features)

        self.assertEqual(features["TyreAge"], 15.0)
        self.assertAlmostEqual(features["Log_TyreAge"], np.log1p(15.0))
        self.assertEqual(features["StintLap"], 15.0)
        self.assertEqual(features["Compound_Encoded"], COMPOUND_MAP["HARD"])
        self.assertAlmostEqual(features["RaceProgressFraction"], 0.4)
        self.assertAlmostEqual(features["TrackTemp"], 38.0)
        self.assertAlmostEqual(features["AirTemp"], 26.5)
        self.assertEqual(features["is_safety_car"], 0)
        self.assertEqual(features["is_vsc"], 0)

    def test_compound_encoding(self):
        self.assertEqual(COMPOUND_MAP["SOFT"], 1)
        self.assertEqual(COMPOUND_MAP["MEDIUM"], 2)
        self.assertEqual(COMPOUND_MAP["HARD"], 3)
        self.assertEqual(COMPOUND_MAP["INTERMEDIATE"], -1)
        self.assertEqual(COMPOUND_MAP["WET"], -2)

    def test_stint_tracking_and_residuals(self):
        cond = CurrentConditions(track_temp=30.0, air_temp=25.0)
        # Lap 1
        s1 = RaceState(current_lap=1, total_laps_expected=50, current_conditions=cond)
        s1.participants["NOR"] = ParticipantState(driver="NOR", stint=1, tyre_life=1.0, laps_since_last_pit=0, last_lap_time_seconds=84.0, is_active=True)
        f1 = self.adapter.build_features(s1, s1.participants["NOR"])
        self.assertEqual(f1["Lag1_Pace_Residual"], 0.0)

        # Observe lap 1 completion
        self.adapter.observe_lap(s1)

        # Lap 2
        s2 = RaceState(current_lap=2, total_laps_expected=50, current_conditions=cond)
        s2.participants["NOR"] = ParticipantState(driver="NOR", stint=1, tyre_life=2.0, laps_since_last_pit=1, last_lap_time_seconds=84.2, is_active=True)
        f2 = self.adapter.build_features(s2, s2.participants["NOR"])
        self.assertNotEqual(f2["Lag1_Pace_Residual"], 0.0)

        # Observe lap 2 completion
        self.adapter.observe_lap(s2)

        # New stint (pit stop) should reset residuals
        s3 = RaceState(current_lap=3, total_laps_expected=50, current_conditions=cond)
        s3.participants["NOR"] = ParticipantState(driver="NOR", stint=2, tyre_life=1.0, laps_since_last_pit=0, last_lap_time_seconds=84.5, is_active=True)
        f3 = self.adapter.build_features(s3, s3.participants["NOR"])
        self.assertEqual(f3["Lag1_Pace_Residual"], 0.0)

    def test_repeated_build_features_is_idempotent(self):
        adapter = TyreDegAdapter(model=self.mock_model, circuit_base_pace=80.0)
        cond = CurrentConditions(track_temp=30.0, air_temp=25.0)
        s = RaceState(current_lap=5, total_laps_expected=50, current_conditions=cond)
        p = ParticipantState(driver="VER", stint=1, tyre_life=5.0, last_lap_time_seconds=82.0, is_active=True)
        
        # Multiple calls to build_features must not duplicate residuals
        f_a = adapter.build_features(s, p)
        f_b = adapter.build_features(s, p)
        self.assertEqual(f_a["Lag1_Pace_Residual"], f_b["Lag1_Pace_Residual"])
        self.assertEqual(len(adapter._driver_residuals.get("VER", [])), 0)

    def test_circuit_resolution_and_base_pace(self):
        adapter = TyreDegAdapter(model=self.mock_model)
        
        s_austria = RaceState(location="Spielberg", grand_prix="Austrian Grand Prix")
        self.assertEqual(adapter.resolve_circuit(s_austria), "Austria")
        self.assertAlmostEqual(adapter.get_circuit_base(s_austria), 67.583)

        s_spa = RaceState(location="Spa-Francorchamps", grand_prix="Belgian Grand Prix")
        self.assertEqual(adapter.resolve_circuit(s_spa), "Belgium")
        self.assertAlmostEqual(adapter.get_circuit_base(s_spa), 107.305)

        s_monza = RaceState(location="Monza", grand_prix="Italian Grand Prix")
        self.assertEqual(adapter.resolve_circuit(s_monza), "Italy")
        self.assertAlmostEqual(adapter.get_circuit_base(s_monza), 84.030)

    def test_caution_suppresses_residual_tracking(self):
        adapter = TyreDegAdapter(model=self.mock_model, circuit_base_pace=80.0)
        cond = CurrentConditions(has_safety_car=True)
        s = RaceState(current_lap=10, total_laps_expected=50, current_conditions=cond)
        s.participants["HAM"] = ParticipantState(driver="HAM", last_lap_time_seconds=120.0, is_active=True)

        adapter.observe_lap(s)
        self.assertNotIn("HAM", adapter._driver_residuals)

    def test_predict_degradation_and_lap_time(self):
        state = RaceState(current_lap=10, total_laps_expected=50)
        state.participants["VER"] = ParticipantState(driver="VER", tyre_life=20.0, compound="MEDIUM", is_active=True)
        state.participants["HAM"] = ParticipantState(driver="HAM", tyre_life=10.0, compound="HARD", is_active=True)
        state.participants["BOT"] = ParticipantState(driver="BOT", tyre_life=5.0, compound="SOFT", is_active=False)

        pred_ver = self.adapter.predict_degradation(state, "VER")
        self.assertAlmostEqual(pred_ver, 0.05 * 20.0)

        pred_ham = self.adapter.predict_degradation(state, "HAM")
        self.assertAlmostEqual(pred_ham, 0.05 * 10.0)

        # Inactive driver predict_all should be omitted
        all_preds = self.adapter.predict_all(state)
        self.assertIn("VER", all_preds)
        self.assertIn("HAM", all_preds)
        self.assertNotIn("BOT", all_preds)

        # Predict reconstructed lap time
        # base_pace = 80.0, remaining laps = 50 - 10 = 40 => fuel = 40 * 0.065 = 2.6
        # deg = 1.0 => lap time = 80.0 + 2.6 + 1.0 = 83.6
        lt_ver = self.adapter.predict_lap_time(state, "VER", base_pace=80.0)
        self.assertAlmostEqual(lt_ver, 80.0 + (40 * 0.065) + 1.0)

    def test_edge_cases_and_missing_values(self):
        state = RaceState()
        p_empty = ParticipantState(driver="TEST")
        features = self.adapter.build_features(state, p_empty)
        self.assertEqual(features["TyreAge"], 1.0)
        self.assertEqual(features["StintLap"], 1.0)
        self.assertEqual(features["Compound_Encoded"], 2)  # default MEDIUM
        self.assertEqual(features["TrackTemp"], 30.0)
        self.assertEqual(features["AirTemp"], 25.0)


if __name__ == "__main__":
    unittest.main()
