import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from lap_time.adapter import LapTimeAdapter
from race_state.manager import RaceStateManager
from race_state.models import ParticipantState, RaceState


class LapTimeAdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.adapter = LapTimeAdapter()

    def test_adapter_initialization(self):
        self.assertIsNotNone(self.adapter.model)
        self.assertIsNotNone(self.adapter.meta)
        self.assertEqual(len(self.adapter.circuits_set), 25)
        self.assertIn("Italy", self.adapter.circuits_set)
        self.assertIn("Great_Britain", self.adapter.circuits_set)

    def test_circuit_resolution(self):
        state1 = RaceState(location="Monza", country="Italy", grand_prix="Italian Grand Prix")
        self.assertEqual(self.adapter.resolve_circuit(state1), "Italy")

        state2 = RaceState(location="Silverstone", country="Great Britain", grand_prix="British Grand Prix")
        self.assertEqual(self.adapter.resolve_circuit(state2), "Great_Britain")

        state3 = RaceState(location="Yas Island", country="Abu Dhabi", grand_prix="Abu Dhabi Grand Prix")
        self.assertEqual(self.adapter.resolve_circuit(state3), "Abu_Dhabi")

    def test_team_resolution(self):
        self.assertEqual(self.adapter.resolve_team("Red Bull"), "Red Bull Racing")
        self.assertEqual(self.adapter.resolve_team("Ferrari"), "Ferrari")
        self.assertEqual(self.adapter.resolve_team("McLaren"), "McLaren")
        self.assertEqual(self.adapter.resolve_team("Haas"), "Haas F1 Team")

    def test_build_features(self):
        state = RaceState(location="Monza", country="Italy", grand_prix="Italian Grand Prix", current_lap=10, total_laps_expected=53)
        participant = ParticipantState(driver="VER", team="Red Bull Racing", position=1, compound="MEDIUM", tyre_life=10.0, stint=1.0, fresh_tyre=False)
        features = self.adapter.build_features(state, participant)

        self.assertIsNotNone(features)
        for col in self.adapter.feature_cols:
            self.assertIn(col, features)
        self.assertEqual(features["LapNumber"], 10)
        self.assertEqual(features["Compound_MEDIUM"], 1)
        self.assertEqual(features["Compound_HARD"], 0)
        self.assertEqual(features["Position"], 1)

    def test_predict_dry_vs_wet(self):
        state = RaceState(location="Monza", country="Italy", grand_prix="Italian Grand Prix", current_lap=10, total_laps_expected=53)
        state.participants["VER"] = ParticipantState(driver="VER", team="Red Bull Racing", position=1, compound="MEDIUM", tyre_life=10.0)
        state.participants["HAM"] = ParticipantState(driver="HAM", team="Mercedes", position=2, compound="INTERMEDIATE", tyre_life=5.0)

        pred_ver = self.adapter.predict_lap_time(state, "VER")
        self.assertIsNotNone(pred_ver)
        self.assertGreater(pred_ver, 70.0)
        self.assertLess(pred_ver, 120.0)

        # Intermediate should return None (dry-lap model only)
        pred_ham = self.adapter.predict_lap_time(state, "HAM")
        self.assertIsNone(pred_ham)

    def test_predict_all(self):
        state = RaceState(location="Silverstone", country="Great Britain", grand_prix="British Grand Prix", current_lap=15, total_laps_expected=52)
        state.participants["VER"] = ParticipantState(driver="VER", team="Red Bull Racing", position=1, compound="HARD", tyre_life=15.0, is_active=True)
        state.participants["NOR"] = ParticipantState(driver="NOR", team="McLaren", position=2, compound="MEDIUM", tyre_life=8.0, is_active=True)
        state.participants["RET"] = ParticipantState(driver="RET", team="Williams", position=None, compound="HARD", is_active=False)

        preds = self.adapter.predict_all(state)
        self.assertIn("VER", preds)
        self.assertIn("NOR", preds)
        self.assertNotIn("RET", preds)

    def test_missing_fields_edge_cases(self):
        state = RaceState()
        p_empty = ParticipantState(driver="TEST", compound="MEDIUM")
        features = self.adapter.build_features(state, p_empty)
        self.assertIsNotNone(features)
        self.assertEqual(features["TyreLife"], 1.0)
        self.assertEqual(features["Position"], 10)
        self.assertEqual(features["stint_number"], 1)

    def test_circuit_aliases_expanded(self):
        for loc, expected in [
            ("Spa-Francorchamps", "Belgium"),
            ("Interlagos", "Brazil"),
            ("Austin", "United_States"),
            ("Melbourne", "Australia"),
            ("Baku", "Azerbaijan"),
        ]:
            s = RaceState(location=loc)
            self.assertEqual(self.adapter.resolve_circuit(s), expected)

    def test_auto_calibration(self):
        adapter = LapTimeAdapter()
        state = RaceState(location="Monza", country="Italy", grand_prix="Italian Grand Prix", current_lap=3, total_laps_expected=53)
        
        # Simulate 6 active runners completing clean laps with realistic lap times (~84s)
        drivers = ["LEC", "PIA", "NOR", "VER", "HAM", "SAI"]
        for idx, d in enumerate(drivers, 1):
            state.participants[d] = ParticipantState(
                driver=d,
                team="Ferrari" if d in ["LEC", "SAI"] else "McLaren" if d in ["PIA", "NOR"] else "Red Bull Racing",
                position=idx,
                compound="MEDIUM",
                tyre_life=3.0,
                last_lap_time_seconds=84.0 + (idx * 0.1),
                is_active=True,
                is_pit_in_lap=False,
                is_pit_out_lap=False,
            )
            
        # Prior to calibration, calibrated_bases is empty
        self.assertNotIn("Italy", adapter.calibrated_bases)
        
        # predict_all triggers auto-calibration
        preds = adapter.predict_all(state)
        self.assertIn("Italy", adapter.calibrated_bases)
        calibrated_base = adapter.calibrated_bases["Italy"]
        
        # Ground-truth base should be ~82.5 - 83.5s
        self.assertGreater(calibrated_base, 80.0)
        self.assertLess(calibrated_base, 85.0)
        
    def test_caution_suppresses_calibration(self):
        adapter = LapTimeAdapter()
        state = RaceState(location="Monza", country="Italy", grand_prix="Italian Grand Prix", current_lap=5, total_laps_expected=53)
        state.current_conditions.has_safety_car = True
        
        state.participants["VER"] = ParticipantState(
            driver="VER", team="Red Bull Racing", position=1, compound="MEDIUM",
            tyre_life=5.0, last_lap_time_seconds=115.0, is_active=True
        )
        
        res = adapter.calibrate_from_state(state)
        self.assertIsNone(res)
        self.assertNotIn("Italy", adapter.calibrated_bases)


if __name__ == "__main__":
    unittest.main()
