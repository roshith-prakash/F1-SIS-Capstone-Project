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
from race_state.validation import REQUIRED_PARTICIPANT_FIELDS, validate_race_state

CSV_PATH = ROOT / "data_fastf1_v1" / "laps" / "2024" / "British_Grand_Prix.csv"


def read_csv_rows():
    with open(CSV_PATH, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


class RaceStateManagerTests(unittest.TestCase):
    def _rows_for_laps(self, max_lap):
        rows = []
        for row in read_csv_rows():
            lap_number = parse_lap_number(row.get("LapNumber"))
            if lap_number is not None and lap_number <= max_lap:
                rows.append(row)
        return rows

    def test_lap_one_commit_creates_full_race_snapshot(self):
        manager = RaceStateManager()
        lap_one_rows = self._rows_for_laps(1)
        manager.update_from_rows(lap_one_rows)
        state = manager.flush()
        validate_race_state(state)

        self.assertEqual(state.current_lap, 1)
        self.assertGreater(len(state.participants), 1)
        self.assertEqual(len(state.classification), len(state.participants))
        self.assertEqual(
            [entry.position for entry in state.classification],
            sorted(pos for entry in state.classification if (pos := entry.position) is not None),
        )

        example = state.participants.get("VER")
        self.assertIsNotNone(example)
        assert example is not None
        for field_name in REQUIRED_PARTICIPANT_FIELDS:
            self.assertTrue(hasattr(example, field_name), field_name)
        self.assertIsNotNone(example.last_lap_time_seconds)
        self.assertIsNotNone(example.total_race_time_seconds)
        self.assertIsNotNone(example.compound)
        self.assertIsNotNone(example.team)
        self.assertIsInstance(example.recent_lap_times, list)

    def test_row_by_row_and_lap_batch_match_after_same_commit(self):
        all_rows = self._rows_for_laps(3)

        row_by_row = RaceStateManager()
        for row in all_rows:
            row_by_row.update_from_row(row)
        row_state = row_by_row.flush()

        lap_batch = RaceStateManager()
        lap_numbers = sorted({
            lap for row in all_rows
            if (lap := parse_lap_number(row.get("LapNumber"))) is not None
        })
        for lap_number in lap_numbers:
            lap_rows = [row for row in all_rows if parse_lap_number(row.get("LapNumber")) == lap_number]
            lap_batch.update_from_rows(lap_rows)
            lap_batch.commit_lap(lap_number)

        self.assertEqual(row_state.current_lap, lap_batch.get_current_state().current_lap)
        self.assertEqual(row_state.to_dict()["participants"]["VER"]["last_lap_time_seconds"], lap_batch.get_current_state().to_dict()["participants"]["VER"]["last_lap_time_seconds"])
        validate_race_state(row_state)
        validate_race_state(lap_batch.get_current_state())

    def test_rolling_history_uses_committed_laps_only(self):
        rows = [row for row in read_csv_rows() if row.get("Driver") == "VER" and parse_lap_number(row.get("LapNumber")) in (1, 2, 3)]
        manager = RaceStateManager()
        manager.update_from_rows(rows)
        state = manager.flush()

        participant = state.participants["VER"]
        expected = (96.711 + 91.773 + 92.2) / 3
        self.assertIsNotNone(participant.rolling_3_lap_avg)
        assert participant.rolling_3_lap_avg is not None
        self.assertAlmostEqual(participant.rolling_3_lap_avg, expected, places=3)
        self.assertEqual(participant.recent_lap_times[-3:], [96.711, 91.773, 92.2])
        validate_race_state(state)

    def test_pit_flags_and_counts_are_derived(self):
        all_rows = read_csv_rows()
        pit_rows = [row for row in all_rows if row.get("Driver") == "VER" and (row.get("PitInTimeSeconds") not in (None, "") or row.get("PitOutTimeSeconds") not in (None, ""))]
        self.assertTrue(pit_rows)

        manager = RaceStateManager()
        manager.update_from_rows(all_rows)
        state = manager.flush()

        participant = state.participants["VER"]
        self.assertTrue(participant.pit_count >= 1)
        self.assertIsNotNone(participant.last_pit_lap)
        assert participant.last_pit_lap is not None
        self.assertGreaterEqual(participant.last_pit_lap, 1)
        validate_race_state(state)

    def test_current_conditions_update_from_lap_rows(self):
        lap_rows = self._rows_for_laps(1)
        manager = RaceStateManager()
        manager.update_from_rows(lap_rows)
        manager.flush()

        conditions = manager.get_current_state().current_conditions
        self.assertIsNotNone(conditions.track_status)
        self.assertIsNotNone(conditions.air_temp)
        self.assertIsNotNone(conditions.track_temp)
        validate_race_state(manager.get_current_state())

    def test_constructors_are_grouped_and_tracked_by_season_id(self):
        manager = RaceStateManager()
        manager.update_from_rows(self._rows_for_laps(5))
        state = manager.flush()
        validate_race_state(state)

        self.assertGreater(len(state.constructors), 1)
        mclaren = state.constructors.get("McLaren_2024")
        self.assertIsNotNone(mclaren)
        assert mclaren is not None
        self.assertEqual(len(mclaren.drivers), 2)
        self.assertIn("NOR", mclaren.drivers)
        self.assertIn("PIA", mclaren.drivers)
        self.assertIsNotNone(mclaren.lead_driver)
        self.assertIsNotNone(mclaren.second_driver)
        self.assertIsNotNone(mclaren.intra_team_gap_seconds)
        self.assertGreater(mclaren.intra_team_gap_seconds, 0)
        self.assertIsNotNone(mclaren.best_position)
        self.assertIsNotNone(mclaren.worst_position)
        self.assertLessEqual(mclaren.best_position, mclaren.worst_position)


if __name__ == "__main__":
    unittest.main()
