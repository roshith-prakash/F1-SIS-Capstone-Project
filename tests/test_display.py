import io
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.display import (
    format_driver_detail,
    format_gap,
    format_in_out,
    format_lap_time,
    format_pits,
    format_pos_delta,
    format_race_state,
    format_race_state_compact,
    format_speeds,
    format_total_time,
    format_tyre_info,
    print_race_state,
)
from race_state.models import (
    ClassificationEntry,
    CurrentConditions,
    ParticipantState,
    RaceState,
)


class DisplayFormattingTests(unittest.TestCase):
    def setUp(self):
        self.state = RaceState(
            race_id="2024_12",
            year=2024,
            grand_prix="British Grand Prix",
            round=12,
            country="United Kingdom",
            location="Silverstone",
            current_lap=15,
            total_laps_seen=15,
            total_laps_expected=52,
            race_completion_pct=28.846,
            last_committed_lap=15,
            current_conditions=CurrentConditions(
                track_status="1",
                has_green=True,
                has_yellow=False,
                has_safety_car=False,
                has_vsc=False,
                has_red_flag=False,
                has_vsc_ending=False,
                air_temp=15.8,
                track_temp=25.2,
                humidity=69.0,
                pressure=992.5,
                rainfall=False,
                wind_direction="241",
                wind_speed=0.7,
            ),
        )

        p1 = ParticipantState(
            driver="VER",
            driver_number=1,
            driver_full_name="Max Verstappen",
            team="Red Bull Racing",
            constructor_season_id="Red_Bull_Racing_2024",
            position=1,
            previous_position=1,
            position_change=0,
            last_lap_number=15,
            last_lap_time_seconds=91.450,
            total_race_time_seconds=1371.75,
            compound="MEDIUM",
            tyre_life=15.0,
            stint=1.0,
            fresh_tyre=True,
            pit_count=0,
            last_pit_lap=None,
            is_pit_in_lap=False,
            is_pit_out_lap=False,
            laps_since_last_pit=15,
            gap_to_leader_seconds=0.0,
            interval_to_position_ahead_seconds=0.0,
            gap_behind_seconds=1.825,
            sector_1_time_seconds=29.120,
            sector_2_time_seconds=36.850,
            sector_3_time_seconds=25.480,
            speed_i1=298.0,
            speed_i2=260.0,
            speed_fl=248.0,
            speed_st=305.0,
            recent_lap_times=[92.100, 91.800, 91.450],
            rolling_3_lap_avg=91.783,
            rolling_5_lap_avg=91.950,
            is_active=True,
        )

        p2 = ParticipantState(
            driver="HAM",
            driver_number=44,
            driver_full_name="Lewis Hamilton",
            team="Mercedes",
            constructor_season_id="Mercedes_2024",
            position=2,
            previous_position=3,
            position_change=1,
            last_lap_number=15,
            last_lap_time_seconds=91.650,
            total_race_time_seconds=1373.575,
            compound="HARD",
            tyre_life=15.0,
            stint=1.0,
            fresh_tyre=False,
            pit_count=1,
            last_pit_lap=10,
            is_pit_in_lap=False,
            is_pit_out_lap=True,
            laps_since_last_pit=5,
            gap_to_leader_seconds=1.825,
            interval_to_position_ahead_seconds=1.825,
            gap_behind_seconds=3.200,
            sector_1_time_seconds=29.250,
            sector_2_time_seconds=36.900,
            sector_3_time_seconds=25.500,
            speed_i1=295.0,
            speed_i2=258.0,
            speed_fl=246.0,
            speed_st=302.0,
            recent_lap_times=[92.400, 91.900, 91.650],
            rolling_3_lap_avg=91.983,
            rolling_5_lap_avg=92.120,
            is_active=True,
        )

        self.state.participants = {"VER": p1, "HAM": p2}
        self.state.classification = [
            ClassificationEntry(
                driver="VER",
                team="Red Bull Racing",
                position=1,
                rank=1,
                gap_to_leader_seconds=0.0,
                interval_to_position_ahead_seconds=0.0,
                gap_behind_seconds=1.825,
            ),
            ClassificationEntry(
                driver="HAM",
                team="Mercedes",
                position=2,
                rank=2,
                gap_to_leader_seconds=1.825,
                interval_to_position_ahead_seconds=1.825,
                gap_behind_seconds=3.200,
            ),
        ]

    def test_format_lap_time(self):
        self.assertEqual(format_lap_time(None), "-")
        self.assertEqual(format_lap_time(91.773), "1:31.773")
        self.assertEqual(format_lap_time(59.999), "59.999")
        self.assertEqual(format_lap_time(60.005), "1:00.005")

    def test_format_total_time(self):
        self.assertEqual(format_total_time(None), "-")
        self.assertEqual(format_total_time(1371.750), "22:51.750")
        self.assertEqual(format_total_time(5000.500), "1:23:20.500")

    def test_format_gap(self):
        self.assertEqual(format_gap(0.0, is_leader=True), "")
        self.assertEqual(format_gap(1.534), "1.534")
        self.assertEqual(format_gap(None), "-")

    def test_format_pos_delta(self):
        self.assertEqual(format_pos_delta(0), "-")
        self.assertEqual(format_pos_delta(None), "-")
        self.assertEqual(format_pos_delta(2), "+2")
        self.assertEqual(format_pos_delta(-3), "-3")

    def test_format_tyre_info(self):
        self.assertEqual(format_tyre_info("MEDIUM", 14.0, True), "MEDIUM (L14) [F]")
        self.assertEqual(format_tyre_info("HARD", 20.0, False), "HARD (L20) [U]")
        self.assertEqual(format_tyre_info(None, None, None), "UNK (L0)")

    def test_format_in_out(self):
        self.assertEqual(format_in_out(False, False), "-")
        self.assertEqual(format_in_out(True, False), "IN")
        self.assertEqual(format_in_out(False, True), "OUT")
        self.assertEqual(format_in_out(True, True), "IN/OUT")

    def test_format_pits(self):
        self.assertEqual(format_pits(0, None), "0")
        self.assertEqual(format_pits(1, 10), "1 (L10)")

    def test_format_speeds(self):
        self.assertEqual(format_speeds(298.0, 260.0, 248.0, 305.0), "298/260/248/305")
        self.assertEqual(format_speeds(None, None, None, None), "-/-/-/-")

    def test_format_race_state_contains_all_features(self):
        output = format_race_state(self.state)

        # Header
        self.assertIn("Race: British Grand Prix (2024)", output)
        self.assertIn("Lap: 15/52", output)
        self.assertIn("Drivers tracked: 2", output)

        # Table 1: Classification & Timing
        self.assertIn("Classification & Timing:", output)
        self.assertIn("Pos", output)
        self.assertIn("Chg", output)
        self.assertIn("Driver", output)
        self.assertIn("Team", output)
        self.assertIn("Last Lap", output)
        self.assertIn("Total Time", output)
        self.assertIn("S1", output)
        self.assertIn("S2", output)
        self.assertIn("S3", output)
        self.assertIn("Gap (s)", output)
        self.assertIn("Int (s)", output)
        self.assertIn("Behind (s)", output)

        # Table 2: Tyre & Telemetry
        self.assertIn("Tyre, Pit Strategy & Telemetry:", output)
        self.assertIn("Tyre (Life)", output)
        self.assertIn("Stint", output)
        self.assertIn("Pits (Last)", output)
        self.assertIn("In/Out", output)
        self.assertIn("Roll 3L", output)
        self.assertIn("Speeds (I1/I2/FL/ST)", output)
        self.assertIn("Recent Laps (Last 5)", output)

        # Data rows
        self.assertIn("VER", output)
        self.assertIn("Red Bull Racing", output)
        self.assertIn("1:31.450", output)
        self.assertIn("22:51.750", output)
        self.assertIn("29.12", output)
        self.assertIn("36.85", output)
        self.assertIn("25.48", output)
        self.assertIn("MEDIUM (L15) [F]", output)
        self.assertIn("298/260/248/305", output)

        # Conditions
        self.assertIn("Conditions:", output)
        self.assertIn("Track status: 1", output)
        self.assertIn("Air/track temperature: 15.8 / 25.2", output)
        self.assertIn("Rainfall: False", output)
        self.assertIn("Safety car: False", output)

    def test_format_driver_detail(self):
        ver_card = format_driver_detail(self.state, "VER")
        self.assertIn("DRIVER TELEMETRY & STRATEGY CARD: VER (#1)", ver_card)
        self.assertIn("Max Verstappen", ver_card)
        self.assertIn("P1", ver_card)
        self.assertIn("MEDIUM", ver_card)
        self.assertIn("1:31.450", ver_card)

        missing_card = format_driver_detail(self.state, "NOR")
        self.assertIn("not found", missing_card)

    def test_print_race_state_and_compact(self):
        buf = io.StringIO()
        print_race_state(self.state, output=buf)
        self.assertIn("Race: British Grand Prix (2024)", buf.getvalue())

        compact_buf = io.StringIO()
        print_race_state(self.state, compact=True, output=compact_buf)
        self.assertIn("Pos  Driver  Team                         Gap (s)", compact_buf.getvalue())

        driver_buf = io.StringIO()
        print_race_state(self.state, driver="HAM", output=driver_buf)
        self.assertIn("DRIVER TELEMETRY & STRATEGY CARD: HAM (#44)", driver_buf.getvalue())


if __name__ == "__main__":
    unittest.main()
