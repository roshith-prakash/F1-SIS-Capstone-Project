from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.display import print_race_state
from race_state.replay import replay_csv_rows
from race_state.validation import validate_race_state


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a FastF1 CSV and display its race state.")
    parser.add_argument("csv_path", type=Path, help="Path to a lap CSV file")
    parser.add_argument(
        "--mode",
        choices=("row", "lap"),
        default="row",
        help="Replay rows individually or submit complete laps (default: row)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of classified drivers to display; use 0 for all (default: 10)",
    )
    parser.add_argument(
        "--driver",
        type=str,
        default=None,
        help="Inspect telemetry and strategy detail for a specific driver code (e.g. VER, HAM)",
    )
    parser.add_argument(
        "--compact",
        action="store_true",
        help="Display the minimal compact summary instead of the full internal model view",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate the complete internal race state before printing",
    )
    args = parser.parse_args()

    if not args.csv_path.is_file():
        parser.error(f"CSV file does not exist: {args.csv_path}")

    manager = replay_csv_rows(args.csv_path, mode=args.mode)
    state = manager.get_current_state()
    if args.validate:
        validate_race_state(state)
    print_race_state(
        state,
        participant_limit=None if args.limit == 0 else args.limit,
        compact=args.compact,
        driver=args.driver,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
