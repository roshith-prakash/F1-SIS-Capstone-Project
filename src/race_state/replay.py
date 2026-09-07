from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Iterator

from .manager import RaceStateManager
from .models import parse_lap_number


def load_csv_rows(csv_path: str | Path) -> Iterator[dict[str, Any]]:
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def replay_csv_rows(csv_path: str | Path, manager: RaceStateManager | None = None, mode: str = "row") -> RaceStateManager:
    manager = manager or RaceStateManager()
    rows = list(load_csv_rows(csv_path))
    if mode == "row":
        manager.update_from_rows(rows)
        manager.flush()
        return manager
    if mode == "lap":
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in rows:
            lap_number = parse_lap_number(row.get("LapNumber"))
            if lap_number is None:
                continue
            grouped.setdefault(lap_number, []).append(row)
        for lap_number in sorted(grouped):
            manager.update_from_rows(grouped[lap_number])
            manager.commit_lap(lap_number)
        return manager
    raise ValueError("mode must be 'row' or 'lap'")
