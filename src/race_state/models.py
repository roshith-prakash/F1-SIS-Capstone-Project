from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


def safe_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text in {"", "nan", "NaN", "None", "null"}:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> Optional[int]:
    num = safe_float(value)
    if num is None:
        return None
    return int(num)


def safe_bool(value: Any) -> Optional[bool]:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"true", "1", "yes", "y"}:
        return True
    if text in {"false", "0", "no", "n"}:
        return False
    return None


def normalize_driver(driver: Any) -> Optional[str]:
    if driver is None or str(driver).strip() == "":
        return None
    return str(driver).strip().upper()


def normalize_team(team: Any) -> Optional[str]:
    if team is None or str(team).strip() == "":
        return None
    cleaned = str(team).strip()
    return cleaned.replace(" ", "_").replace("-", "_")


def parse_lap_number(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        if isinstance(value, float) and not value.is_integer():
            return int(value)
        return int(value)
    text = str(value).strip()
    if text == "":
        return None
    try:
        numeric = float(text)
    except ValueError:
        return None
    if numeric.is_integer():
        return int(numeric)
    return int(numeric)


@dataclass
class CurrentConditions:
    track_status: Optional[str] = None
    has_green: Optional[bool] = None
    has_yellow: Optional[bool] = None
    has_safety_car: Optional[bool] = None
    has_vsc: Optional[bool] = None
    has_red_flag: Optional[bool] = None
    has_vsc_ending: Optional[bool] = None
    air_temp: Optional[float] = None
    track_temp: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    rainfall: Optional[bool] = None
    wind_direction: Optional[str] = None
    wind_speed: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ClassificationEntry:
    driver: str
    team: Optional[str] = None
    position: Optional[int] = None
    rank: Optional[int] = None
    gap_to_leader_seconds: Optional[float] = None
    interval_to_position_ahead_seconds: Optional[float] = None
    gap_behind_seconds: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParticipantState:
    driver: Optional[str] = None
    driver_number: Optional[int] = None
    driver_full_name: Optional[str] = None
    team: Optional[str] = None
    constructor_season_id: Optional[str] = None
    position: Optional[int] = None
    previous_position: Optional[int] = None
    position_change: Optional[int] = None
    last_lap_number: Optional[int] = None
    last_lap_time_seconds: Optional[float] = None
    total_race_time_seconds: Optional[float] = None
    time_seconds: Optional[float] = None
    compound: Optional[str] = None
    tyre_life: Optional[float] = None
    stint: Optional[float] = None
    fresh_tyre: Optional[bool] = None
    pit_count: int = 0
    last_pit_lap: Optional[int] = None
    is_pit_in_lap: bool = False
    is_pit_out_lap: bool = False
    laps_since_last_pit: Optional[int] = None
    gap_to_leader_seconds: Optional[float] = None
    interval_to_position_ahead_seconds: Optional[float] = None
    gap_behind_seconds: Optional[float] = None
    sector_1_time_seconds: Optional[float] = None
    sector_2_time_seconds: Optional[float] = None
    sector_3_time_seconds: Optional[float] = None
    speed_i1: Optional[float] = None
    speed_i2: Optional[float] = None
    speed_fl: Optional[float] = None
    speed_st: Optional[float] = None
    recent_lap_times: list[float] = field(default_factory=list)
    rolling_3_lap_avg: Optional[float] = None
    rolling_5_lap_avg: Optional[float] = None
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ConstructorState:
    constructor_season_id: str
    team: str
    drivers: list[str] = field(default_factory=list)
    lead_driver: Optional[str] = None
    second_driver: Optional[str] = None
    best_position: Optional[int] = None
    worst_position: Optional[int] = None
    lead_gap_to_leader_seconds: Optional[float] = None
    intra_team_gap_seconds: Optional[float] = None
    total_pit_stops: int = 0
    is_double_stack_risk: bool = False
    rolling_3_lap_avg: Optional[float] = None
    is_active: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RaceState:
    race_id: Optional[str] = None
    year: Optional[int] = None
    grand_prix: Optional[str] = None
    round: Optional[int] = None
    country: Optional[str] = None
    location: Optional[str] = None
    current_lap: Optional[int] = None
    total_laps_seen: int = 0
    total_laps_expected: Optional[int] = None
    race_completion_pct: Optional[float] = None
    current_conditions: CurrentConditions = field(default_factory=CurrentConditions)
    constructors: dict[str, ConstructorState] = field(default_factory=dict)
    participants: dict[str, ParticipantState] = field(default_factory=dict)
    classification: list[ClassificationEntry] = field(default_factory=list)
    lap_history: list[dict[str, Any]] = field(default_factory=list)
    last_committed_lap: Optional[int] = None
    race_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._sync_race_info()

    def _sync_race_info(self) -> None:
        self.race_info = {
            "race_id": self.race_id,
            "year": self.year,
            "grand_prix": self.grand_prix,
            "round": self.round,
            "country": self.country,
            "location": self.location,
            "total_laps_expected": self.total_laps_expected,
        }

    def to_dict(self) -> dict[str, Any]:
        self._sync_race_info()
        return {
            "race_id": self.race_id,
            "year": self.year,
            "grand_prix": self.grand_prix,
            "round": self.round,
            "country": self.country,
            "location": self.location,
            "current_lap": self.current_lap,
            "total_laps_seen": self.total_laps_seen,
            "total_laps_expected": self.total_laps_expected,
            "race_completion_pct": self.race_completion_pct,
            "current_conditions": self.current_conditions.to_dict(),
            "constructors": {cid: c.to_dict() for cid, c in self.constructors.items()},
            "participants": {driver: participant.to_dict() for driver, participant in self.participants.items()},
            "classification": [entry.to_dict() for entry in self.classification],
            "lap_history": self.lap_history,
            "last_committed_lap": self.last_committed_lap,
            "race_info": self.race_info,
        }
