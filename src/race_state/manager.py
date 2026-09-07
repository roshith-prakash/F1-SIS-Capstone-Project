from __future__ import annotations

from typing import Any, Iterable

from .models import (
    ClassificationEntry,
    ConstructorState,
    CurrentConditions,
    ParticipantState,
    RaceState,
    normalize_driver,
    normalize_team,
    parse_lap_number,
    safe_bool,
    safe_float,
    safe_int,
)


def validate_state(state: RaceState | None) -> None:
    if state is None:
        raise ValueError("state cannot be None")
    if not isinstance(state, RaceState):
        raise TypeError("state must be a RaceState instance")

    required_participant_fields = [
        "driver",
        "team",
        "position",
        "last_lap_time_seconds",
        "total_race_time_seconds",
        "compound",
        "tyre_life",
        "stint",
        "pit_count",
        "last_pit_lap",
        "is_pit_in_lap",
        "is_pit_out_lap",
        "interval_to_position_ahead_seconds",
        "gap_to_leader_seconds",
        "recent_lap_times",
        "rolling_3_lap_avg",
        "is_active",
    ]

    for driver, participant in state.participants.items():
        missing = [field for field in required_participant_fields if not hasattr(participant, field)]
        if missing:
            raise ValueError(f"{driver} is missing participant state fields: {missing}")

        if participant.driver is None:
            raise ValueError(f"{driver} is missing a driver code")
        if participant.team is None:
            raise ValueError(f"{driver} is missing a team")
        if participant.position is None:
            raise ValueError(f"{driver} is missing a position")
        if participant.recent_lap_times is None:
            raise ValueError(f"{driver} recent_lap_times is None")


class RaceStateManager:
    def __init__(self, metadata: dict[str, Any] | None = None):
        self.state = RaceState()
        self.buffer: dict[int, list[dict[str, Any]]] = {}
        self._last_seen_lap: int | None = None
        self.start_race(metadata)

    def start_race(self, metadata: dict[str, Any] | None = None) -> RaceState:
        metadata = metadata or {}
        self._apply_race_metadata(metadata)
        return self.state

    def _apply_race_metadata(self, metadata: dict[str, Any]) -> None:
        if not isinstance(metadata, dict):
            raise TypeError("race metadata must be a dictionary")
        if "year" in metadata and metadata["year"] is not None:
            self.state.year = safe_int(metadata["year"])
        if "grand_prix" in metadata and metadata["grand_prix"] is not None:
            self.state.grand_prix = str(metadata["grand_prix"])
        if "grandPrix" in metadata and metadata["grandPrix"] is not None:
            self.state.grand_prix = str(metadata["grandPrix"])
        if "round" in metadata and metadata["round"] is not None:
            self.state.round = safe_int(metadata["round"])
        if "country" in metadata and metadata["country"] is not None:
            self.state.country = str(metadata["country"])
        if "location" in metadata and metadata["location"] is not None:
            self.state.location = str(metadata["location"])
        if "race_id" in metadata and metadata["race_id"] is not None:
            self.state.race_id = str(metadata["race_id"])
        if "total_laps_expected" in metadata and metadata["total_laps_expected"] is not None:
            self.state.total_laps_expected = safe_int(metadata["total_laps_expected"])
        self.state._sync_race_info()

    def _validate_row(self, row: dict[str, Any]) -> None:
        if not isinstance(row, dict):
            raise TypeError("row must be a dictionary")
        if self.state.year is not None and row.get("Year") not in (None, ""):
            row_year = safe_int(row.get("Year"))
            if row_year is not None and row_year != self.state.year:
                raise ValueError(f"row year {row_year} does not match active race year {self.state.year}")
        if self.state.grand_prix is not None and row.get("GrandPrix") not in (None, ""):
            if str(row.get("GrandPrix")) != str(self.state.grand_prix):
                raise ValueError(
                    f"row grand prix {row.get('GrandPrix')} does not match active race {self.state.grand_prix}"
                )

    def _seed_race_metadata_from_row(self, row: dict[str, Any]) -> None:
        metadata: dict[str, Any] = {}
        if row.get("Year") not in (None, ""):
            metadata["year"] = row.get("Year")
        if row.get("GrandPrix") not in (None, ""):
            metadata["grand_prix"] = row.get("GrandPrix")
        if row.get("Round") not in (None, ""):
            metadata["round"] = row.get("Round")
        if row.get("Country") not in (None, ""):
            metadata["country"] = row.get("Country")
        if row.get("Location") not in (None, ""):
            metadata["location"] = row.get("Location")
        if row.get("LapNumber") not in (None, ""):
            metadata["total_laps_expected"] = safe_int(row.get("LapNumber"))
        self._apply_race_metadata(metadata)

    def update_from_row(self, row: dict[str, Any]) -> None:
        if row is None:
            return
        self._seed_race_metadata_from_row(row)
        self._validate_row(row)
        lap_number = parse_lap_number(row.get("LapNumber"))
        if lap_number is None:
            return
        self.buffer.setdefault(lap_number, []).append(dict(row))
        if self._last_seen_lap is not None and lap_number != self._last_seen_lap:
            self.commit_lap(self._last_seen_lap)
        self._last_seen_lap = lap_number

    def update_from_rows(self, rows: Iterable[dict[str, Any]]) -> None:
        for row in rows:
            self.update_from_row(row)

    def _update_current_conditions(self, rows: list[dict[str, Any]]) -> None:
        condition_fields = {
            "track_status": lambda value: str(value) if value not in (None, "") else None,
            "has_green": safe_bool,
            "has_yellow": safe_bool,
            "has_safety_car": safe_bool,
            "has_vsc": safe_bool,
            "has_red_flag": safe_bool,
            "has_vsc_ending": safe_bool,
            "air_temp": safe_float,
            "track_temp": safe_float,
            "humidity": safe_float,
            "pressure": safe_float,
            "rainfall": safe_bool,
            "wind_direction": lambda value: str(value) if value not in (None, "") else None,
            "wind_speed": safe_float,
        }
        for field_name, parser in condition_fields.items():
            last_value = None
            for row in rows:
                value = row.get(field_name.replace("_", "").title().replace("", ""))
                if field_name == "track_status":
                    value = row.get("TrackStatus")
                elif field_name == "has_green":
                    value = row.get("HasGreen")
                elif field_name == "has_yellow":
                    value = row.get("HasYellow")
                elif field_name == "has_safety_car":
                    value = row.get("HasSafetyCar")
                elif field_name == "has_vsc":
                    value = row.get("HasVSC")
                elif field_name == "has_red_flag":
                    value = row.get("HasRedFlag")
                elif field_name == "has_vsc_ending":
                    value = row.get("HasVSCEnding")
                elif field_name == "air_temp":
                    value = row.get("AirTemp")
                elif field_name == "track_temp":
                    value = row.get("TrackTemp")
                elif field_name == "humidity":
                    value = row.get("Humidity")
                elif field_name == "pressure":
                    value = row.get("Pressure")
                elif field_name == "rainfall":
                    value = row.get("Rainfall")
                elif field_name == "wind_direction":
                    value = row.get("WindDirection")
                elif field_name == "wind_speed":
                    value = row.get("WindSpeed")
                if value not in (None, ""):
                    last_value = parser(value)
            setattr(self.state.current_conditions, field_name, last_value)

    def _apply_row_to_participant(self, lap_number: int, row: dict[str, Any]) -> None:
        driver_code = normalize_driver(row.get("Driver"))
        if driver_code is None:
            return

        participant = self.state.participants.get(driver_code)
        if participant is None:
            participant = ParticipantState(driver=driver_code)
            self.state.participants[driver_code] = participant

        participant.driver = driver_code
        participant.driver_number = safe_int(row.get("DriverNumber"))
        participant.driver_full_name = row.get("DriverFullName") or row.get("Driver")
        participant.team = row.get("Team") or participant.team
        if self.state.year is not None and participant.team:
            participant.constructor_season_id = f"{normalize_team(participant.team)}_{self.state.year}"
        participant.position = safe_int(row.get("Position"))
        participant.last_lap_number = lap_number
        participant.last_lap_time_seconds = safe_float(row.get("LapTimeSeconds"))
        participant.total_race_time_seconds = safe_float(row.get("TimeSeconds"))
        participant.time_seconds = safe_float(row.get("TimeSeconds"))
        participant.compound = row.get("Compound")
        participant.tyre_life = safe_float(row.get("TyreLife"))
        participant.stint = safe_float(row.get("Stint"))
        participant.fresh_tyre = safe_bool(row.get("FreshTyre"))
        participant.is_pit_in_lap = safe_float(row.get("PitInTimeSeconds")) is not None
        participant.is_pit_out_lap = safe_float(row.get("PitOutTimeSeconds")) is not None
        if participant.is_pit_in_lap or participant.is_pit_out_lap:
            if participant.last_pit_lap != lap_number:
                participant.pit_count += 1
                participant.last_pit_lap = lap_number
        participant.laps_since_last_pit = (
            0 if participant.last_pit_lap == lap_number else (lap_number - (participant.last_pit_lap or lap_number))
        )
        participant.gap_to_leader_seconds = safe_float(row.get("GapToLeaderSeconds"))
        participant.interval_to_position_ahead_seconds = safe_float(row.get("IntervalToPositionAheadSeconds"))
        participant.gap_behind_seconds = participant.interval_to_position_ahead_seconds
        participant.sector_1_time_seconds = safe_float(row.get("Sector1TimeSeconds"))
        participant.sector_2_time_seconds = safe_float(row.get("Sector2TimeSeconds"))
        participant.sector_3_time_seconds = safe_float(row.get("Sector3TimeSeconds"))
        participant.speed_i1 = safe_float(row.get("SpeedI1"))
        participant.speed_i2 = safe_float(row.get("SpeedI2"))
        participant.speed_fl = safe_float(row.get("SpeedFL"))
        participant.speed_st = safe_float(row.get("SpeedST"))
        participant.is_active = True

        if participant.last_lap_time_seconds is not None:
            participant.recent_lap_times.append(participant.last_lap_time_seconds)
            participant.recent_lap_times = participant.recent_lap_times[-5:]
            participant.rolling_3_lap_avg = self._average_last_n(participant.recent_lap_times, 3)
            participant.rolling_5_lap_avg = self._average_last_n(participant.recent_lap_times, 5)

    def _average_last_n(self, values: list[float], n: int) -> float | None:
        if not values:
            return None
        window = values[-n:]
        if not window:
            return None
        return sum(window) / len(window)

    def _refresh_classification(self, lap_number: int) -> None:
        ranked_rows = []
        for driver, participant in self.state.participants.items():
            if participant.last_lap_number == lap_number and participant.position is not None:
                ranked_rows.append((participant.position, driver, participant))
        if not ranked_rows:
            return

        ranked_rows.sort(key=lambda item: item[0])
        for _, _, participant in ranked_rows:
            participant.gap_behind_seconds = None

        ordered_entries: list[ClassificationEntry] = []
        for rank_index, (_, driver_code, participant) in enumerate(ranked_rows, start=1):
            previous_position = participant.previous_position
            if previous_position is None:
                previous_position = participant.position
            participant.previous_position = participant.position
            if previous_position is not None and participant.position is not None:
                participant.position_change = previous_position - participant.position
            else:
                participant.position_change = 0
            gap_behind_seconds = None
            if rank_index < len(ranked_rows):
                next_participant = ranked_rows[rank_index][2]
                gap_behind_seconds = next_participant.interval_to_position_ahead_seconds
            participant.gap_behind_seconds = gap_behind_seconds
            entry = ClassificationEntry(
                driver=driver_code,
                team=participant.team,
                position=participant.position,
                rank=rank_index,
                gap_to_leader_seconds=participant.gap_to_leader_seconds,
                interval_to_position_ahead_seconds=participant.interval_to_position_ahead_seconds,
                gap_behind_seconds=gap_behind_seconds,
            )
            ordered_entries.append(entry)
        self.state.classification = ordered_entries

    def _calculate_completion_pct(self) -> float | None:
        if self.state.total_laps_expected is None or self.state.total_laps_expected <= 0:
            return None
        return min(100.0, (self.state.total_laps_seen / self.state.total_laps_expected) * 100.0)

    def commit_lap(self, lap_number: int | float) -> RaceState:
        lap_number = parse_lap_number(lap_number)
        if lap_number is None:
            return self.state
        rows = self.buffer.pop(lap_number, [])
        if not rows:
            return self.state

        self.state.current_lap = lap_number
        self.state.total_laps_seen = max(self.state.total_laps_seen, lap_number)
        self.state.last_committed_lap = lap_number
        self.state.race_completion_pct = self._calculate_completion_pct()
        self._update_current_conditions(rows)

        for row in rows:
            self._apply_row_to_participant(lap_number, row)

        self._refresh_classification(lap_number)
        self._refresh_constructors(lap_number)
        self.state.lap_history.append(
            {
                "lap_number": lap_number,
                "current_conditions": self.state.current_conditions.to_dict(),
                "classification": [entry.to_dict() for entry in self.state.classification],
            }
        )
        validate_state(self.state)
        return self.state

    def _refresh_constructors(self, lap_number: int) -> None:
        team_groups: dict[str, list[ParticipantState]] = {}
        for driver_code, participant in self.state.participants.items():
            cid = participant.constructor_season_id or (
                f"{normalize_team(participant.team)}_{self.state.year}"
                if participant.team and self.state.year
                else (participant.team or "UNKNOWN")
            )
            team_groups.setdefault(cid, []).append(participant)

        constructors: dict[str, ConstructorState] = {}
        for cid, members in team_groups.items():
            sorted_members = sorted(
                members,
                key=lambda p: (
                    0 if (p.position is not None and p.is_active) else 1,
                    p.position if p.position is not None else 999,
                ),
            )
            lead = sorted_members[0] if sorted_members else None
            second = sorted_members[1] if len(sorted_members) > 1 else None

            drivers = [p.driver for p in sorted_members if p.driver is not None]
            team_name = lead.team if lead and lead.team else (members[0].team if members else cid)

            best_pos = lead.position if lead else None
            worst_pos = second.position if second else (lead.position if lead else None)
            lead_gap = lead.gap_to_leader_seconds if lead else None

            intra_gap = None
            if (
                lead is not None
                and second is not None
                and lead.gap_to_leader_seconds is not None
                and second.gap_to_leader_seconds is not None
            ):
                intra_gap = abs(second.gap_to_leader_seconds - lead.gap_to_leader_seconds)
            elif (
                lead is not None
                and second is not None
                and lead.total_race_time_seconds is not None
                and second.total_race_time_seconds is not None
            ):
                intra_gap = abs(second.total_race_time_seconds - lead.total_race_time_seconds)

            total_stops = sum(p.pit_count for p in members)
            is_risk = bool(intra_gap is not None and intra_gap <= 4.5 and len(members) >= 2)

            rolling_paces = [p.rolling_3_lap_avg for p in members if p.rolling_3_lap_avg is not None]
            avg_pace = (sum(rolling_paces) / len(rolling_paces)) if rolling_paces else None
            is_active = any(p.is_active for p in members)

            c_state = ConstructorState(
                constructor_season_id=cid,
                team=team_name or cid,
                drivers=drivers,
                lead_driver=lead.driver if lead else None,
                second_driver=second.driver if second else None,
                best_position=best_pos,
                worst_position=worst_pos,
                lead_gap_to_leader_seconds=lead_gap,
                intra_team_gap_seconds=intra_gap,
                total_pit_stops=total_stops,
                is_double_stack_risk=is_risk,
                rolling_3_lap_avg=avg_pace,
                is_active=is_active,
            )
            constructors[cid] = c_state

        self.state.constructors = constructors

    def flush(self) -> RaceState:
        if self._last_seen_lap is not None:
            self.commit_lap(self._last_seen_lap)
            self._last_seen_lap = None
        for lap_number in sorted(self.buffer):
            self.commit_lap(lap_number)
        self.buffer.clear()
        self._last_seen_lap = None
        return self.state

    def get_current_state(self) -> RaceState:
        return self.state

    def get_classification(self) -> list[ClassificationEntry]:
        return list(self.state.classification)

    def get_participant(self, driver: str) -> ParticipantState | None:
        return self.state.participants.get(normalize_driver(driver))

    def get_constructor(self, constructor_season_id: str) -> ConstructorState | None:
        return self.state.constructors.get(constructor_season_id)

    def get_constructors(self) -> dict[str, ConstructorState]:
        return dict(self.state.constructors)

    def export_constructor(self, constructor_season_id: str) -> dict[str, Any] | None:
        c = self.get_constructor(constructor_season_id)
        if c is None:
            return None
        return c.to_dict()

    def validate_state(self, state: RaceState | None = None) -> None:
        validate_state(state if state is not None else self.state)

    def export_participant(self, driver: str) -> dict[str, Any] | None:
        participant = self.get_participant(driver)
        if participant is None:
            return None
        return participant.to_dict()

    def to_dict(self) -> dict[str, Any]:
        return self.state.to_dict()
