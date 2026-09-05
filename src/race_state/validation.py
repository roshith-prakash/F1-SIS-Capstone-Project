from __future__ import annotations

from dataclasses import fields
from typing import Any

from .models import ConstructorState, ParticipantState, RaceState

REQUIRED_PARTICIPANT_FIELDS = {field.name for field in fields(ParticipantState)}
REQUIRED_CONSTRUCTOR_FIELDS = {field.name for field in fields(ConstructorState)}
REQUIRED_RACE_STATE_FIELDS = {field.name for field in fields(RaceState)}


def validate_race_state(state: RaceState) -> None:
    """Raise ValueError if a committed race snapshot is internally incomplete."""
    missing_state_fields = [
        field_name
        for field_name in REQUIRED_RACE_STATE_FIELDS
        if not hasattr(state, field_name)
    ]
    if missing_state_fields:
        raise ValueError(f"RaceState is missing fields: {missing_state_fields}")

    if not isinstance(state.participants, dict):
        raise ValueError("RaceState.participants must be a dictionary keyed by driver code")

    if not isinstance(state.constructors, dict):
        raise ValueError("RaceState.constructors must be a dictionary keyed by constructor_season_id")

    if not isinstance(state.classification, list):
        raise ValueError("RaceState.classification must be a list")

    for driver, participant in state.participants.items():
        _validate_participant(driver, participant)

    for cid, constructor in state.constructors.items():
        _validate_constructor(cid, constructor)

    classified_drivers = {entry.driver for entry in state.classification}
    unknown_classified_drivers = classified_drivers - set(state.participants)
    if unknown_classified_drivers:
        raise ValueError(
            "classification contains drivers missing from participants: "
            f"{sorted(unknown_classified_drivers)}"
        )


def _validate_participant(driver: str, participant: Any) -> None:
    missing_fields = [
        field_name
        for field_name in REQUIRED_PARTICIPANT_FIELDS
        if not hasattr(participant, field_name)
    ]
    if missing_fields:
        raise ValueError(f"{driver} is missing participant fields: {missing_fields}")

    if participant.driver is None:
        raise ValueError(f"{driver} has no driver code")

    if participant.driver != driver:
        raise ValueError(f"participant key {driver} does not match driver {participant.driver}")

    if not isinstance(participant.recent_lap_times, list):
        raise ValueError(f"{driver} recent_lap_times must be a list")

    if len(participant.recent_lap_times) > 5:
        raise ValueError(f"{driver} recent_lap_times must keep at most 5 laps")

    if participant.pit_count < 0:
        raise ValueError(f"{driver} pit_count cannot be negative")


def _validate_constructor(cid: str, constructor: Any) -> None:
    missing_fields = [
        field_name
        for field_name in REQUIRED_CONSTRUCTOR_FIELDS
        if not hasattr(constructor, field_name)
    ]
    if missing_fields:
        raise ValueError(f"{cid} is missing constructor fields: {missing_fields}")

    if constructor.constructor_season_id != cid:
        raise ValueError(f"constructor key {cid} does not match constructor_season_id {constructor.constructor_season_id}")

    if not isinstance(constructor.drivers, list):
        raise ValueError(f"{cid} drivers must be a list")
