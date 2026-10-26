"""Tests for user tactical override and strategy adoption mechanics."""
import os
import pytest
from src.api.session import get_session, reset_session
from src.api.engine import evaluate_lap
from src.api.routers.race import setup_race, set_override
from src.api.schemas import SetupRequest, OverrideRequest

pytestmark = pytest.mark.anyio
RACE_FILE = "2024/Italian_Grand_Prix.csv"

@pytest.fixture(autouse=True)
def clean_session():
    reset_session()
    yield
    reset_session()

async def test_arm_and_trigger_pit_override():
    setup_race(SetupRequest(
        race_file=RACE_FILE,
        driver="VER",
        profile="balanced",
        rollouts=15,
        horizon_laps=8,
    ))
    session = get_session()

    # Evaluate lap 1
    lap1 = await evaluate_lap(session, 1)
    session.current_lap = 1
    ego1 = lap1["participants"]["VER"]
    initial_compound = ego1["compound"]

    # Arm a tactical pit stop for next lap (Lap 2) with alternate compound
    target_compound = "HARD" if initial_compound != "HARD" else "MEDIUM"
    arm_res = await set_override(OverrideRequest(action="pit_next_lap", compound=target_compound))
    assert arm_res["active_override"] is not None
    assert arm_res["active_override"]["action"] == "pit_next_lap"
    assert arm_res["active_override"]["target_lap"] == 2
    assert arm_res["active_override"]["compound"] == target_compound

    # Evaluate lap 2: Should trigger the pit stop override
    lap2 = await evaluate_lap(session, 2)
    session.current_lap = 2
    assert lap2["participants"]["VER"]["compound"] == target_compound
    assert lap2["participants"]["VER"]["tyre_life"] == 0.0
    assert lap2["participants"]["VER"]["is_pit_in_lap"] is True
    # The active override should now be fulfilled and cleared
    assert session.active_override is None

    # Evaluate lap 3: The persistent ego override should keep the target compound and advance tyre life
    lap3 = await evaluate_lap(session, 3)
    session.current_lap = 3
    assert lap3["participants"]["VER"]["compound"] == target_compound
    assert lap3["participants"]["VER"]["tyre_life"] == 1.0

async def test_adopt_alternate_strategy():
    setup_race(SetupRequest(
        race_file=RACE_FILE,
        driver="VER",
        profile="balanced",
        rollouts=15,
        horizon_laps=8,
    ))
    session = get_session()

    lap1 = await evaluate_lap(session, 1)
    session.current_lap = 1
    strategies = lap1.get("strategies", [])
    assert len(strategies) >= 2, "Expected at least 2 candidate strategies"

    alt_strategy = strategies[1]
    alt_id = alt_strategy["id"]

    # Adopt alternate strategy
    adopt_res = await set_override(OverrideRequest(action="adopt_strategy", strategy_id=alt_id))
    assert adopt_res["strategy_id"] == alt_id
    assert session.adopted_strategy_id == alt_id

    # Evaluate lap 2 with adopted strategy active
    lap2 = await evaluate_lap(session, 2)
    session.current_lap = 2
    strategies_lap2 = lap2.get("strategies", [])
    assert len(strategies_lap2) >= 1
    # The adopted strategy should now be ranked #1
    top_strat = strategies_lap2[0]
    assert top_strat["id"] == alt_id
    assert top_strat.get("is_user_adopted") is True

    # Clear adoption
    clear_res = await set_override(OverrideRequest(action="clear"))
    assert clear_res["adopted_strategy_id"] is None
    assert session.adopted_strategy_id is None
