"""
tests/test_overtake_model.py
============================
Unit and integration tests for the Overtake Probability Model and OvertakeAdapter.
"""

import pytest
import numpy as np

from src.overtake.adapter import OvertakeAdapter
from src.overtake.features import (
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    FEATURES_MODEL_C,
    compute_overtake_features,
    resolve_circuit_name,
)
from src.strategy_engine.simulator import RaceScenarioSimulator
from src.strategy_engine.engine import StrategyEngine
from src.strategy_engine.types import Strategy
from src.race_state.models import RaceState, ParticipantState, CurrentConditions


@pytest.fixture
def dummy_race_state():
    participants = {
        "VER": ParticipantState(
            driver="VER",
            driver_number=1,
            team="Red Bull",
            position=1,
            compound="HARD",
            tyre_life=15.0,
            interval_to_position_ahead_seconds=0.0,
            last_lap_time_seconds=86.2,
            rolling_3_lap_avg=86.1,
            pit_count=1,
        ),
        "NOR": ParticipantState(
            driver="NOR",
            driver_number=4,
            team="McLaren",
            position=2,
            compound="MEDIUM",
            tyre_life=5.0,
            interval_to_position_ahead_seconds=0.45,
            last_lap_time_seconds=85.3,
            rolling_3_lap_avg=85.4,
            pit_count=1,
        ),
    }
    conditions = CurrentConditions(
        air_temp=25.0,
        track_temp=35.0,
        rainfall=False,
        has_safety_car=False,
        has_vsc=False,
    )
    return RaceState(
        year=2024,
        grand_prix="Italian Grand Prix",
        location="Monza",
        current_lap=30,
        total_laps_expected=53,
        current_conditions=conditions,
        participants=participants,
    )


def test_circuit_resolution():
    assert resolve_circuit_name("monza") == "Italy"
    assert resolve_circuit_name("silverstone") == "Great_Britain"
    assert resolve_circuit_name("albert park") == "Australia"
    assert resolve_circuit_name("monte carlo") == "Monaco"
    assert resolve_circuit_name("unknown_track") == "Bahrain"


def test_feature_sets_schema():
    assert len(FEATURES_MODEL_A) == 6
    assert len(FEATURES_MODEL_B) == 11
    assert len(FEATURES_MODEL_C) == 20
    assert set(FEATURES_MODEL_A).issubset(set(FEATURES_MODEL_B))
    assert set(FEATURES_MODEL_B).issubset(set(FEATURES_MODEL_C))


def test_compute_overtake_features_types():
    feats = compute_overtake_features(
        gap_seconds=0.65,
        pace_delta=0.40,
        tyre_age_delta=8.0,
        compound_behind="SOFT",
        compound_ahead="HARD",
        is_fresh_tyre_behind=True,
        circuit="Monza",
    )
    assert isinstance(feats, dict)
    for col in FEATURES_MODEL_C:
        assert col in feats, f"Missing feature {col}"
        assert isinstance(feats[col], (int, float, np.floating))


def test_overtake_adapter_probability_validity():
    adapter = OvertakeAdapter()
    p = adapter.predict_overtake_probability(
        gap_seconds=0.50,
        pace_delta=0.60,
        tyre_age_delta=6.0,
        compound_behind="MEDIUM",
        compound_ahead="HARD",
        is_fresh_tyre_behind=False,
        circuit="Monza",
    )
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_overtake_safety_car_suppression():
    adapter = OvertakeAdapter()
    p = adapter.predict_overtake_probability(
        gap_seconds=0.20,
        pace_delta=2.0,
        tyre_age_delta=20.0,
        compound_behind="SOFT",
        compound_ahead="HARD",
        is_fresh_tyre_behind=True,
        circuit="Monza",
        is_sc=True,
    )
    assert p == 0.0, "Overtaking must be strictly impossible under Safety Car"


def test_overtake_monotonicity_gap_and_pace():
    adapter = OvertakeAdapter()
    # Close gap vs far gap with identical advantage
    p_close = adapter.predict_overtake_probability(
        gap_seconds=0.35,
        pace_delta=0.75,
        tyre_age_delta=5.0,
        compound_behind="MEDIUM",
        compound_ahead="HARD",
        is_fresh_tyre_behind=False,
        circuit="Monza",
    )
    p_far = adapter.predict_overtake_probability(
        gap_seconds=1.40,
        pace_delta=0.75,
        tyre_age_delta=5.0,
        compound_behind="MEDIUM",
        compound_ahead="HARD",
        is_fresh_tyre_behind=False,
        circuit="Monza",
    )
    assert p_close > p_far, "Closer gap should yield higher pass probability"

    # Higher pace delta vs lower pace delta with identical gap
    p_faster = adapter.predict_overtake_probability(
        gap_seconds=0.50,
        pace_delta=1.20,
        tyre_age_delta=10.0,
        compound_behind="SOFT",
        compound_ahead="HARD",
        is_fresh_tyre_behind=True,
        circuit="Monza",
    )
    p_slower = adapter.predict_overtake_probability(
        gap_seconds=0.50,
        pace_delta=-0.40,
        tyre_age_delta=-5.0,
        compound_behind="HARD",
        compound_ahead="SOFT",
        is_fresh_tyre_behind=False,
        circuit="Monza",
    )
    assert p_faster > p_slower, "Significant pace advantage should yield higher pass probability"


def test_dirty_air_penalty_bounds():
    adapter = OvertakeAdapter()
    # Outside dirty air threshold
    pen_clear = adapter.predict_dirty_air_penalty(gap_seconds=1.8, circuit="Monza")
    assert pen_clear == 0.0

    # Inside dirty air threshold
    pen_close = adapter.predict_dirty_air_penalty(gap_seconds=0.4, circuit="Monza")
    assert 0.05 <= pen_close <= 0.65

    # Track difficulty sensitivity (Monaco vs Monza)
    pen_monaco = adapter.predict_dirty_air_penalty(gap_seconds=0.4, circuit="Monaco")
    pen_monza = adapter.predict_dirty_air_penalty(gap_seconds=0.4, circuit="Monza")
    assert pen_monaco >= pen_monza


def test_overtake_adapter_fallback():
    # Force empty model and calibrator to test heuristic fallback
    adapter = OvertakeAdapter(model=None, calibrator=None)
    adapter.model = None
    adapter.calibrator = None

    p_fall = adapter.predict_overtake_probability(
        gap_seconds=0.6,
        pace_delta=0.9,
        tyre_age_delta=4.0,
        compound_behind="SOFT",
        compound_ahead="MEDIUM",
        is_fresh_tyre_behind=True,
        circuit="Silverstone",
    )
    assert 0.0 <= p_fall <= 1.0

    pen_fall = adapter.predict_dirty_air_penalty(gap_seconds=0.5, circuit="Silverstone")
    assert 0.05 <= pen_fall <= 0.65


def test_simulator_integration_with_overtake_adapter(dummy_race_state):
    adapter = OvertakeAdapter()
    simulator = RaceScenarioSimulator(
        overtake_adapter=adapter,
        default_rollouts=5,
        random_seed=42,
    )
    strat = Strategy(
        strategy_id="1-stop",
        name="1-Stop Medium-Hard",
        num_stops=1,
        pit_laps=[35],
        compounds=["HARD"],
    )
    res = simulator.simulate_strategy(
        state=dummy_race_state,
        ego_driver="NOR",
        strategy=strat,
        horizon_laps=10,
        n_rollouts=5,
    )
    assert res.n_rollouts == 5
    assert len(res.rollouts) == 5
    for out in res.rollouts:
        assert out.final_position in [1, 2]


def test_engine_load_with_default_models():
    engine = StrategyEngine.load_with_default_models()
    assert hasattr(engine.simulator, "overtake_adapter")
    assert engine.simulator.overtake_adapter is not None
