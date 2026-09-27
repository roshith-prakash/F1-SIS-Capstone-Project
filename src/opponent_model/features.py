"""
src/opponent_model/features.py
==============================
Feature definitions, schemas, and helper functions for Opponent Modeling.
Categorized into:
- Category A: Current Race State
- Category B: Opponent Physical/Performance State
- Category C: Our Car's (Ego) State
- Opponent History (H_t)
- Derived Strategic Features
"""

from __future__ import annotations

import numpy as np

# Nominal stint lengths by dry compound
COMPOUND_NOMINAL_STINTS: dict[str, int] = {
    "SOFT": 18,
    "MEDIUM": 28,
    "HARD": 38,
}
DEFAULT_NOMINAL_STINT: int = 28
TYPICAL_PIT_LOSS_SECONDS: float = 22.0

# -------------------------------------------------------------------------
# Feature Group Definitions
# -------------------------------------------------------------------------

# Category A: Current Race State Features
FEATURES_RACE_STATE = [
    "lap_number",
    "remaining_laps",
    "race_progress_fraction",
    "position",
    "gap_ahead",
    "gap_behind",
    "is_safety_car",
    "is_vsc",
    "track_temp",
    "air_temp",
    "rainfall",
    "pit_loss_seconds",
]

# Category B: Opponent Physical & Performance State Features
FEATURES_OPPONENT_PHYSICAL = [
    "compound_SOFT",
    "compound_MEDIUM",
    "compound_HARD",
    "tyre_age",
    "tyre_age_squared",
    "last_lap_time",
    "rolling_3_lap_avg",
    "predicted_lap_time",
    "predicted_degradation",
    "predicted_lap_time_t1",
    "predicted_degradation_t1",
    "p_sc_h1",
    "p_vsc_h1",
    "p_sc_h3",
]

# Category C: Our Car's (Ego) State Features
FEATURES_EGO_STATE = [
    "ego_position",
    "gap_to_ego",
    "ego_tyre_age",
    "ego_predicted_pace",
    "ego_predicted_deg",
    "ego_recently_pitted",
    "ego_undercut_threat",
]

# Opponent Temporal History (H_t)
FEATURES_OPPONENT_HISTORY = [
    "laps_since_last_pit",
    "pit_count",
    "recent_pace_trend",
    "recent_deg_trend",
    "previous_action",
]

# Derived Strategic Features
FEATURES_DERIVED_STRATEGIC = [
    "pace_delta",
    "deg_rate_acceleration",
    "cost_of_staying",
    "pit_position_cost",
    "gap_ratio",
    "undercut_threat",
    "overcut_window",
    "tyre_life_fraction",
    "laps_past_nominal",
    "position_pressure",
    "sc_adjusted_pit_cost",
]

# -------------------------------------------------------------------------
# Standard Model Feature Sets for Ablation Studies
# -------------------------------------------------------------------------

# Model A: Race State Only
FEATURES_MODEL_A = [
    "lap_number",
    "remaining_laps",
    "race_progress_fraction",
    "position",
    "compound_SOFT",
    "compound_MEDIUM",
    "compound_HARD",
    "tyre_age",
    "tyre_age_squared",
    "laps_since_last_pit",
    "pit_count",
    "gap_ahead",
    "gap_behind",
    "last_lap_time",
    "rolling_3_lap_avg",
    "is_safety_car",
    "is_vsc",
    "track_temp",
    "air_temp",
    "rainfall",
]

# Model B: Race State + Foundational Models
FEATURES_MODEL_B = FEATURES_MODEL_A + [
    "predicted_lap_time",
    "predicted_degradation",
    "predicted_lap_time_t1",
    "predicted_degradation_t1",
    "p_sc_h1",
    "p_vsc_h1",
    "p_sc_h3",
]

# Model C: Race State + Foundational Models + Derived Strategic Features (Production baseline)
FEATURES_MODEL_C = FEATURES_MODEL_B + [
    "pace_delta",
    "deg_rate_acceleration",
    "cost_of_staying",
    "pit_position_cost",
    "gap_ratio",
    "undercut_threat",
    "overcut_window",
    "tyre_life_fraction",
    "laps_past_nominal",
    "position_pressure",
    "sc_adjusted_pit_cost",
]

# Model D (Extended SIS): Model C + Ego State + Opponent History
FEATURES_MODEL_EXTENDED = FEATURES_MODEL_C + FEATURES_EGO_STATE + [
    "recent_pace_trend",
    "recent_deg_trend",
    "previous_action",
]


def get_compound_nominal_stint(compound: str | None) -> int:
    """Return nominal stint length in laps for a given tyre compound."""
    if not compound:
        return DEFAULT_NOMINAL_STINT
    clean_compound = str(compound).strip().upper()
    return COMPOUND_NOMINAL_STINTS.get(clean_compound, DEFAULT_NOMINAL_STINT)


def compute_derived_features(
    tyre_compound: str,
    tyre_age: float,
    gap_ahead: float,
    gap_behind: float,
    position: int,
    predicted_lap_time: float,
    field_median_lap_time: float,
    predicted_degradation: float,
    predicted_degradation_t1: float,
    p_sc_h1: float,
    pit_loss: float = TYPICAL_PIT_LOSS_SECONDS,
    gap_to_ego: float = 0.0,
    ego_tyre_age: float = 0.0,
    ego_recently_pitted: int = 0,
) -> dict[str, float]:
    """
    Compute derived strategic features including competitive pressure relative to Ego.
    """
    nominal = get_compound_nominal_stint(tyre_compound)
    remaining_stint_laps = max(1.0, float(nominal - tyre_age))

    # Gap ratio and tactical flags
    safe_gap_ahead = max(0.0, float(gap_ahead))
    safe_gap_behind = max(0.0, float(gap_behind))
    gap_sum = safe_gap_ahead + safe_gap_behind + 1e-5
    gap_ratio = float(safe_gap_ahead / gap_sum)

    undercut_threat = 1.0 if safe_gap_behind < 2.0 else 0.0
    overcut_window = 1.0 if safe_gap_ahead > 5.0 else 0.0

    # Ego-relative undercut pressure:
    # If Ego is behind opponent within 2.5s and recently pitted or on fresher tyres
    ego_undercut_threat = 0.0
    if gap_to_ego < 0 and abs(gap_to_ego) < 2.5: # Ego is behind opponent
        if ego_recently_pitted == 1 or ego_tyre_age < tyre_age - 5:
            ego_undercut_threat = 1.0

    # Pace & degradation derivatives
    pace_delta = float(predicted_lap_time - field_median_lap_time) if field_median_lap_time > 0 else 0.0
    deg_rate_acceleration = float(predicted_degradation_t1 - predicted_degradation)
    cost_of_staying = float(predicted_degradation * remaining_stint_laps)
    pit_position_cost = float(safe_gap_ahead - pit_loss)

    # Tyre life wear metrics
    tyre_life_fraction = float(tyre_age / max(1.0, float(nominal)))
    laps_past_nominal = float(max(0.0, tyre_age - nominal))
    position_pressure = float(max(0, 10 - position))
    sc_adjusted_pit_cost = float(safe_gap_ahead * (1.0 - p_sc_h1))

    return {
        "pace_delta": pace_delta,
        "deg_rate_acceleration": deg_rate_acceleration,
        "cost_of_staying": cost_of_staying,
        "pit_position_cost": pit_position_cost,
        "gap_ratio": gap_ratio,
        "undercut_threat": undercut_threat,
        "overcut_window": overcut_window,
        "tyre_life_fraction": tyre_life_fraction,
        "laps_past_nominal": laps_past_nominal,
        "position_pressure": position_pressure,
        "sc_adjusted_pit_cost": sc_adjusted_pit_cost,
        "ego_undercut_threat": ego_undercut_threat,
    }
