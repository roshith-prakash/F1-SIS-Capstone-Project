"""
src/opponent_model/state.py
===========================
Opponent state vector definitions and builders.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any, Optional

import numpy as np

from .features import (
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    FEATURES_MODEL_C,
    compute_derived_features,
)

try:
    from race_state.models import RaceState
except ImportError:
    from src.race_state.models import RaceState


@dataclass
class OpponentStateVector:
    """
    Complete state vector for a single opponent driver at lap t (Task 29, Phase A2).
    """
    # Identity
    race_id: str
    driver: str
    lap_number: int

    # Observed state (raw)
    remaining_laps: int
    race_progress_fraction: float
    position: int
    tyre_compound: str
    tyre_age: int
    tyre_age_squared: float
    laps_since_last_pit: int
    pit_count: int
    gap_ahead: float
    gap_behind: float
    last_lap_time: float
    rolling_3_lap_avg: float
    is_safety_car: int
    is_vsc: int
    track_temp: float
    air_temp: float
    rainfall: int

    # Foundational model outputs
    predicted_lap_time: float
    predicted_degradation: float
    predicted_lap_time_t1: float
    predicted_degradation_t1: float
    p_sc_h1: float
    p_vsc_h1: float
    p_sc_h3: float

    # Derived strategic features
    pace_delta: float
    deg_rate_acceleration: float
    cost_of_staying: float
    pit_position_cost: float
    gap_ratio: float
    undercut_threat: int
    overcut_window: int
    tyre_life_fraction: float
    laps_past_nominal: int
    position_pressure: int
    sc_adjusted_pit_cost: float

    def to_dict(self) -> dict[str, Any]:
        """Convert state vector to standard dictionary."""
        return asdict(self)

    def to_feature_dict(self, model_version: str = "C") -> dict[str, float]:
        """
        Convert to flat numerical feature dictionary with one-hot encoded compound.
        Supports 'A' (Race State), 'B' (+ Foundational), 'C' (+ Derived).
        """
        d = self.to_dict()

        # One-hot encode compound
        compound_upper = str(self.tyre_compound).strip().upper()
        d["compound_SOFT"] = 1.0 if compound_upper == "SOFT" else 0.0
        d["compound_MEDIUM"] = 1.0 if compound_upper == "MEDIUM" else 0.0
        d["compound_HARD"] = 1.0 if compound_upper == "HARD" else 0.0

        if model_version.upper() == "A":
            cols = FEATURES_MODEL_A
        elif model_version.upper() == "B":
            cols = FEATURES_MODEL_B
        else:
            cols = FEATURES_MODEL_C

        return {c: float(d.get(c, 0.0)) for c in cols}


def build_opponent_state(
    state: RaceState,
    driver: str,
    lt_adapter: Any = None,
    tyre_adapter: Any = None,
    sc_adapter: Any = None,
    sc_models: tuple[Any, Any, list[str], list[str]] | None = None,
    field_median_lap_time: float | None = None,
    cached_sc_probs: dict[str, dict[int, float]] | None = None,
) -> OpponentStateVector | None:
    """
    Build an OpponentStateVector from a live RaceState snapshot and foundational models.
    """
    driver_code = driver.upper().strip()
    participant = state.participants.get(driver_code)
    if not participant or not participant.is_active:
        return None

    current_lap = int(state.current_lap or 1)
    total_laps = int(state.total_laps_expected or 57)
    remaining_laps = max(0, total_laps - current_lap)
    race_progress_fraction = float(current_lap / max(1, total_laps))

    cond = state.current_conditions
    is_sc = 1 if getattr(cond, "has_safety_car", False) else 0
    is_vsc = 1 if getattr(cond, "has_vsc", False) else 0
    track_temp = float(getattr(cond, "track_temp", 30.0) or 30.0)
    air_temp = float(getattr(cond, "air_temp", 25.0) or 25.0)
    rainfall = 1 if getattr(cond, "rainfall", False) else 0

    tyre_compound = str(participant.compound or "MEDIUM").strip().upper()
    tyre_age = int(participant.tyre_life or 0)
    tyre_age_squared = float(tyre_age ** 2)
    laps_since_last_pit = int(participant.laps_since_last_pit or tyre_age)
    pit_count = int(participant.pit_count or 0)

    # Gaps: fill defaults if leading or last
    gap_ahead = float(participant.interval_to_position_ahead_seconds) if participant.interval_to_position_ahead_seconds is not None else 0.0
    gap_behind = float(participant.gap_behind_seconds) if participant.gap_behind_seconds is not None else 50.0

    last_lap_time = float(participant.last_lap_time_seconds or 90.0)
    rolling_3_lap_avg = float(participant.rolling_3_lap_avg or last_lap_time)

    # Foundational Model 1: Lap Time
    predicted_lap_time = 0.0
    predicted_lap_time_t1 = 0.0
    if lt_adapter is not None:
        try:
            pred_lt = lt_adapter.predict_lap_time(state, driver_code)
            predicted_lap_time = float(pred_lt) if pred_lt is not None else rolling_3_lap_avg
        except Exception:
            predicted_lap_time = rolling_3_lap_avg

        # Approximate t+1 by incrementing tyre_life and lap_number
        try:
            orig_life = participant.tyre_life
            participant.tyre_life = (orig_life or 0) + 1
            pred_lt_t1 = lt_adapter.predict_lap_time(state, driver_code)
            predicted_lap_time_t1 = float(pred_lt_t1) if pred_lt_t1 is not None else predicted_lap_time + 0.1
            participant.tyre_life = orig_life
        except Exception:
            predicted_lap_time_t1 = predicted_lap_time + 0.1
    else:
        predicted_lap_time = rolling_3_lap_avg
        predicted_lap_time_t1 = rolling_3_lap_avg + 0.1

    # Foundational Model 2: Tyre Degradation
    predicted_degradation = 0.0
    predicted_degradation_t1 = 0.0
    if tyre_adapter is not None:
        try:
            deg = tyre_adapter.predict_degradation(state, driver_code)
            predicted_degradation = float(deg) if deg is not None else 0.0
        except Exception:
            predicted_degradation = 0.0

        try:
            orig_life = participant.tyre_life
            participant.tyre_life = (orig_life or 0) + 1
            deg_t1 = tyre_adapter.predict_degradation(state, driver_code)
            predicted_degradation_t1 = float(deg_t1) if deg_t1 is not None else predicted_degradation + 0.05
            participant.tyre_life = orig_life
        except Exception:
            predicted_degradation_t1 = predicted_degradation + 0.05

    # Foundational Model 3: SC Risk
    p_sc_h1 = 0.0
    p_vsc_h1 = 0.0
    p_sc_h3 = 0.0
    if cached_sc_probs is not None:
        p_sc_h1 = float(cached_sc_probs.get("SC", {}).get(1, 0.0))
        p_vsc_h1 = float(cached_sc_probs.get("VSC", {}).get(1, 0.0))
        p_sc_h3 = float(cached_sc_probs.get("SC", {}).get(3, 0.0))
    elif sc_adapter is not None and sc_models is not None:
        try:
            logit_sc, logit_vsc, X_sc, X_vsc = sc_models
            H_t = sc_adapter.build_H_t(state)
            import pandas as pd
            X_curr_sc = pd.DataFrame([H_t], columns=X_sc).fillna(0)
            X_curr_vsc = pd.DataFrame([H_t], columns=X_vsc).fillna(0)
            h_t_sc = float(logit_sc.predict_proba(X_curr_sc)[0, 1])
            h_t_vsc = float(logit_vsc.predict_proba(X_curr_vsc)[0, 1])
            p_sc_h1 = float(1.0 - (1.0 - h_t_sc) ** 1)
            p_vsc_h1 = float(1.0 - (1.0 - h_t_vsc) ** 1)
            p_sc_h3 = float(1.0 - (1.0 - h_t_sc) ** 3)
        except Exception:
            p_sc_h1, p_vsc_h1, p_sc_h3 = 0.02, 0.01, 0.05

    # Field median pace
    if field_median_lap_time is None:
        laps = [
            p.last_lap_time_seconds
            for p in state.participants.values()
            if p.is_active and p.last_lap_time_seconds and p.last_lap_time_seconds > 40.0
        ]
        field_median_lap_time = float(np.median(laps)) if laps else last_lap_time

    # Derived Features
    derived = compute_derived_features(
        tyre_compound=tyre_compound,
        tyre_age=tyre_age,
        gap_ahead=gap_ahead,
        gap_behind=gap_behind,
        position=int(participant.position or 10),
        predicted_lap_time=predicted_lap_time,
        field_median_lap_time=field_median_lap_time,
        predicted_degradation=predicted_degradation,
        predicted_degradation_t1=predicted_degradation_t1,
        p_sc_h1=p_sc_h1,
    )

    race_id = str(state.race_id or f"{state.year}_{state.grand_prix}")

    return OpponentStateVector(
        race_id=race_id,
        driver=driver_code,
        lap_number=current_lap,
        remaining_laps=remaining_laps,
        race_progress_fraction=race_progress_fraction,
        position=int(participant.position or 10),
        tyre_compound=tyre_compound,
        tyre_age=tyre_age,
        tyre_age_squared=tyre_age_squared,
        laps_since_last_pit=laps_since_last_pit,
        pit_count=pit_count,
        gap_ahead=gap_ahead,
        gap_behind=gap_behind,
        last_lap_time=last_lap_time,
        rolling_3_lap_avg=rolling_3_lap_avg,
        is_safety_car=is_sc,
        is_vsc=is_vsc,
        track_temp=track_temp,
        air_temp=air_temp,
        rainfall=rainfall,
        predicted_lap_time=predicted_lap_time,
        predicted_degradation=predicted_degradation,
        predicted_lap_time_t1=predicted_lap_time_t1,
        predicted_degradation_t1=predicted_degradation_t1,
        p_sc_h1=p_sc_h1,
        p_vsc_h1=p_vsc_h1,
        p_sc_h3=p_sc_h3,
        pace_delta=derived["pace_delta"],
        deg_rate_acceleration=derived["deg_rate_acceleration"],
        cost_of_staying=derived["cost_of_staying"],
        pit_position_cost=derived["pit_position_cost"],
        gap_ratio=derived["gap_ratio"],
        undercut_threat=int(derived["undercut_threat"]),
        overcut_window=int(derived["overcut_window"]),
        tyre_life_fraction=derived["tyre_life_fraction"],
        laps_past_nominal=int(derived["laps_past_nominal"]),
        position_pressure=int(derived["position_pressure"]),
        sc_adjusted_pit_cost=derived["sc_adjusted_pit_cost"],
    )
