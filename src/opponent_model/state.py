"""
src/opponent_model/state.py
===========================
Opponent state vector definitions and builders conforming to F1 SIS architecture.
Encapsulates:
- Category A: Current Race State
- Category B: Opponent Physical/Performance State
- Category C: Our Car's (Ego) State
- Opponent Temporal History (H_t)
- Derived Strategic Features
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
    FEATURES_MODEL_EXTENDED,
    compute_derived_features,
)

try:
    from race_state.models import RaceState
except ImportError:
    from src.race_state.models import RaceState


@dataclass
class OpponentStateVector:
    """
    Complete state vector S_t for a single opponent driver at lap t.
    Grounded in F1 SIS architecture with 3 explicit input categories,
    opponent temporal history, and derived strategic signals.
    """
    # Identity
    race_id: str
    driver: str
    lap_number: int

    # Category A: Current Race State
    remaining_laps: int
    race_progress_fraction: float
    position: int
    gap_ahead: float
    gap_behind: float
    is_safety_car: int
    is_vsc: int
    track_temp: float
    air_temp: float
    rainfall: int
    pit_loss_seconds: float = 22.0
    race_phase: str = "mid"  # "early", "mid", "late"

    # Category B: Opponent Physical & Performance State
    tyre_compound: str = "MEDIUM"
    tyre_age: int = 0
    tyre_age_squared: float = 0.0
    last_lap_time: float = 90.0
    rolling_3_lap_avg: float = 90.0
    predicted_lap_time: float = 90.0
    predicted_degradation: float = 0.0
    predicted_lap_time_t1: float = 90.1
    predicted_degradation_t1: float = 0.05
    p_sc_h1: float = 0.02
    p_vsc_h1: float = 0.01
    p_sc_h3: float = 0.05

    # Category C: Our Car's (Ego) State
    ego_driver: str = ""
    ego_position: int = 0
    gap_to_ego: float = 0.0
    ego_compound: str = "MEDIUM"
    ego_tyre_age: int = 0
    ego_predicted_pace: float = 90.0
    ego_predicted_deg: float = 0.0
    ego_recently_pitted: int = 0
    ego_undercut_threat: int = 0

    # Opponent Temporal History (H_t)
    laps_since_last_pit: int = 0
    pit_count: int = 0
    recent_pace_trend: float = 0.0
    recent_deg_trend: float = 0.0
    previous_action: int = 0  # 0=STAY, 1=PIT

    # Derived Strategic Features
    pace_delta: float = 0.0
    deg_rate_acceleration: float = 0.0
    cost_of_staying: float = 0.0
    pit_position_cost: float = 0.0
    gap_ratio: float = 0.5
    undercut_threat: int = 0
    overcut_window: int = 0
    tyre_life_fraction: float = 0.0
    laps_past_nominal: int = 0
    position_pressure: int = 0
    sc_adjusted_pit_cost: float = 0.0

    # -------------------------------------------------------------------------
    # Explicit Semantic Properties (Tasks 3 & 4)
    # -------------------------------------------------------------------------
    @property
    def opponent_predicted_lap_time(self) -> float:
        """Explicit semantic alias for opponent's predicted lap time."""
        return self.predicted_lap_time

    @property
    def opponent_predicted_degradation(self) -> float:
        """Explicit semantic alias for opponent's predicted tyre degradation."""
        return self.predicted_degradation

    @property
    def opponent_tyre_age(self) -> int:
        """Explicit semantic alias for opponent's tyre age."""
        return self.tyre_age

    @property
    def opponent_compound(self) -> str:
        """Explicit semantic alias for opponent's tyre compound."""
        return self.tyre_compound

    @property
    def ego_predicted_lap_time(self) -> float:
        """Explicit semantic alias for ego car's predicted lap time."""
        return self.ego_predicted_pace

    @property
    def ego_predicted_degradation(self) -> float:
        """Explicit semantic alias for ego car's predicted tyre degradation."""
        return self.ego_predicted_deg

    @property
    def gap_opponent_to_ego(self) -> float:
        """Explicit semantic alias for gap from opponent to ego car in seconds."""
        return self.gap_to_ego

    def to_dict(self) -> dict[str, Any]:
        """Convert state vector to standard dictionary with explicit semantic aliases."""
        d = asdict(self)
        d["opponent_predicted_lap_time"] = self.predicted_lap_time
        d["opponent_predicted_degradation"] = self.predicted_degradation
        d["opponent_tyre_age"] = self.tyre_age
        d["opponent_compound"] = self.tyre_compound
        d["ego_predicted_lap_time"] = self.ego_predicted_pace
        d["ego_predicted_degradation"] = self.ego_predicted_deg
        d["gap_opponent_to_ego"] = self.gap_to_ego
        return d

    def __iter__(self):
        """Allow dict(vector) conversion for compatibility."""
        return iter(self.to_dict().items())

    def to_feature_dict(
        self,
        model_version: str = "C",
        feature_names: list[str] | None = None,
    ) -> dict[str, float]:
        """
        Convert to flat numerical feature dictionary with one-hot encoded compound.
        Supports 'A', 'B', 'C', 'EXTENDED', or custom feature_names list.
        """
        d = self.to_dict()

        # One-hot encode compound
        compound_upper = str(self.tyre_compound).strip().upper()
        d["compound_SOFT"] = 1.0 if compound_upper == "SOFT" else 0.0
        d["compound_MEDIUM"] = 1.0 if compound_upper == "MEDIUM" else 0.0
        d["compound_HARD"] = 1.0 if compound_upper == "HARD" else 0.0

        if feature_names is not None:
            cols = feature_names
        elif model_version.upper() == "A":
            cols = FEATURES_MODEL_A
        elif model_version.upper() == "B":
            cols = FEATURES_MODEL_B
        elif model_version.upper() == "EXTENDED":
            cols = FEATURES_MODEL_EXTENDED
        else:
            cols = FEATURES_MODEL_C

        return {c: float(d.get(c, 0.0)) for c in cols}


def build_opponent_state(
    state: RaceState,
    driver: str,
    ego_driver: str | None = None,
    lt_adapter: Any = None,
    tyre_adapter: Any = None,
    sc_adapter: Any = None,
    sc_models: tuple[Any, Any, list[str], list[str]] | None = None,
    field_median_lap_time: float | None = None,
    cached_sc_probs: dict[str, dict[int, float]] | None = None,
    pit_loss_seconds: float = 22.0,
) -> OpponentStateVector | None:
    """
    Build an OpponentStateVector from a live RaceState snapshot, foundational models,
    and Ego car state.
    """
    driver_code = driver.upper().strip()
    participant = state.participants.get(driver_code)
    if not participant or not participant.is_active:
        return None

    current_lap = int(state.current_lap or 1)
    total_laps = int(state.total_laps_expected or 57)
    remaining_laps = max(0, total_laps - current_lap)
    race_progress_fraction = float(current_lap / max(1, total_laps))

    # Determine race phase
    if race_progress_fraction < 0.33:
        race_phase = "early"
    elif race_progress_fraction < 0.67:
        race_phase = "mid"
    else:
        race_phase = "late"

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

    # Opponent history trends
    recent_pace_trend = float(rolling_3_lap_avg - last_lap_time)  # positive = getting slower
    previous_action = 1 if laps_since_last_pit == 1 else 0

    # -------------------------------------------------------------------------
    # Category C: Our Car's (Ego) State Resolution
    # -------------------------------------------------------------------------
    ego_code = ""
    ego_participant = None
    if ego_driver:
        ego_code = ego_driver.upper().strip()
        ego_participant = state.participants.get(ego_code)

    # If no ego_driver specified or invalid, pick the nearest competitor
    if ego_participant is None:
        opp_pos = int(participant.position or 10)
        closest_p = None
        min_pos_diff = 999
        for p_code, p in state.participants.items():
            if p_code != driver_code and p.is_active and p.position is not None:
                diff = abs(int(p.position) - opp_pos)
                if diff < min_pos_diff:
                    min_pos_diff = diff
                    closest_p = p
                    ego_code = p_code
        ego_participant = closest_p

    ego_position = int(ego_participant.position or 0) if ego_participant else 0
    ego_compound = str(ego_participant.compound or "MEDIUM").strip().upper() if ego_participant else "MEDIUM"
    ego_tyre_age = int(ego_participant.tyre_life or 0) if ego_participant else 0
    ego_laps_since_pit = int(ego_participant.laps_since_last_pit or ego_tyre_age) if ego_participant else 0
    ego_recently_pitted = 1 if ego_laps_since_pit <= 2 else 0

    # Calculate gap to ego (in seconds)
    gap_to_ego = 0.0
    if ego_participant and participant.position is not None and ego_participant.position is not None:
        if participant.position < ego_participant.position:
            # Opponent is ahead of Ego: gap is positive
            gap_to_ego = float(gap_behind) if (int(ego_participant.position) == int(participant.position) + 1) else float(abs(participant.position - ego_participant.position) * 2.0)
        elif participant.position > ego_participant.position:
            # Opponent is behind Ego: gap is negative
            gap_to_ego = -float(gap_ahead) if (int(ego_participant.position) == int(participant.position) - 1) else -float(abs(participant.position - ego_participant.position) * 2.0)

    # -------------------------------------------------------------------------
    # Foundational Models Integration
    # -------------------------------------------------------------------------
    # 1. Lap-Time Model
    predicted_lap_time = 0.0
    predicted_lap_time_t1 = 0.0
    if lt_adapter is not None:
        try:
            pred_lt = lt_adapter.predict_lap_time(state, driver_code)
            predicted_lap_time = float(pred_lt) if pred_lt is not None else rolling_3_lap_avg
        except Exception:
            predicted_lap_time = rolling_3_lap_avg

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

    # Ego predicted pace
    ego_predicted_pace = 90.0
    if lt_adapter is not None and ego_code:
        try:
            p_lt = lt_adapter.predict_lap_time(state, ego_code)
            ego_predicted_pace = float(p_lt) if p_lt is not None else 90.0
        except Exception:
            ego_predicted_pace = 90.0

    # 2. Tyre Degradation Model
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

    # Ego predicted deg
    ego_predicted_deg = 0.0
    if tyre_adapter is not None and ego_code:
        try:
            deg = tyre_adapter.predict_degradation(state, ego_code)
            ego_predicted_deg = float(deg) if deg is not None else 0.0
        except Exception:
            ego_predicted_deg = 0.0

    # 3. SC Risk Model
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
        pit_loss=pit_loss_seconds,
        gap_to_ego=gap_to_ego,
        ego_tyre_age=ego_tyre_age,
        ego_recently_pitted=ego_recently_pitted,
    )

    race_id = str(state.race_id or f"{state.year}_{state.grand_prix}")

    return OpponentStateVector(
        race_id=race_id,
        driver=driver_code,
        lap_number=current_lap,
        remaining_laps=remaining_laps,
        race_progress_fraction=race_progress_fraction,
        position=int(participant.position or 10),
        gap_ahead=gap_ahead,
        gap_behind=gap_behind,
        is_safety_car=is_sc,
        is_vsc=is_vsc,
        track_temp=track_temp,
        air_temp=air_temp,
        rainfall=rainfall,
        pit_loss_seconds=pit_loss_seconds,
        race_phase=race_phase,
        tyre_compound=tyre_compound,
        tyre_age=tyre_age,
        tyre_age_squared=tyre_age_squared,
        last_lap_time=last_lap_time,
        rolling_3_lap_avg=rolling_3_lap_avg,
        predicted_lap_time=predicted_lap_time,
        predicted_degradation=predicted_degradation,
        predicted_lap_time_t1=predicted_lap_time_t1,
        predicted_degradation_t1=predicted_degradation_t1,
        p_sc_h1=p_sc_h1,
        p_vsc_h1=p_vsc_h1,
        p_sc_h3=p_sc_h3,
        ego_driver=ego_code,
        ego_position=ego_position,
        gap_to_ego=gap_to_ego,
        ego_compound=ego_compound,
        ego_tyre_age=ego_tyre_age,
        ego_predicted_pace=ego_predicted_pace,
        ego_predicted_deg=ego_predicted_deg,
        ego_recently_pitted=ego_recently_pitted,
        ego_undercut_threat=int(derived.get("ego_undercut_threat", 0)),
        laps_since_last_pit=laps_since_last_pit,
        pit_count=pit_count,
        recent_pace_trend=recent_pace_trend,
        recent_deg_trend=derived["deg_rate_acceleration"],
        previous_action=previous_action,
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
