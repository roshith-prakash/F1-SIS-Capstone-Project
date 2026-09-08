from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import xgboost as xgb

try:
    from race_state.models import ParticipantState, RaceState, normalize_driver
except ImportError:
    from src.race_state.models import ParticipantState, RaceState, normalize_driver


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_PATHS = [
    PROJECT_ROOT / "outputs" / "models" / "xgboost_baseline.json",
    PROJECT_ROOT / "models" / "xgboost_baseline.json",
    PROJECT_ROOT / "models" / "Tyre Degradation" / "xgboost_baseline.json",
    Path("outputs") / "models" / "xgboost_baseline.json",
]

COMPOUND_MAP: dict[str, int] = {
    "SOFT": 1,
    "MEDIUM": 2,
    "HARD": 3,
    "INTERMEDIATE": -1,
    "WET": -2,
    "TEST_UNKNOWN": 0,
}

FEATURE_COLS = [
    "TyreAge",
    "Log_TyreAge",
    "StintLap",
    "Compound_Encoded",
    "RaceProgressFraction",
    "TrackTemp",
    "AirTemp",
    "is_safety_car",
    "is_vsc",
    "Lag1_Pace_Residual",
    "Rolling3_Degradation_Trend",
    "Team_Median_Pace_Lag1",
]


class TyreDegAdapter:
    """
    Adapter bridging committed ``RaceState`` snapshots to the 12-feature
    Tyre Degradation XGBoost model trained in ``tyre_deg _model.ipynb``.
    """

    def __init__(
        self,
        model: Any | None = None,
        model_path: str | Path | None = None,
        circuit_base_pace: float | None = None,
    ) -> None:
        self.feature_cols = list(FEATURE_COLS)
        self.circuit_base_pace = circuit_base_pace or 85.0

        # In-memory stint history tracker: driver -> list of pace residuals in current stint
        self._driver_residuals: dict[str, list[float]] = {}
        self._driver_stints: dict[str, int] = {}

        if model is not None:
            self.model = model
        else:
            loaded_model = None
            candidate_paths = [Path(model_path)] if model_path else DEFAULT_MODEL_PATHS
            for p in candidate_paths:
                if p and p.exists():
                    try:
                        m = xgb.XGBRegressor()
                        m.load_model(str(p))
                        loaded_model = m
                        break
                    except Exception:
                        continue
            self.model = loaded_model

    def _calc_slope(self, values: list[float]) -> float:
        """Calculate linear slope across recent residuals."""
        clean = [v for v in values if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if len(clean) < 3:
            return 0.0
        x = np.arange(len(clean))
        try:
            return float(np.polyfit(x, clean, 1)[0])
        except Exception:
            return 0.0

    def build_features(
        self,
        state: RaceState,
        participant: ParticipantState,
    ) -> dict[str, Any]:
        """
        Build the 12 causal features matching the tyre degradation model schema
        directly from the RaceState snapshot.
        """
        # 1. Tyre age & log transform
        tl = participant.tyre_life
        tyre_age = float(tl) if tl is not None and not (isinstance(tl, float) and math.isnan(tl)) else 1.0
        tyre_age = max(1.0, tyre_age)
        log_tyre_age = float(np.log1p(tyre_age))

        # 2. Stint lap (progress within current stint)
        laps_since_pit = participant.laps_since_last_pit
        if laps_since_pit is not None and not (isinstance(laps_since_pit, float) and math.isnan(laps_since_pit)):
            stint_lap = float(laps_since_pit + 1)
        else:
            stint_lap = float(tyre_age)

        # 3. Compound ordinal encoding
        comp_str = str(participant.compound or "MEDIUM").strip().upper()
        compound_encoded = COMPOUND_MAP.get(comp_str, 2)

        # 4. Race progress fraction
        current_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        if total_laps <= 0:
            total_laps = 57
        race_progress = min(1.0, max(0.0, float(current_lap / total_laps)))

        # 5. Track & Air Temperatures
        cond = state.current_conditions
        tt = cond.track_temp if cond else None
        track_temp = float(tt) if tt is not None and not (isinstance(tt, float) and math.isnan(tt)) else 30.0

        at = cond.air_temp if cond else None
        air_temp = float(at) if at is not None and not (isinstance(at, float) and math.isnan(at)) else 25.0

        # 6. Safety Car / VSC flags
        is_sc = 1 if (cond and cond.has_safety_car) else 0
        is_vsc = 1 if (cond and cond.has_vsc) else 0

        # 7. Dynamic Lag1 residual and rolling 3-lap trend tracking
        driver_code = participant.driver or "UNKNOWN"
        stint_idx = int(participant.stint or 1)

        # Reset residuals if driver entered a new stint
        if self._driver_stints.get(driver_code) != stint_idx:
            self._driver_stints[driver_code] = stint_idx
            self._driver_residuals[driver_code] = []

        res_history = self._driver_residuals.get(driver_code, [])

        # Lag 1 residual (from previous lap in this stint)
        lag1_residual = float(res_history[-1]) if res_history else 0.0

        # Rolling 3-lap trend (slope over t-3 to t-1)
        rolling3_trend = self._calc_slope(res_history[-3:]) if len(res_history) >= 3 else 0.0

        # Update residual history with this lap's residual if clean lap time exists
        last_lap_time = participant.last_lap_time_seconds
        if (
            last_lap_time is not None
            and not (isinstance(last_lap_time, float) and math.isnan(last_lap_time))
            and not participant.is_pit_in_lap
            and not participant.is_pit_out_lap
        ):
            fuel_penalty = max(0.0, (total_laps - current_lap) * 0.065)
            expected_non_tyre = self.circuit_base_pace + fuel_penalty
            current_residual = float(last_lap_time - expected_non_tyre)
            self._driver_residuals.setdefault(driver_code, []).append(current_residual)

        # 8. Team historical context
        team_median_lag1 = 0.0

        return {
            "TyreAge": tyre_age,
            "Log_TyreAge": log_tyre_age,
            "StintLap": stint_lap,
            "Compound_Encoded": compound_encoded,
            "RaceProgressFraction": race_progress,
            "TrackTemp": track_temp,
            "AirTemp": air_temp,
            "is_safety_car": is_sc,
            "is_vsc": is_vsc,
            "Lag1_Pace_Residual": lag1_residual,
            "Rolling3_Degradation_Trend": rolling3_trend,
            "Team_Median_Pace_Lag1": team_median_lag1,
        }

    def predict_degradation(self, state: RaceState, driver: str) -> float | None:
        """
        Predict expected pure tyre degradation (seconds of pace loss) for a driver
        from the current RaceState snapshot.
        """
        if self.model is None:
            raise RuntimeError("Tyre degradation model is not loaded.")

        driver_code = normalize_driver(driver)
        if driver_code is None:
            return None

        participant = state.participants.get(driver_code)
        if participant is None:
            return None

        features = self.build_features(state, participant)
        df_row = pd.DataFrame([features])[self.feature_cols]
        pred_deg = float(self.model.predict(df_row)[0])
        return pred_deg

    def predict_all(self, state: RaceState) -> dict[str, float]:
        """
        Batch predict pure tyre degradation for all active participants in the race state.
        Returns {driver_code: degradation_seconds}.
        """
        if self.model is None:
            raise RuntimeError("Tyre degradation model is not loaded.")

        rows: list[dict[str, Any]] = []
        driver_codes: list[str] = []

        for driver_code, participant in state.participants.items():
            if not participant.is_active:
                continue
            features = self.build_features(state, participant)
            rows.append(features)
            driver_codes.append(driver_code)

        if not rows:
            return {}

        df_batch = pd.DataFrame(rows)[self.feature_cols]
        preds = self.model.predict(df_batch)

        return {d: float(p) for d, p in zip(driver_codes, preds)}

    def predict_lap_time(self, state: RaceState, driver: str, base_pace: float | None = None) -> float | None:
        """
        Reconstruct total expected lap time:
        predicted_lap_time = base_pace + fuel_weight_penalty + predicted_degradation
        """
        deg = self.predict_degradation(state, driver)
        if deg is None:
            return None

        circuit_base = base_pace or self.circuit_base_pace
        current_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        fuel_penalty = max(0.0, (total_laps - current_lap) * 0.065)

        return float(circuit_base + fuel_penalty + deg)
