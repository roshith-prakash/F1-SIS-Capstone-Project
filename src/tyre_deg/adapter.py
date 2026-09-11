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
    PROJECT_ROOT / "models" / "Tyre Degradation Estimation" / "xgboost_baseline.json",
    PROJECT_ROOT / "models" / "Tyre Degradation Estimation" / "tyre_deg_model.json",
    PROJECT_ROOT / "models" / "Tyre Degradation" / "xgboost_baseline.json",
    PROJECT_ROOT / "outputs" / "models" / "xgboost_baseline.json",
    PROJECT_ROOT / "models" / "xgboost_baseline.json",
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

CIRCUIT_BASE_PACES: dict[str, float] = {
    "Abu_Dhabi": 88.391,
    "Australia": 80.260,
    "Austria": 67.583,
    "Azerbaijan": 103.370,
    "Bahrain": 93.996,
    "Belgium": 107.305,
    "Brazil": 72.486,
    "Canada": 74.481,
    "China": 97.810,
    "Emilia_Romagna": 78.446,
    "France": 98.088,
    "Great_Britain": 90.510,
    "Hungary": 80.504,
    "Italy": 84.030,
    "Japan": 93.706,
    "Las_Vegas": 95.490,
    "Mexico": 80.153,
    "Miami": 91.361,
    "Monaco": 74.693,
    "Netherlands": 73.652,
    "Qatar": 84.319,
    "Saudi_Arabia": 91.634,
    "Singapore": 95.867,
    "Spain": 84.108,
    "United_States": 98.139,
}

CIRCUIT_ALIASES: dict[str, str] = {
    "monza": "Italy",
    "italian": "Italy",
    "italian grand prix": "Italy",
    "italy": "Italy",
    "silverstone": "Great_Britain",
    "great britain": "Great_Britain",
    "british": "Great_Britain",
    "british grand prix": "Great_Britain",
    "spielberg": "Austria",
    "red bull ring": "Austria",
    "austria": "Austria",
    "austrian": "Austria",
    "austrian grand prix": "Austria",
    "spa": "Belgium",
    "spa-francorchamps": "Belgium",
    "belgium": "Belgium",
    "belgian": "Belgium",
    "belgian grand prix": "Belgium",
    "sakhir": "Bahrain",
    "bahrain": "Bahrain",
    "bahrain grand prix": "Bahrain",
    "yas island": "Abu_Dhabi",
    "abu dhabi": "Abu_Dhabi",
    "abu_dhabi": "Abu_Dhabi",
    "abu dhabi grand prix": "Abu_Dhabi",
    "monte carlo": "Monaco",
    "monaco": "Monaco",
    "monaco grand prix": "Monaco",
    "albert park": "Australia",
    "melbourne": "Australia",
    "australia": "Australia",
    "australian": "Australia",
    "australian grand prix": "Australia",
    "baku": "Azerbaijan",
    "azerbaijan": "Azerbaijan",
    "interlagos": "Brazil",
    "sao paulo": "Brazil",
    "brazil": "Brazil",
    "montreal": "Canada",
    "canada": "Canada",
    "shanghai": "China",
    "china": "China",
    "zandvoort": "Netherlands",
    "netherlands": "Netherlands",
    "imola": "Emilia_Romagna",
    "emilia romagna": "Emilia_Romagna",
    "hungaroring": "Hungary",
    "hungary": "Hungary",
    "suzuka": "Japan",
    "japan": "Japan",
    "las vegas": "Las_Vegas",
    "mexico": "Mexico",
    "miami": "Miami",
    "losail": "Qatar",
    "qatar": "Qatar",
    "jeddah": "Saudi_Arabia",
    "saudi arabia": "Saudi_Arabia",
    "marina bay": "Singapore",
    "singapore": "Singapore",
    "catalunya": "Spain",
    "barcelona": "Spain",
    "spain": "Spain",
    "cota": "United_States",
    "austin": "United_States",
    "united states": "United_States",
}

TEAM_PACE_PRIORS: dict[str, float] = {
    "Red Bull": -1.1012,
    "Red Bull Racing": -1.1012,
    "Mercedes": -0.6200,
    "Mercedes-AMG": -0.6200,
    "Ferrari": -0.6080,
    "Scuderia Ferrari": -0.6080,
    "Alpine": -0.0638,
    "Alpine F1 Team": -0.0638,
    "McLaren": 0.0585,
    "McLaren F1 Team": 0.0585,
    "Aston Martin": 0.1680,
    "Aston Martin F1 Team": 0.1680,
    "VCARB": 0.1783,
    "RB": 0.1783,
    "Racing Bulls": 0.1783,
    "AlphaTauri": 0.1783,
    "Sauber": 0.1940,
    "Kick Sauber": 0.1940,
    "Alfa Romeo": 0.1940,
    "Haas": 0.4630,
    "Haas F1 Team": 0.4630,
    "Williams": 0.7173,
    "Williams Racing": 0.7173,
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
        self.default_base_pace = circuit_base_pace
        self.circuit_base_pace = circuit_base_pace or 85.0

        # In-memory stint history tracker: driver -> list of pace residuals in current stint
        self._driver_residuals: dict[str, list[float]] = {}
        self._driver_stints: dict[str, int] = {}
        self._last_recorded_lap: dict[str, int] = {}

        if model is not None:
            self.model = model
        else:
            loaded_model = None
            candidate_paths = [Path(model_path)] if model_path else DEFAULT_MODEL_PATHS
            for p in candidate_paths:
                if p and p.exists():
                    try:
                        if str(p).endswith((".joblib", ".pkl")):
                            import joblib
                            loaded_model = joblib.load(str(p))
                        else:
                            m = xgb.XGBRegressor()
                            m.load_model(str(p))
                            loaded_model = m
                        break
                    except Exception:
                        try:
                            import joblib
                            loaded_model = joblib.load(str(p))
                            break
                        except Exception:
                            continue
            self.model = loaded_model

    def resolve_circuit(self, state: RaceState) -> str:
        """Resolve the canonical circuit name from RaceState attributes."""
        candidates = [
            getattr(state, "location", None),
            getattr(state, "country", None),
            getattr(state, "grand_prix", None),
        ]
        for candidate in candidates:
            if not candidate:
                continue
            cleaned = str(candidate).strip()
            if cleaned in CIRCUIT_BASE_PACES:
                return cleaned
            low = cleaned.lower()
            if low in CIRCUIT_ALIASES:
                target = CIRCUIT_ALIASES[low]
                if target in CIRCUIT_BASE_PACES:
                    return target
            clean_under = cleaned.replace(" ", "_")
            if clean_under in CIRCUIT_BASE_PACES:
                return clean_under
            for circ in CIRCUIT_BASE_PACES:
                if circ.lower() in low or low in circ.lower():
                    return circ
        return "Italy"

    def get_circuit_base(self, state: RaceState) -> float:
        """Get the circuit baseline pace (seconds)."""
        if self.default_base_pace is not None:
            return self.default_base_pace
        circ = self.resolve_circuit(state)
        return CIRCUIT_BASE_PACES.get(circ, 85.0)

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
        directly from the RaceState snapshot. This method is a pure reader and does
        not mutate residual history.
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
            self._last_recorded_lap[driver_code] = 0

        res_history = self._driver_residuals.get(driver_code, [])

        # Lag 1 residual (from previous lap in this stint, bounded to prevent outlier divergence)
        lag1_residual = float(np.clip(res_history[-1], -4.0, 6.0)) if res_history else 0.0

        # Rolling 3-lap trend (slope over t-3 to t-1)
        rolling3_trend = float(self._calc_slope(res_history[-3:])) if len(res_history) >= 3 else 0.0

        # 8. Team historical context
        team_str = str(participant.team or "Red Bull Racing").strip()
        team_median_lag1 = TEAM_PACE_PRIORS.get(team_str, 0.0)

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

    def observe_lap(self, state: RaceState) -> None:
        """
        Record pace residuals once per committed lap boundary.
        Ignores caution laps (SC/VSC), pit laps, and extreme outlier laps.
        """
        cond = state.current_conditions
        if cond and (cond.has_safety_car or cond.has_vsc or cond.has_red_flag):
            return

        current_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        base_pace = self.get_circuit_base(state)
        fuel_penalty = max(0.0, (total_laps - current_lap) * 0.065)
        expected_non_tyre = base_pace + fuel_penalty

        for driver_code, participant in state.participants.items():
            if not participant.is_active:
                continue
            last_lap_time = participant.last_lap_time_seconds
            if (
                last_lap_time is None
                or (isinstance(last_lap_time, float) and math.isnan(last_lap_time))
                or participant.is_pit_in_lap
                or participant.is_pit_out_lap
            ):
                continue

            if self._last_recorded_lap.get(driver_code) == current_lap:
                continue

            # Filter out abnormal laps (> 15s off pace from incidents)
            if abs(last_lap_time - expected_non_tyre) > 15.0:
                continue

            residual = float(last_lap_time - expected_non_tyre)
            self._driver_residuals.setdefault(driver_code, []).append(residual)
            self._last_recorded_lap[driver_code] = current_lap

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

        circuit_base = base_pace if base_pace is not None else self.get_circuit_base(state)
        current_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        fuel_penalty = max(0.0, (total_laps - current_lap) * 0.065)

        return float(circuit_base + fuel_penalty + deg)
