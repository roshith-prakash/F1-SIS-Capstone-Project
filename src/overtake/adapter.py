"""
src/overtake/adapter.py
=======================
Production OvertakeAdapter interface class (Phase 5).

Serves calibrated per-lap overtake probabilities and data-derived dirty air penalties
for any trailing/leading driver pair during race simulation and analysis.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

from .features import (
    DEFAULT_CIRCUIT_DIFFICULTY,
    FEATURES_MODEL_C,
    compute_overtake_features,
    load_circuit_overtake_difficulty,
    resolve_circuit_name,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Overtake Model"


class OvertakeAdapter:
    """
    Production adapter providing P(overtake | state) and dirty air wake penalties.
    Directly interfaces with RaceScenarioSimulator.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        model: Any = None,
        calibrator: Any = None,
        feature_names: list[str] | None = None,
        circuit_difficulties: dict[str, float] | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.model = model
        self.calibrator = calibrator
        self.feature_names = feature_names or FEATURES_MODEL_C
        self.circuit_difficulties = circuit_difficulties or {}

        # High-performance simulation inference caches
        self._cache: dict[tuple[Any, ...], float] = {}
        self._dirty_air_cache: dict[tuple[float, str], float] = {}

        # If model/calibrator not supplied, attempt to load from disk
        if self.model is None or self.calibrator is None:
            self._load_artifacts()

    def clear_cache(self) -> None:
        """Clear model memoization caches."""
        self._cache.clear()
        self._dirty_air_cache.clear()

    def _load_artifacts(self) -> None:
        """Load trained XGBoost model, calibrator, metadata, and circuit difficulties."""
        if not self.model_dir.exists():
            return

        # 1. Load feature names
        feat_path = self.model_dir / "overtake_feature_names.json"
        if feat_path.exists():
            try:
                with open(feat_path, "r", encoding="utf-8") as f:
                    self.feature_names = json.load(f)
            except Exception:
                self.feature_names = FEATURES_MODEL_C

        # 2. Load circuit difficulty index
        self.circuit_difficulties = load_circuit_overtake_difficulty(self.model_dir)

        # 3. Load Calibrator
        calib_path = self.model_dir / "overtake_calibrator.joblib"
        if calib_path.exists():
            try:
                self.calibrator = joblib.load(calib_path)
            except Exception:
                self.calibrator = None

        # 4. Load base XGBoost Model
        model_path = self.model_dir / "overtake_model.json"
        if model_path.exists():
            try:
                classifier = xgb.XGBClassifier()
                classifier.load_model(str(model_path))
                self.model = classifier
            except Exception:
                self.model = None

    def predict_overtake_probability(
        self,
        gap_seconds: float,
        pace_delta: float,
        tyre_age_delta: float,
        compound_behind: str,
        compound_ahead: str,
        is_fresh_tyre_behind: bool | int,
        circuit: str,
        is_sc_restart: bool | int = False,
        is_sc: bool | int = False,
        is_vsc: bool | int = False,
        rolling_3lap_pace_delta: float | None = None,
        predicted_lap_time_behind: float | None = None,
        predicted_lap_time_ahead: float | None = None,
        predicted_deg_behind: float | None = None,
        predicted_deg_ahead: float | None = None,
    ) -> float:
        """
        Compute calibrated P(overtake) in [0.0, 1.0].

        Parameters
        ----------
        gap_seconds : float
            Current gap between trailing and leading car at lap end.
        pace_delta : float
            Positive if trailing car is faster: lap_time_ahead - lap_time_behind.
        tyre_age_delta : float
            tyre_age_ahead - tyre_age_behind (positive = trailing car fresher).
        compound_behind : str
            Compound string ("SOFT", "MEDIUM", "HARD", etc.).
        compound_ahead : str
            Compound string of car ahead.
        is_fresh_tyre_behind : bool
            True if trailing car is on lap 1 or 2 of stint.
        circuit : str
            Track name or location identifier.
        is_sc_restart : bool
            True if lap is 1 or 2 laps post Safety Car restart.
        is_sc : bool
            True if Safety Car is active.
        """
        # Under Safety Car, overtaking is illegal
        if is_sc:
            return 0.0

        # Beyond 2.0s gap, probability of an on-track pass on the next lap is negligible
        if gap_seconds > 2.0:
            return 0.001

        # Fast pruning: if trailing car has severe pace disadvantage and no tyre advantage, pass is impossible
        if pace_delta < -0.7 and tyre_age_delta <= 0:
            return 0.001

        canon_circuit = resolve_circuit_name(circuit)
        # Quantize inputs for high-performance Monte Carlo memoization
        g_bucket = round(gap_seconds * 10) / 10.0
        p_bucket = round(pace_delta * 2) / 2.0
        t_bucket = round(tyre_age_delta / 2.0) * 2.0

        cache_key = (
            g_bucket,
            p_bucket,
            t_bucket,
            str(compound_behind).strip().upper(),
            str(compound_ahead).strip().upper(),
            bool(is_fresh_tyre_behind),
            canon_circuit,
            bool(is_sc_restart),
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Check if model or calibrator is available
        if self.calibrator is not None or self.model is not None:
            try:
                feat_dict = compute_overtake_features(
                    gap_seconds=g_bucket,
                    pace_delta=p_bucket,
                    tyre_age_delta=t_bucket,
                    compound_behind=compound_behind,
                    compound_ahead=compound_ahead,
                    is_fresh_tyre_behind=is_fresh_tyre_behind,
                    circuit=circuit,
                    is_sc_restart=is_sc_restart,
                    is_sc=is_sc,
                    is_vsc=is_vsc,
                    rolling_3lap_pace_delta=rolling_3lap_pace_delta,
                    predicted_lap_time_behind=predicted_lap_time_behind,
                    predicted_lap_time_ahead=predicted_lap_time_ahead,
                    predicted_deg_behind=predicted_deg_behind,
                    predicted_deg_ahead=predicted_deg_ahead,
                    circuit_difficulty_map=self.circuit_difficulties,
                )
                df_row = pd.DataFrame([feat_dict])[self.feature_names].fillna(0.0)

                if self.calibrator is not None:
                    p_overtake = float(self.calibrator.predict_proba(df_row)[0, 1])
                else:
                    p_overtake = float(self.model.predict_proba(df_row)[0, 1])

                prob = float(np.clip(p_overtake, 0.001, 0.999))
                self._cache[cache_key] = prob
                return prob
            except Exception:
                pass

        # Fallback heuristic: smooth logistic sigmoid on pace delta with gap modulation
        # Centred at 0.80s pace advantage threshold
        x = (p_bucket - 0.80) / 0.35
        base_p = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, x))))
        gap_penalty = max(0.1, 1.0 - (g_bucket / 1.5))
        prob = float(np.clip(base_p * gap_penalty, 0.001, 0.95))
        self._cache[cache_key] = prob
        return prob

    def predict_dirty_air_penalty(
        self,
        gap_seconds: float,
        circuit: str,
    ) -> float:
        """
        Returns estimated pace penalty (seconds) suffered by the trailing car
        from dirty air turbulence when following closely without passing.

        Replaces the hardcoded constant 0.35s with a circuit-aware, proximity-modulated penalty.
        """
        if gap_seconds >= 1.5:
            return 0.0

        canon_circuit = resolve_circuit_name(circuit)
        g_bucket = round(gap_seconds * 10) / 10.0
        dirty_key = (g_bucket, canon_circuit)
        if dirty_key in self._dirty_air_cache:
            return self._dirty_air_cache[dirty_key]

        difficulty = self.circuit_difficulties.get(canon_circuit, DEFAULT_CIRCUIT_DIFFICULTY)

        # Proximity decay: highest when glued to the gearbox (< 0.5s)
        proximity_factor = float(np.clip(1.0 - (g_bucket / 1.5), 0.0, 1.0))

        # Modulated by circuit difficulty (twisty circuits incur harsher dirty air disruption)
        circuit_multiplier = 0.5 + 0.6 * difficulty
        penalty = float(np.clip(0.40 * (proximity_factor ** 0.8) * circuit_multiplier, 0.05, 0.65))

        self._dirty_air_cache[dirty_key] = penalty
        return penalty
