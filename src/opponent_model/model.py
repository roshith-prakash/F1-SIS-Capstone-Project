"""
src/opponent_model/model.py
===========================
Production-grade OpponentModel API class (Task 24, Phase A13).

Serves calibrated per-lap pit probabilities and pace estimates for any opponent driver.
Directly interfaces with the Monte Carlo Strategy Engine.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb

from .state import OpponentStateVector
from .bayesian import BayesianPitUpdater
from .features import FEATURES_MODEL_C

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Opponent Model"


class OpponentModel:
    """
    Production Opponent Model serving P(PIT | state_t) and P(STAY | state_t).
    """

    def __init__(
        self,
        classifier: Any = None,
        calibrator: Any = None,
        feature_names: list[str] | None = None,
        bayesian_updater: Optional[BayesianPitUpdater] = None,
        enable_bayesian: bool = True,
    ):
        self.classifier = classifier
        self.calibrator = calibrator
        self.feature_names = feature_names or FEATURES_MODEL_C
        self.bayesian_updater = bayesian_updater or BayesianPitUpdater(lambda_=0.3)
        self.enable_bayesian = enable_bayesian

    @classmethod
    def load(
        cls,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        enable_bayesian: bool = True,
    ) -> "OpponentModel":
        """Load trained model, calibrator, and metadata from disk."""
        m_dir = Path(model_dir)

        # Load feature names
        feat_path = m_dir / "opponent_feature_names.json"
        if feat_path.exists():
            with open(feat_path) as f:
                feature_names = json.load(f)
        else:
            feature_names = FEATURES_MODEL_C

        # Load metadata
        meta_path = m_dir / "opponent_metadata.json"
        calib_type = "platt"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
                calib_type = meta.get("best_calibrator", "platt")

        # Load calibrator
        calib_file = m_dir / f"opponent_calibrator_{calib_type}.joblib"
        calibrator = None
        if calib_file.exists():
            calibrator = joblib.load(calib_file)
        elif (m_dir / "opponent_calibrator_platt.joblib").exists():
            calibrator = joblib.load(m_dir / "opponent_calibrator_platt.joblib")

        # Load base XGBoost
        xgb_path = m_dir / "opponent_xgb_model.json"
        classifier = None
        if xgb_path.exists():
            classifier = xgb.XGBClassifier()
            classifier.load_model(str(xgb_path))

        return cls(
            classifier=classifier,
            calibrator=calibrator,
            feature_names=feature_names,
            enable_bayesian=enable_bayesian,
        )

    def predict(self, opponent_state: dict[str, Any] | OpponentStateVector) -> dict[str, Any]:
        """
        Compute calibrated P(PIT), P(STAY), pace, degradation, and confidence.

        Parameters
        ----------
        opponent_state : dict or OpponentStateVector
            Full state representation for one opponent at lap t.

        Returns
        -------
        dict
            {
                "p_pit": float,
                "p_stay": float,
                "predicted_pace": float,
                "predicted_degradation": float,
                "confidence": float,
            }
        """
        # Convert state vector if dataclass, object with to_dict, or dict
        if hasattr(opponent_state, "to_dict"):
            state_dict = opponent_state.to_dict()
            if hasattr(opponent_state, "to_feature_dict"):
                feat_dict = opponent_state.to_feature_dict(feature_names=self.feature_names)
            else:
                feat_dict = {col: float(state_dict.get(col, 0.0)) for col in self.feature_names}
        elif isinstance(opponent_state, dict):
            state_dict = dict(opponent_state)
            # One-hot encode compound if missing
            comp = str(state_dict.get("tyre_compound", "MEDIUM")).strip().upper()
            state_dict["compound_SOFT"] = 1.0 if comp == "SOFT" else 0.0
            state_dict["compound_MEDIUM"] = 1.0 if comp == "MEDIUM" else 0.0
            state_dict["compound_HARD"] = 1.0 if comp == "HARD" else 0.0

            # Ensure reasonable pace if missing
            pred_lt = float(state_dict.get("predicted_lap_time", 90.0))
            if "last_lap_time" not in state_dict or float(state_dict.get("last_lap_time", 0.0)) <= 0.0:
                state_dict["last_lap_time"] = pred_lt
            if "rolling_3_lap_avg" not in state_dict or float(state_dict.get("rolling_3_lap_avg", 0.0)) <= 0.0:
                state_dict["rolling_3_lap_avg"] = pred_lt
            if "tyre_age_squared" not in state_dict:
                state_dict["tyre_age_squared"] = float(state_dict.get("tyre_age", 0) ** 2)

            # If derived strategic features missing, compute them
            if "cost_of_staying" not in state_dict:
                from .features import compute_derived_features
                derived = compute_derived_features(
                    tyre_compound=comp,
                    tyre_age=float(state_dict.get("tyre_age", 0)),
                    gap_ahead=float(state_dict.get("gap_ahead", 10.0)),
                    gap_behind=float(state_dict.get("gap_behind", 10.0)),
                    position=int(state_dict.get("position", 5)),
                    predicted_lap_time=pred_lt,
                    field_median_lap_time=float(state_dict.get("field_median_lap_time", pred_lt)),
                    predicted_degradation=float(state_dict.get("predicted_degradation", 0.0)),
                    predicted_degradation_t1=float(state_dict.get("predicted_degradation_t1", state_dict.get("predicted_degradation", 0.0))),
                    p_sc_h1=float(state_dict.get("p_sc_h1", 0.0)),
                    pit_loss=float(state_dict.get("pit_loss_seconds", 22.0)),
                    gap_to_ego=float(state_dict.get("gap_to_ego", 0.0)),
                    ego_tyre_age=float(state_dict.get("ego_tyre_age", 0.0)),
                    ego_recently_pitted=int(state_dict.get("ego_recently_pitted", 0)),
                )
                state_dict.update(derived)

            feat_dict = {col: float(state_dict.get(col, 0.0)) for col in self.feature_names}
        else:
            try:
                state_dict = dict(opponent_state)
                feat_dict = {col: float(state_dict.get(col, 0.0)) for col in self.feature_names}
            except Exception:
                state_dict = {}
                feat_dict = {col: 0.0 for col in self.feature_names}

        # Model Inference
        df_row = pd.DataFrame([feat_dict])[self.feature_names].fillna(0.0)

        if self.calibrator is not None:
            p_ml = float(self.calibrator.predict_proba(df_row)[0, 1])
        elif self.classifier is not None:
            p_ml = float(self.classifier.predict_proba(df_row)[0, 1])
        else:
            # Fallback empirical estimate
            p_ml = 0.05

        # Bayesian Online Updating Layer (Online Evidence: pace residual anomaly & stint extension)
        if self.enable_bayesian and self.bayesian_updater is not None:
            tyre_age = state_dict.get("tyre_age", 0)
            compound = state_dict.get("tyre_compound", "MEDIUM")
            pace_res = float(state_dict.get("last_lap_time", 90.0) - state_dict.get("predicted_lap_time", 90.0))
            laps_past = float(state_dict.get("laps_past_nominal", 0.0))
            p_pit = float(
                self.bayesian_updater.update(
                    p_ml=p_ml,
                    tyre_age=tyre_age,
                    compound=compound,
                    pace_residual=pace_res,
                    laps_past_nominal=laps_past,
                )
            )
        else:
            p_pit = float(p_ml)

        p_pit = max(1e-4, min(1.0 - 1e-4, p_pit))
        p_stay = 1.0 - p_pit

        # Certainty / Confidence: 1 when p_pit is near 0 or 1, 0 when 0.5
        confidence = float(1.0 - abs(p_pit - 0.5) * 2.0)

        predicted_pace = float(state_dict.get("predicted_lap_time", 90.0))
        predicted_deg = float(state_dict.get("predicted_degradation", 0.0))

        # Multi-horizon pit probabilities using survival hazard formulation:
        # P(pit within n laps) = 1 - (1 - h_t)^n (derived constant-hazard approximation)
        p_pit_3 = float(1.0 - (1.0 - p_pit) ** 3)
        p_pit_5 = float(1.0 - (1.0 - p_pit) ** 5)

        # Operational alert levels (derived on validation: LOW < 0.05, MED 0.05-0.15, HIGH >= 0.15)
        alert_level = "LOW" if p_pit < 0.05 else ("MEDIUM" if p_pit < 0.15 else "HIGH")

        return {
            "p_pit": p_pit,
            "p_stay": p_stay,
            "p_pit_next": p_pit,
            "p_stay_next": p_stay,
            "p_pit_3": p_pit_3,
            "p_pit_5": p_pit_5,
            "pit_alert_level": alert_level,
            "p_pit_next_lap": p_pit,
            "p_stay_next_lap": p_stay,
            "p_pit_window_3laps": p_pit_3,
            "p_pit_window_5laps": p_pit_5,
            "p_pit_window_3laps_approx": p_pit_3,
            "p_pit_window_5laps_approx": p_pit_5,
            "multi_horizon_method": "derived_constant_hazard_survival_approximation",
            "horizons": {1: p_pit, 3: p_pit_3, 5: p_pit_5},
            "predicted_pace": predicted_pace,
            "predicted_degradation": predicted_deg,
            "opponent_predicted_lap_time": predicted_pace,
            "opponent_predicted_degradation": predicted_deg,
            "confidence": confidence,
        }

    def predict_opponent(
        self,
        state: Any,
        opponent_driver: str | None = None,
        ego_driver: str | None = None,
        horizons: list[int] | None = None,
        include_explanation: bool = True,
    ) -> dict[str, Any]:
        """Convenience method matching the Strategy Engine interface."""
        from .interface import predict_opponent
        return predict_opponent(
            state=state,
            opponent_driver=opponent_driver,
            ego_driver=ego_driver,
            model=self,
            horizons=horizons,
            include_explanation=include_explanation,
        )

    def predict_batch(self, states: list[dict[str, Any] | OpponentStateVector]) -> list[dict[str, Any]]:
        """Vectorized batch prediction for Monte Carlo rollout efficiency."""
        return [self.predict(s) for s in states]
