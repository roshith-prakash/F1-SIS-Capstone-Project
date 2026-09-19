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
        # Convert state vector if dataclass
        if isinstance(opponent_state, OpponentStateVector):
            state_dict = opponent_state.to_dict()
            feat_dict = opponent_state.to_feature_dict("C")
        else:
            state_dict = dict(opponent_state)
            feat_dict = {}
            # One-hot encode compound if missing
            comp = str(state_dict.get("tyre_compound", "MEDIUM")).strip().upper()
            state_dict["compound_SOFT"] = 1.0 if comp == "SOFT" else 0.0
            state_dict["compound_MEDIUM"] = 1.0 if comp == "MEDIUM" else 0.0
            state_dict["compound_HARD"] = 1.0 if comp == "HARD" else 0.0
            for col in self.feature_names:
                feat_dict[col] = float(state_dict.get(col, 0.0))

        # Model Inference
        df_row = pd.DataFrame([feat_dict])[self.feature_names].fillna(0.0)

        if self.calibrator is not None:
            p_ml = float(self.calibrator.predict_proba(df_row)[0, 1])
        elif self.classifier is not None:
            p_ml = float(self.classifier.predict_proba(df_row)[0, 1])
        else:
            # Fallback empirical estimate
            p_ml = 0.05

        # Bayesian Online Updating Layer
        if self.enable_bayesian and self.bayesian_updater is not None:
            tyre_age = state_dict.get("tyre_age", 0)
            compound = state_dict.get("tyre_compound", "MEDIUM")
            p_pit = float(self.bayesian_updater.update(p_ml, tyre_age, compound))
        else:
            p_pit = float(p_ml)

        p_pit = max(1e-4, min(1.0 - 1e-4, p_pit))
        p_stay = 1.0 - p_pit

        # Certainty / Confidence: 1 when p_pit is near 0 or 1, 0 when 0.5
        confidence = float(1.0 - abs(p_pit - 0.5) * 2.0)

        predicted_pace = float(state_dict.get("predicted_lap_time", 90.0))
        predicted_deg = float(state_dict.get("predicted_degradation", 0.0))

        # Multi-horizon pit probabilities using survival hazard formulation:
        # P(pit within n laps) = 1 - (1 - h_t)^n
        p_pit_3 = float(1.0 - (1.0 - p_pit) ** 3)
        p_pit_5 = float(1.0 - (1.0 - p_pit) ** 5)

        return {
            "p_pit": p_pit,
            "p_stay": p_stay,
            "p_pit_next_lap": p_pit,
            "p_stay_next_lap": p_stay,
            "p_pit_window_3laps": p_pit_3,
            "p_pit_window_5laps": p_pit_5,
            "horizons": {1: p_pit, 3: p_pit_3, 5: p_pit_5},
            "predicted_pace": predicted_pace,
            "predicted_degradation": predicted_deg,
            "confidence": confidence,
        }

    def predict_batch(self, states: list[dict[str, Any] | OpponentStateVector]) -> list[dict[str, Any]]:
        """Vectorized batch prediction for Monte Carlo rollout efficiency."""
        return [self.predict(s) for s in states]
