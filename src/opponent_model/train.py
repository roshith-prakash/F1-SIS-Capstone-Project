"""
src/opponent_model/train.py
===========================
Training pipeline, ablation study, and calibration for Opponent Modeling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV

from .features import FEATURES_MODEL_A, FEATURES_MODEL_B, FEATURES_MODEL_C
from .evaluate import compute_all_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Opponent Model"


class OpponentTrainer:
    """
    Manages XGBoost training, ablation study across feature sets,
    and probability calibration on validation split.
    """

    def __init__(
        self,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
        random_state: int = 42,
    ):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

        # Trained artifacts
        self.models: dict[str, xgb.XGBClassifier] = {}
        self.calibrators: dict[str, CalibratedClassifierCV] = {}
        self.best_calibrator_name: str = "platt"
        self.feature_names: list[str] = FEATURES_MODEL_C

    def build_xgb_classifier(self, scale_pos_weight: float = 19.0) -> xgb.XGBClassifier:
        """Instantiate XGBClassifier with the canonical hyperparameters."""
        return xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            early_stopping_rounds=50,
            random_state=self.random_state,
            n_jobs=-1,
        )

    def train_model(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        features: list[str],
        model_name: str = "Model_C",
    ) -> xgb.XGBClassifier:
        """Train a single XGBoost model with early stopping on validation set."""
        X_train = df_train[features].fillna(0.0)
        y_train = df_train["target"].astype(int)

        X_val = df_val[features].fillna(0.0)
        y_val = df_val["target"].astype(int)

        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        scale_pos_weight = float(n_neg / max(1, n_pos))

        model = self.build_xgb_classifier(scale_pos_weight=scale_pos_weight)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self.models[model_name] = model
        return model

    def run_ablation_study(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
        """
        Train and evaluate Model A, Model B, and Model C on identical test data.
        Returns (ablation_df, raw_results).
        """
        feature_sets = {
            "Model A (Race State Only)": FEATURES_MODEL_A,
            "Model B (+ Foundational Models)": FEATURES_MODEL_B,
            "Model C (+ Derived Strategic)": FEATURES_MODEL_C,
        }

        results = {}
        rows = []

        for name, feats in feature_sets.items():
            model = self.train_model(df_train, df_val, feats, model_name=name)

            X_test = df_test[feats].fillna(0.0)
            y_test = df_test["target"].astype(int)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = compute_all_metrics(y_test, y_prob)
            results[name] = metrics

            rows.append({
                "Model": name,
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
                "Precision": f"{metrics['precision']:.4f}",
                "Recall": f"{metrics['recall']:.4f}",
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "Brier Score": f"{metrics['brier_score']:.4f}",
                "Log Loss": f"{metrics['log_loss']:.4f}",
                "ECE": f"{metrics['ece']:.4f}",
                "Action Error": f"{metrics['action_error']:.4f}",
            })

        ablation_df = pd.DataFrame(rows).set_index("Model")
        return ablation_df, results

    def fit_calibration(
        self,
        df_train: pd.DataFrame,
        df_val: pd.DataFrame,
        df_test: pd.DataFrame,
    ) -> tuple[pd.DataFrame, Any]:
        """
        Fit Platt (sigmoid) and Isotonic calibration on validation set.
        Compare Raw vs Platt vs Isotonic on test set.
        """
        model_c = self.models.get("Model C (+ Derived Strategic)")
        if model_c is None:
            model_c = self.train_model(df_train, df_val, FEATURES_MODEL_C, "Model C (+ Derived Strategic)")

        X_val = df_val[FEATURES_MODEL_C].fillna(0.0)
        y_val = df_val["target"].astype(int)

        X_test = df_test[FEATURES_MODEL_C].fillna(0.0)
        y_test = df_test["target"].astype(int)

        # 1. Platt Scaling & 2. Isotonic Regression on pre-fitted model
        try:
            from sklearn.frozen import FrozenEstimator
            est = FrozenEstimator(model_c)
            platt = CalibratedClassifierCV(estimator=est, method="sigmoid")
            isotonic = CalibratedClassifierCV(estimator=est, method="isotonic")
        except ImportError:
            platt = CalibratedClassifierCV(estimator=model_c, method="sigmoid", cv="prefit")
            isotonic = CalibratedClassifierCV(estimator=model_c, method="isotonic", cv="prefit")

        platt.fit(X_val, y_val)
        self.calibrators["platt"] = platt

        isotonic.fit(X_val, y_val)
        self.calibrators["isotonic"] = isotonic

        # Compare on Validation to pick best
        val_prob_platt = platt.predict_proba(X_val)[:, 1]
        val_prob_iso = isotonic.predict_proba(X_val)[:, 1]

        m_platt_val = compute_all_metrics(y_val, val_prob_platt)
        m_iso_val = compute_all_metrics(y_val, val_prob_iso)

        if m_platt_val["ece"] <= m_iso_val["ece"]:
            self.best_calibrator_name = "platt"
            best_calibrator = platt
        else:
            self.best_calibrator_name = "isotonic"
            best_calibrator = isotonic

        # Evaluate Raw vs Platt vs Isotonic on Test Set
        raw_prob_test = model_c.predict_proba(X_test)[:, 1]
        platt_prob_test = platt.predict_proba(X_test)[:, 1]
        iso_prob_test = isotonic.predict_proba(X_test)[:, 1]

        m_raw = compute_all_metrics(y_test, raw_prob_test)
        m_platt = compute_all_metrics(y_test, platt_prob_test)
        m_iso = compute_all_metrics(y_test, iso_prob_test)

        calib_df = pd.DataFrame([
            {
                "Method": "Raw XGBoost",
                "Brier Score": f"{m_raw['brier_score']:.4f}",
                "Log Loss": f"{m_raw['log_loss']:.4f}",
                "ECE": f"{m_raw['ece']:.4f}",
                "MCE": f"{m_raw['mce']:.4f}",
                "ROC-AUC": f"{m_raw['roc_auc']:.4f}",
                "F1": f"{m_raw['f1']:.4f}",
            },
            {
                "Method": "Platt Scaling (Sigmoid)",
                "Brier Score": f"{m_platt['brier_score']:.4f}",
                "Log Loss": f"{m_platt['log_loss']:.4f}",
                "ECE": f"{m_platt['ece']:.4f}",
                "MCE": f"{m_platt['mce']:.4f}",
                "ROC-AUC": f"{m_platt['roc_auc']:.4f}",
                "F1": f"{m_platt['f1']:.4f}",
            },
            {
                "Method": "Isotonic Regression",
                "Brier Score": f"{m_iso['brier_score']:.4f}",
                "Log Loss": f"{m_iso['log_loss']:.4f}",
                "ECE": f"{m_iso['ece']:.4f}",
                "MCE": f"{m_iso['mce']:.4f}",
                "ROC-AUC": f"{m_iso['roc_auc']:.4f}",
                "F1": f"{m_iso['f1']:.4f}",
            },
        ]).set_index("Method")

        return calib_df, best_calibrator

    def save_artifacts(
        self,
        best_calibrator: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save production model artifacts to disk."""
        model_c = self.models.get("Model C (+ Derived Strategic)")
        if model_c is not None:
            model_path = self.model_dir / "opponent_xgb_model.json"
            model_c.save_model(str(model_path))

        if "platt" in self.calibrators:
            joblib.dump(self.calibrators["platt"], self.model_dir / "opponent_calibrator_platt.joblib")
        if "isotonic" in self.calibrators:
            joblib.dump(self.calibrators["isotonic"], self.model_dir / "opponent_calibrator_isotonic.joblib")

        # Save feature names
        with open(self.model_dir / "opponent_feature_names.json", "w") as f:
            json.dump(FEATURES_MODEL_C, f, indent=2)

        # Save metadata
        meta = {
            "model_version": "C",
            "best_calibrator": self.best_calibrator_name,
            "feature_count": len(FEATURES_MODEL_C),
            **(metadata or {}),
        }
        with open(self.model_dir / "opponent_metadata.json", "w") as f:
            json.dump(meta, f, indent=2)
