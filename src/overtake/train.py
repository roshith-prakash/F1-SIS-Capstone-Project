"""
src/overtake/train.py
=====================
Training pipeline, feature ablation study, probability calibration,
and artifact serialization for the Overtake Probability Model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
import xgboost as xgb

from .features import (
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    FEATURES_MODEL_C,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Overtake Model"


def compute_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    
    total = len(y_true)
    if total == 0:
        return 0.0, 0.0
        
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        mask = (y_prob >= low) & (y_prob < high if i < n_bins - 1 else y_prob <= high)
        n_k = np.sum(mask)
        if n_k > 0:
            acc_k = float(np.mean(y_true[mask]))
            conf_k = float(np.mean(y_prob[mask]))
            err = abs(acc_k - conf_k)
            ece += (n_k / total) * err
            mce = max(mce, err)
    return float(ece), float(mce)


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """Compute comprehensive classification and calibration metrics."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float(np.mean(y_true))
    brier = float(brier_score_loss(y_true, y_prob))
    ll = float(log_loss(y_true, np.clip(y_prob, 1e-6, 1.0 - 1e-6)))
    ece, mce = compute_calibration_error(y_true, y_prob)

    # Precision at top 10%
    n_top10 = max(1, int(len(y_prob) * 0.10))
    top10_idx = np.argsort(y_prob)[::-1][:n_top10]
    p_top10 = float(np.mean(y_true[top10_idx])) if n_top10 > 0 else 0.0

    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "brier_score": brier,
        "log_loss": ll,
        "ece": ece,
        "mce": mce,
        "precision_top10": p_top10,
        "accuracy": acc,
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }


class OvertakeTrainer:
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

        self.models: dict[str, xgb.XGBClassifier] = {}
        self.calibrators: dict[str, Any] = {}
        self.best_calibrator_name: str = "platt"
        self.feature_names: list[str] = FEATURES_MODEL_C

    def build_xgb_classifier(self, scale_pos_weight: float = 7.0) -> xgb.XGBClassifier:
        """Instantiate XGBClassifier with canonical hyperparameters."""
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
        """Train a single XGBoost model with early stopping on validation split."""
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
    ) -> tuple[pd.DataFrame, dict[str, dict[str, float]]]:
        """
        Train and evaluate Model A (Gap only), Model B (+ Pace/Tyre), Model C (Full).
        Returns (ablation_df, raw_results).
        """
        feature_sets = {
            "Model A (Gap & Basic State)": FEATURES_MODEL_A,
            "Model B (+ Direct Pace & Tyre)": FEATURES_MODEL_B,
            "Model C (+ Circuit & Foundational)": FEATURES_MODEL_C,
        }

        results = {}
        rows = []

        for name, feats in feature_sets.items():
            model = self.train_model(df_train, df_val, feats, model_name=name)

            X_test = df_test[feats].fillna(0.0)
            y_test = df_test["target"].astype(int)
            y_prob = model.predict_proba(X_test)[:, 1]

            metrics = compute_metrics(y_test, y_prob)
            results[name] = metrics

            rows.append({
                "Model": name,
                "ROC-AUC": f"{metrics['roc_auc']:.4f}",
                "PR-AUC": f"{metrics['pr_auc']:.4f}",
                "Precision@top10%": f"{metrics['precision_top10']:.4f}",
                "Brier Score": f"{metrics['brier_score']:.4f}",
                "Log Loss": f"{metrics['log_loss']:.4f}",
                "ECE": f"{metrics['ece']:.4f}",
                "Accuracy": f"{metrics['accuracy']:.4f}",
                "F1": f"{metrics['f1']:.4f}",
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
        Compare Raw vs Platt vs Isotonic on 2025 test set.
        """
        model_c = self.models.get("Model C (+ Circuit & Foundational)")
        if model_c is None:
            model_c = self.train_model(df_train, df_val, FEATURES_MODEL_C, "Model C (+ Circuit & Foundational)")

        X_val = df_val[FEATURES_MODEL_C].fillna(0.0)
        y_val = df_val["target"].astype(int)

        X_test = df_test[FEATURES_MODEL_C].fillna(0.0)
        y_test = df_test["target"].astype(int)

        # 1. Platt Scaling & 2. Isotonic Regression on pre-fitted base model
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

        # Compare on Validation set to pick best
        val_prob_platt = platt.predict_proba(X_val)[:, 1]
        val_prob_iso = isotonic.predict_proba(X_val)[:, 1]

        ece_platt_val, _ = compute_calibration_error(y_val, val_prob_platt)
        ece_iso_val, _ = compute_calibration_error(y_val, val_prob_iso)

        if ece_platt_val <= ece_iso_val:
            self.best_calibrator_name = "platt"
            best_calibrator = platt
        else:
            self.best_calibrator_name = "isotonic"
            best_calibrator = isotonic

        # Evaluate Raw vs Platt vs Isotonic on Test Set
        raw_prob_test = model_c.predict_proba(X_test)[:, 1]
        platt_prob_test = platt.predict_proba(X_test)[:, 1]
        iso_prob_test = isotonic.predict_proba(X_test)[:, 1]

        m_raw = compute_metrics(y_test, raw_prob_test)
        m_platt = compute_metrics(y_test, platt_prob_test)
        m_iso = compute_metrics(y_test, iso_prob_test)

        calib_df = pd.DataFrame([
            {
                "Method": "Raw XGBoost (Pre-calibration)",
                "ROC-AUC": f"{m_raw['roc_auc']:.4f}",
                "PR-AUC": f"{m_raw['pr_auc']:.4f}",
                "Precision@top10%": f"{m_raw['precision_top10']:.4f}",
                "Brier Score": f"{m_raw['brier_score']:.4f}",
                "Log Loss": f"{m_raw['log_loss']:.4f}",
                "ECE": f"{m_raw['ece']:.4f}",
            },
            {
                "Method": "Platt Scaling (Sigmoid)",
                "ROC-AUC": f"{m_platt['roc_auc']:.4f}",
                "PR-AUC": f"{m_platt['pr_auc']:.4f}",
                "Precision@top10%": f"{m_platt['precision_top10']:.4f}",
                "Brier Score": f"{m_platt['brier_score']:.4f}",
                "Log Loss": f"{m_platt['log_loss']:.4f}",
                "ECE": f"{m_platt['ece']:.4f}",
            },
            {
                "Method": "Isotonic Regression",
                "ROC-AUC": f"{m_iso['roc_auc']:.4f}",
                "PR-AUC": f"{m_iso['pr_auc']:.4f}",
                "Precision@top10%": f"{m_iso['precision_top10']:.4f}",
                "Brier Score": f"{m_iso['brier_score']:.4f}",
                "Log Loss": f"{m_iso['log_loss']:.4f}",
                "ECE": f"{m_iso['ece']:.4f}",
            },
        ]).set_index("Method")

        return calib_df, best_calibrator

    def save_artifacts(
        self,
        best_calibrator: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Save production model artifacts to disk."""
        model_c = self.models.get("Model C (+ Circuit & Foundational)")
        if model_c is not None:
            model_path = self.model_dir / "overtake_model.json"
            model_c.save_model(str(model_path))

        calib_to_save = best_calibrator or self.calibrators.get(self.best_calibrator_name)
        if calib_to_save is not None:
            joblib.dump(calib_to_save, self.model_dir / "overtake_calibrator.joblib")

        with open(self.model_dir / "overtake_feature_names.json", "w", encoding="utf-8") as f:
            json.dump(FEATURES_MODEL_C, f, indent=2)

        meta = {
            "model_type": "xgboost_classifier",
            "model_version": "Model_C",
            "best_calibrator": self.best_calibrator_name,
            "feature_count": len(FEATURES_MODEL_C),
            **(metadata or {}),
        }
        with open(self.model_dir / "overtake_metadata.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
