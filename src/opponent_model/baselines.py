"""
src/opponent_model/baselines.py
===============================
Baseline models for Opponent Modeling benchmark comparison:
- Baseline 1: Majority Class Baseline (Always STAY)
- Baseline 2: Historical Marginal Frequency Baseline
- Baseline 3: State-Conditioned Frequency Baseline (binned lookup)
- Standard ML Baselines: Logistic Regression, Random Forest
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
)


class MajorityClassBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline 1: Majority Class Baseline.
    Always predicts the most common action (STAY).
    """

    def __init__(self, p_stay: float = 0.999):
        self.p_stay = p_stay
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        n = len(X)
        p_pit = 1.0 - self.p_stay
        return np.full((n, 2), [self.p_stay, p_pit])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)


# Alias for backwards compatibility
AlwaysStayBaseline = MajorityClassBaseline


class AlwaysPitBaseline(BaseEstimator, ClassifierMixin):
    """Baseline that always predicts PIT with high probability."""

    def __init__(self, p_pit: float = 0.95):
        self.p_pit = p_pit
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X):
        n = len(X)
        p_stay = 1.0 - self.p_pit
        return np.full((n, 2), [p_stay, self.p_pit])

    def predict(self, X):
        return np.ones(len(X), dtype=int)


class HistoricalFrequencyBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline 2: Historical Frequency Baseline.
    Predicts constant pit/stay probabilities from historical marginal frequencies.
    """

    def __init__(self):
        self.p_pit = 0.05
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.p_pit = float(np.clip(np.mean(y_arr), 1e-4, 1.0 - 1e-4))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p_pit), np.full(n, self.p_pit)])

    def predict(self, X):
        return (np.full(len(X), self.p_pit) >= 0.5).astype(int)


# Alias for backwards compatibility
EmpiricalBaseline = HistoricalFrequencyBaseline


class StateConditionedBaseline(BaseEstimator, ClassifierMixin):
    """
    Baseline 3: State-Conditioned Frequency Baseline.
    Estimates pit probabilities conditioned on simple race-state groupings:
    - Tyre age bins: [0-10], [11-20], [21-30], [31+]
    - Compound: SOFT, MEDIUM, HARD
    - Race phase: Early, Mid, Late
    Uses Bayesian/Laplace smoothing for small sample bins:
    P(PIT | bin) = (n_pit + alpha) / (n_total + alpha + beta)
    """

    def __init__(self, alpha: float = 1.0, beta: float = 20.0):
        self.alpha = alpha
        self.beta = beta
        self.lookup_table_: dict[tuple[int, str, str], float] = {}
        self.global_p_pit_: float = 0.05
        self.classes_ = np.array([0, 1])

    def _get_bins(self, X: pd.DataFrame | np.ndarray) -> list[tuple[int, str, str]]:
        if not isinstance(X, pd.DataFrame):
            # Fallback if raw numpy array
            return [(0, "MEDIUM", "mid")] * len(X)

        bins = []
        for _, row in X.iterrows():
            # Tyre age bin
            age = row.get("tyre_age", 0)
            if age <= 10:
                age_bin = 0
            elif age <= 20:
                age_bin = 1
            elif age <= 30:
                age_bin = 2
            else:
                age_bin = 3

            # Compound
            if row.get("compound_SOFT", 0) == 1:
                compound = "SOFT"
            elif row.get("compound_HARD", 0) == 1:
                compound = "HARD"
            else:
                compound = "MEDIUM"

            # Race phase
            prog = row.get("race_progress_fraction", 0.5)
            if prog < 0.33:
                phase = "early"
            elif prog < 0.67:
                phase = "mid"
            else:
                phase = "late"

            bins.append((age_bin, compound, phase))
        return bins

    def fit(self, X: pd.DataFrame | np.ndarray, y: Any):
        y_arr = np.asarray(y)
        self.global_p_pit_ = float(np.mean(y_arr))

        if isinstance(X, pd.DataFrame):
            bins = self._get_bins(X)
            df = pd.DataFrame({"bin": bins, "target": y_arr})
            grouped = df.groupby("bin")["target"].agg(["sum", "count"])
            for bin_key, row in grouped.iterrows():
                n_pit = row["sum"]
                n_total = row["count"]
                smoothed_p = (n_pit + self.alpha) / (n_total + self.alpha + self.beta)
                self.lookup_table_[bin_key] = float(smoothed_p)
        return self

    def predict_proba(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        bins = self._get_bins(X)
        p_pits = np.array([
            self.lookup_table_.get(b, self.global_p_pit_) for b in bins
        ])
        p_pits = np.clip(p_pits, 1e-4, 1.0 - 1e-4)
        return np.column_stack([1.0 - p_pits, p_pits])

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        p_pits = self.predict_proba(X)[:, 1]
        return (p_pits >= 0.5).astype(int)


def build_logistic_regression_baseline(C: float = 1.0) -> Pipeline:
    """Build standardized Logistic Regression baseline with balanced class weights."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            random_state=42,
        )),
    ])


def build_random_forest_baseline(
    n_estimators: int = 300,
    max_depth: int = 10,
    random_state: int = 42,
) -> RandomForestClassifier:
    """Build Random Forest baseline with balanced class weights."""
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def evaluate_all_baselines(
    models: dict[str, Any],
    X_test: pd.DataFrame | np.ndarray,
    y_test: np.ndarray | pd.Series,
) -> pd.DataFrame:
    """
    Evaluate all baselines and ML models on a test set.
    Returns comparison table covering:
    - Log Loss, Brier Score, Accuracy, Precision, Recall, F1.
    """
    y_true = np.asarray(y_test)
    records = []

    for name, model in models.items():
        probs = model.predict_proba(X_test)[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        preds = (probs >= 0.5).astype(int)

        brier = brier_score_loss(y_true, probs)
        ll = log_loss(y_true, probs)
        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        try:
            auc = roc_auc_score(y_true, probs)
        except Exception:
            auc = 0.5
        try:
            prauc = average_precision_score(y_true, probs)
        except Exception:
            prauc = float(np.mean(y_true))

        records.append({
            "Model": name,
            "ROC-AUC": round(float(auc), 4),
            "PR-AUC": round(float(prauc), 4),
            "Log Loss": round(float(ll), 4),
            "Brier Score": round(float(brier), 4),
            "Accuracy": round(float(acc), 4),
            "Precision (PIT)": round(float(prec), 4),
            "Recall (PIT)": round(float(rec), 4),
            "F1-Score": round(float(f1), 4),
        })

    return pd.DataFrame(records).set_index("Model")
