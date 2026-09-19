"""
src/opponent_model/baselines.py
===============================
Baseline models for Opponent Modeling benchmark comparison (Task 9, Phase A6).
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class AlwaysStayBaseline(BaseEstimator, ClassifierMixin):
    """Baseline that always predicts STAY with high probability."""

    def __init__(self, p_stay: float = 0.95):
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


class EmpiricalBaseline(BaseEstimator, ClassifierMixin):
    """Baseline predicting the empirical prior probability observed in training data."""

    def __init__(self):
        self.p_pit = 0.05
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        y_arr = np.asarray(y)
        self.p_pit = float(np.mean(y_arr))
        return self

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1.0 - self.p_pit), np.full(n, self.p_pit)])

    def predict(self, X):
        return (np.full(len(X), self.p_pit) >= 0.5).astype(int)


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
