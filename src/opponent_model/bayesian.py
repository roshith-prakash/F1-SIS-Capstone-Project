"""
src/opponent_model/bayesian.py
==============================
Bayesian Online Updating Layer for Opponent Pit Probability (Task 20, Phase A11).

Blends calibrated ML prior P_0(PIT) with race-specific likelihood based on tyre age
and compound degradation characteristics using a log-odds formulation.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


class BayesianPitUpdater:
    """
    Bayesian updater for live race P(PIT) predictions.

    Prior: P(PIT) from calibrated ML model
    Likelihood: L(PIT | TyreAge) = sigmoid((TyreAge - mu) / sigma)
    Posterior: logit(P_post) = logit(P_ML) + lambda * logit(L)
    """

    # Historical stint length distributions (mu, sigma)
    COMPOUND_PARAMS: dict[str, tuple[float, float]] = {
        "SOFT": (18.0, 4.0),
        "MEDIUM": (28.0, 6.0),
        "HARD": (38.0, 8.0),
    }
    DEFAULT_PARAMS: tuple[float, float] = (28.0, 6.0)

    def __init__(self, lambda_: float = 0.3, eps: float = 1e-6):
        self.lambda_ = lambda_
        self.eps = eps

    def compute_likelihood(self, tyre_age: float, compound: str) -> float:
        """Compute Gaussian-based likelihood L(PIT | TyreAge, compound)."""
        clean_compound = str(compound).strip().upper()
        mu, sigma = self.COMPOUND_PARAMS.get(clean_compound, self.DEFAULT_PARAMS)
        z = (float(tyre_age) - mu) / max(0.1, sigma)
        return float(expit(z))

    def update(
        self,
        p_ml: float,
        tyre_age: int | float,
        compound: str,
        lambda_: float | None = None,
    ) -> float:
        """
        Compute posterior P(PIT) by updating ML prior with likelihood.
        """
        lmb = self.lambda_ if lambda_ is None else lambda_

        # Clamp ML prior to avoid inf in logit
        p_clamped = max(self.eps, min(1.0 - self.eps, float(p_ml)))
        p_lik = max(self.eps, min(1.0 - self.eps, self.compute_likelihood(tyre_age, compound)))

        logit_prior = float(logit(p_clamped))
        logit_lik = float(logit(p_lik))

        logit_posterior = logit_prior + lmb * logit_lik
        p_posterior = float(expit(logit_posterior))

        # Clamp output
        return max(self.eps, min(1.0 - self.eps, p_posterior))

    def update_batch(
        self,
        p_ml_arr: np.ndarray,
        tyre_ages: np.ndarray | list[int | float],
        compounds: list[str],
        lambda_: float | None = None,
    ) -> np.ndarray:
        """Vectorized update across multiple predictions."""
        lmb = self.lambda_ if lambda_ is None else lambda_
        p_arr = np.clip(np.asarray(p_ml_arr, dtype=float), self.eps, 1.0 - self.eps)
        ages = np.asarray(tyre_ages, dtype=float)

        p_liks = []
        for age, comp in zip(ages, compounds):
            p_liks.append(self.compute_likelihood(age, comp))
        lik_arr = np.clip(np.asarray(p_liks, dtype=float), self.eps, 1.0 - self.eps)

        logit_prior = logit(p_arr)
        logit_lik = logit(lik_arr)
        logit_post = logit_prior + lmb * logit_lik

        return np.clip(expit(logit_post), self.eps, 1.0 - self.eps)

    def tune_lambda(
        self,
        p_ml_val: np.ndarray,
        tyre_ages_val: np.ndarray,
        compounds_val: list[str],
        y_val: np.ndarray,
        candidates: list[float] | None = None,
    ) -> tuple[float, dict[float, float]]:
        """
        Tune blending weight lambda on validation set to minimize Brier Score.
        Returns: (best_lambda, score_dict)
        """
        if candidates is None:
            candidates = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]

        best_score = float("inf")
        best_lmb = 0.3
        scores = {}

        y_true = np.asarray(y_val, dtype=float)
        for lmb in candidates:
            p_post = self.update_batch(p_ml_val, tyre_ages_val, compounds_val, lambda_=lmb)
            brier = float(np.mean((p_post - y_true) ** 2))
            scores[lmb] = brier
            if brier < best_score:
                best_score = brier
                best_lmb = lmb

        self.lambda_ = best_lmb
        return best_lmb, scores
