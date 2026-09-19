"""
src/opponent_model/bayesian.py
==============================
Bayesian Online Updating Layer for Opponent Pit Probability (Task 11, 12 & Phase A11).

Statistically sound formulation:
Avoids double-counting static features (tyre age, compound) that the offline ML model
has already absorbed. Instead, conditions the online likelihood on real-time evidence
observed during the live race:
1. Pace Degradation Residual Anomaly:
   epsilon_t = last_lap_time - opponent_predicted_lap_time
   (Measures unexpected tyre cliff / severe blistering above foundational model prediction)
2. Live Stint Extension Deviation:
   Delta_stint = max(0, tyre_age - mu_nominal)
   (Measures anomalous stint stretching beyond historical window)

When epsilon_t = 0 and within nominal window: L = 0.5 (logit(L) = 0), so P_post = P_ML.
"""

from __future__ import annotations

import numpy as np
from scipy.special import expit, logit


class BayesianPitUpdater:
    """
    Bayesian updater for live race P(PIT) predictions.

    Prior: P(PIT) from calibrated offline ML model
    Evidence: Real-time pace degradation residual anomaly and live stint extension
    Likelihood: L(PIT | epsilon_t, Delta_stint) = sigmoid(epsilon_t / sigma_eps + 0.5 * Delta_stint / sigma_stint)
    Posterior: logit(P_post) = logit(P_ML) + lambda * logit(L)
    """

    # Historical stint length distributions (mu, sigma)
    COMPOUND_PARAMS: dict[str, tuple[float, float]] = {
        "SOFT": (18.0, 4.0),
        "MEDIUM": (28.0, 6.0),
        "HARD": (38.0, 8.0),
    }
    DEFAULT_PARAMS: tuple[float, float] = (28.0, 6.0)

    def __init__(self, lambda_: float = 0.3, eps: float = 1e-6, sigma_residual: float = 1.2):
        self.lambda_ = lambda_
        self.eps = eps
        self.sigma_residual = sigma_residual

    def compute_likelihood(
        self,
        pace_residual: float = 0.0,
        tyre_age: float | None = None,
        compound: str = "MEDIUM",
        laps_past_nominal: float = 0.0,
    ) -> float:
        """
        Compute likelihood based on live online evidence NOT already encoded in static ML.

        Primary evidence: pace degradation residual anomaly (last_lap_time - predicted_lap_time).
        When pace_residual == 0 and not past nominal window: returns 0.5 (neutral evidence, logit=0).
        """
        # 1. Pace degradation residual anomaly (cliff / graining detection)
        z_residual = float(pace_residual) / max(0.1, self.sigma_residual)

        # 2. Live stint extension beyond nominal window
        clean_compound = str(compound).strip().upper()
        mu, sigma = self.COMPOUND_PARAMS.get(clean_compound, self.DEFAULT_PARAMS)

        if tyre_age is not None:
            excess_laps = max(0.0, float(tyre_age) - mu)
        else:
            excess_laps = max(0.0, float(laps_past_nominal))

        z_excess = excess_laps / max(1.0, sigma)

        # Combined online evidence score
        z = z_residual + 0.5 * z_excess
        return float(expit(z))

    def update(
        self,
        p_ml: float,
        tyre_age: int | float = 0,
        compound: str = "MEDIUM",
        lambda_: float | None = None,
        pace_residual: float = 0.0,
        laps_past_nominal: float = 0.0,
    ) -> float:
        """
        Compute posterior P(PIT) by updating ML prior with live online evidence.
        Guarantees that when pace_residual = 0 and tyre_age <= mu_nominal,
        L = 0.5, logit(L) = 0, and P_posterior = P_ML (zero double-counting).
        """
        lmb = self.lambda_ if lambda_ is None else lambda_

        # Clamp ML prior to avoid inf in logit
        p_clamped = max(self.eps, min(1.0 - self.eps, float(p_ml)))
        p_lik = max(
            self.eps,
            min(
                1.0 - self.eps,
                self.compute_likelihood(
                    pace_residual=pace_residual,
                    tyre_age=tyre_age,
                    compound=compound,
                    laps_past_nominal=laps_past_nominal,
                ),
            ),
        )

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
        pace_residuals: np.ndarray | list[float] | None = None,
        lambda_: float | None = None,
    ) -> np.ndarray:
        """Vectorized update across multiple predictions."""
        lmb = self.lambda_ if lambda_ is None else lambda_
        p_arr = np.clip(np.asarray(p_ml_arr, dtype=float), self.eps, 1.0 - self.eps)
        ages = np.asarray(tyre_ages, dtype=float)
        resids = np.zeros(len(p_arr)) if pace_residuals is None else np.asarray(pace_residuals, dtype=float)

        p_liks = [
            self.compute_likelihood(pace_residual=res, tyre_age=age, compound=comp)
            for res, age, comp in zip(resids, ages, compounds)
        ]
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
        pace_residuals_val: np.ndarray | None = None,
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
            p_post = self.update_batch(
                p_ml_arr=p_ml_val,
                tyre_ages=tyre_ages_val,
                compounds=compounds_val,
                pace_residuals=pace_residuals_val,
                lambda_=lmb,
            )
            brier = float(np.mean((p_post - y_true) ** 2))
            scores[lmb] = brier
            if brier < best_score:
                best_score = brier
                best_lmb = lmb

        self.lambda_ = best_lmb
        return best_lmb, scores
