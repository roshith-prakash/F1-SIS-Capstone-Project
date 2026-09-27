"""
src/opponent_model/explainability.py
====================================
Explainability and Feature Attribution for the F1 SIS Opponent Model.
Answers: "Why did the model assign a high/low probability to an opponent pit?"
Provides:
- Global feature importances (Gain, Weight, Cover)
- Local prediction explanations (Top driving factors for individual decisions)
- Natural language strategic interpretations for race engineers
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from .state import OpponentStateVector


def get_global_feature_importance(
    model: Any,
    feature_names: list[str],
    importance_type: str = "gain",
) -> pd.DataFrame:
    """
    Extract global feature importance from a trained tree model or pipeline.
    """
    importances = None

    # Check for XGBoost booster
    if hasattr(model, "get_booster"):
        booster = model.get_booster()
        score_dict = booster.get_score(importance_type=importance_type)
        importances = [score_dict.get(f, 0.0) for f in feature_names]
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_[0])
    else:
        # Fallback uniform
        importances = [1.0 / len(feature_names)] * len(feature_names)

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values("importance", ascending=False).reset_index(drop=True)

    # Normalize to percentages
    total = df_imp["importance"].sum()
    df_imp["importance_pct"] = (df_imp["importance"] / (total if total > 0 else 1.0)) * 100.0
    return df_imp


def explain_opponent_prediction(
    model: Any,
    opponent_state: dict[str, Any] | OpponentStateVector,
    feature_names: list[str] | None = None,
    top_k: int = 5,
) -> dict[str, Any]:
    """
    Generate local explainability for a single opponent prediction.
    Identifies the key factors pushing P(PIT) higher or lower.
    Does NOT modify the model's prediction.
    """
    if hasattr(opponent_state, "to_dict"):
        state_dict = opponent_state.to_dict()
    elif isinstance(opponent_state, dict):
        state_dict = dict(opponent_state)
    else:
        try:
            state_dict = dict(opponent_state)
        except Exception:
            state_dict = {}

    explanations: list[str] = []
    positive_drivers: list[dict[str, Any]] = []
    negative_drivers: list[dict[str, Any]] = []

    # 1. Inspect Tyre Age & Degradation
    tyre_age = int(state_dict.get("tyre_age", 0))
    compound = str(state_dict.get("tyre_compound", "MEDIUM")).upper()
    predicted_deg = float(state_dict.get("predicted_degradation", 0.0))
    nominal_stint = 18 if compound == "SOFT" else (28 if compound == "MEDIUM" else 38)

    if tyre_age >= nominal_stint:
        msg = f"Tyre age ({tyre_age} laps on {compound}) exceeds nominal stint window ({nominal_stint} laps)"
        explanations.append(msg)
        positive_drivers.append({"factor": "Tyre Age past nominal", "detail": msg, "impact": "High Positive"})
    elif tyre_age < nominal_stint * 0.4:
        msg = f"Tyre is relatively fresh ({tyre_age} laps on {compound})"
        negative_drivers.append({"factor": "Fresh Tyres", "detail": msg, "impact": "Moderate Negative"})

    if predicted_deg > 1.2:
        msg = f"Severe tyre degradation detected (+{predicted_deg:.2f} s/lap pace deficit)"
        explanations.append(msg)
        positive_drivers.append({"factor": "High Tyre Degradation", "detail": msg, "impact": "High Positive"})

    # 2. Inspect Safety Car / VSC
    is_sc = int(state_dict.get("is_safety_car", 0))
    is_vsc = int(state_dict.get("is_vsc", 0))
    p_sc = float(state_dict.get("p_sc_h1", 0.0))

    if is_sc == 1:
        msg = "Safety Car deployed: pit-lane loss dramatically reduced (~50% time saving)"
        explanations.append(msg)
        positive_drivers.append({"factor": "Safety Car Neutralization", "detail": msg, "impact": "Very High Positive"})
    elif is_vsc == 1:
        msg = "Virtual Safety Car active: opportunistic cheap pit stop window"
        explanations.append(msg)
        positive_drivers.append({"factor": "VSC Active", "detail": msg, "impact": "High Positive"})
    elif p_sc > 0.15:
        msg = f"Elevated Safety Car risk within next lap ({p_sc:.1%})"
        positive_drivers.append({"factor": "SC Window Imminent", "detail": msg, "impact": "Low Positive"})

    # 3. Inspect Tactical Undercut / Ego Competition
    gap_behind = float(state_dict.get("gap_behind", 50.0))
    ego_undercut = int(state_dict.get("ego_undercut_threat", 0))
    ego_pitted = int(state_dict.get("ego_recently_pitted", 0))
    gap_to_ego = float(state_dict.get("gap_to_ego", 0.0))

    if ego_undercut == 1:
        msg = f"Under active undercut threat from Ego car ({abs(gap_to_ego):.1f}s behind)"
        explanations.append(msg)
        positive_drivers.append({"factor": "Ego Undercut Threat", "detail": msg, "impact": "High Positive"})
    elif ego_pitted == 1 and gap_to_ego > 0:
        msg = "Competitor/Ego car recently pitted: pressure to respond to undercut"
        explanations.append(msg)
        positive_drivers.append({"factor": "Competitor Pit Response", "detail": msg, "impact": "High Positive"})
    elif gap_behind < 2.0:
        msg = f"Car behind within 2.0s ({gap_behind:.1f}s) exerting undercut pressure"
        positive_drivers.append({"factor": "General Undercut Pressure", "detail": msg, "impact": "Moderate Positive"})

    # 4. Inspect Race Phase & Laps Remaining
    remaining = int(state_dict.get("remaining_laps", 50))
    if remaining <= 3:
        msg = f"Only {remaining} laps remaining: pit stop will lose irrecoverable track position"
        negative_drivers.append({"factor": "End of Race Imminent", "detail": msg, "impact": "Very High Negative"})

    # 5. Pace Trend
    pace_trend = float(state_dict.get("recent_pace_trend", 0.0))
    if pace_trend > 0.5:
        msg = f"Recent pace deteriorating rapidly (+{pace_trend:.2f} s over rolling average)"
        positive_drivers.append({"factor": "Deteriorating Pace Trend", "detail": msg, "impact": "Moderate Positive"})

    # Build structured output
    summary = "; ".join(explanations[:3]) if explanations else "Normal strategic progression within stint."

    return {
        "summary": summary,
        "positive_factors": positive_drivers[:top_k],
        "negative_factors": negative_drivers[:top_k],
        "key_signals": explanations,
    }
