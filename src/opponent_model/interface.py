"""
src/opponent_model/interface.py
===============================
Standardized prediction interface connecting the Opponent Model to the
Strategy Engine and Monte Carlo Simulator (Task 14 & 15).

Strict architectural boundary:
Opponent Model answers "What is the opponent most likely to do next?"
It does NOT call RL decision logic or choose our car's strategy.
"""

from __future__ import annotations

from typing import Any, Optional
import numpy as np

from .state import OpponentStateVector, build_opponent_state
from .model import OpponentModel
from .explainability import explain_opponent_prediction

try:
    from race_state.models import RaceState
except ImportError:
    from src.race_state.models import RaceState

# Singleton model instance for fast repeated calls
_DEFAULT_MODEL: Optional[OpponentModel] = None


def get_default_opponent_model() -> OpponentModel:
    """Retrieve or initialize the default cached OpponentModel instance."""
    global _DEFAULT_MODEL
    if _DEFAULT_MODEL is None:
        try:
            _DEFAULT_MODEL = OpponentModel.load()
        except Exception:
            _DEFAULT_MODEL = OpponentModel()
    return _DEFAULT_MODEL


def predict_opponent(
    state: RaceState | OpponentStateVector | dict[str, Any],
    opponent_driver: str | None = None,
    ego_driver: str | None = None,
    model: OpponentModel | None = None,
    horizons: list[int] | None = None,
    include_explanation: bool = True,
    lt_adapter: Any = None,
    tyre_adapter: Any = None,
    sc_adapter: Any = None,
) -> dict[str, Any]:
    """
    Standardized Strategy Engine Interface.
    
    Given the current race state, returns the probabilistic prediction of the
    opponent's next action across single and multi-lap horizons.

    Parameters
    ----------
    state : RaceState, OpponentStateVector, or dict
        Current race state representation.
    opponent_driver : str, optional
        3-letter code of opponent driver (e.g. "PIA", "NOR", "LEC").
        Required if `state` is a RaceState object.
    ego_driver : str, optional
        3-letter code of our car (e.g. "VER", "HAM").
    model : OpponentModel, optional
        OpponentModel instance. If None, uses default loaded model.
    horizons : list of int, optional
        Horizons to predict (default [1, 2, 3, 4, 5]).
    include_explanation : bool, optional
        Whether to generate feature importance and natural language explanation.

    Returns
    -------
    dict
        Structured probabilistic opponent behavior consumption dictionary.
    """
    if horizons is None:
        horizons = [1, 2, 3, 4, 5]

    active_model = model or get_default_opponent_model()

    # 1. Resolve State Vector
    if hasattr(state, "to_dict"):
        vec = state
        opp_code = getattr(vec, "driver", str(opponent_driver or "OPP"))
        ego_code = getattr(vec, "ego_driver", str(ego_driver or ""))
    elif isinstance(state, dict):
        # Already a dictionary
        opp_code = str(opponent_driver or state.get("driver", "OPP")).upper()
        ego_code = str(ego_driver or state.get("ego_driver", "EGO")).upper()
        vec = state
    else:
        # Must be RaceState
        if not opponent_driver:
            raise ValueError("opponent_driver must be provided when state is a RaceState object.")
        opp_code = opponent_driver.upper().strip()
        ego_code = ego_driver.upper().strip() if ego_driver else ""
        vec = build_opponent_state(
            state=state,
            driver=opp_code,
            ego_driver=ego_code,
            lt_adapter=lt_adapter,
            tyre_adapter=tyre_adapter,
            sc_adapter=sc_adapter,
        )
        if vec is None:
            # Fallback for inactive/unknown opponent
            return {
                "opponent_driver": opp_code,
                "ego_driver": ego_code,
                "lap": getattr(state, "current_lap", 1),
                "pit_probability": 0.05,
                "stay_out_probability": 0.95,
                "predicted_actions": ["STAY", "PIT"],
                "confidence": 0.90,
                "horizon": 1,
                "multi_horizon": {f"lap_{h}": {"pit": 0.05, "stay_out": 0.95} for h in horizons},
                "status": "inactive_or_missing",
            }

    # 2. Model Prediction
    raw_pred = active_model.predict(vec)
    p_pit_1 = float(raw_pred["p_pit"])
    p_stay_1 = float(raw_pred["p_stay"])

    # 3. Multi-Horizon Predictions
    # Survival formulation: P(pit within h laps) = 1 - (1 - h_t)^h
    # Clearly labeled as a derived constant-hazard approximation (Task 13)
    multi_horizon = {}
    for h in horizons:
        p_pit_h = float(np.clip(1.0 - (1.0 - p_pit_1) ** h, 1e-4, 1.0 - 1e-4))
        multi_horizon[f"lap_{h}"] = {
            "pit": round(p_pit_h, 4),
            "stay_out": round(1.0 - p_pit_h, 4),
            "is_derived_approximation": True if h > 1 else False,
        }

    # 4. Extract Category Sub-States for Strategy Engine Context
    if hasattr(vec, "to_dict"):
        opp_physical = {
            "compound": getattr(vec, "tyre_compound", "MEDIUM"),
            "tyre_age": getattr(vec, "tyre_age", 0),
            "predicted_pace": round(getattr(vec, "predicted_lap_time", 90.0), 2),
            "predicted_degradation": round(getattr(vec, "predicted_degradation", 0.0), 2),
            "opponent_predicted_lap_time": round(getattr(vec, "predicted_lap_time", 90.0), 2),
            "opponent_predicted_degradation": round(getattr(vec, "predicted_degradation", 0.0), 2),
            "cost_of_staying": round(getattr(vec, "cost_of_staying", 0.0), 2),
        }
        ego_state = {
            "driver": getattr(vec, "ego_driver", ""),
            "position": getattr(vec, "ego_position", 0),
            "gap_to_opponent": round(getattr(vec, "gap_to_ego", 0.0), 2),
            "gap_opponent_to_ego": round(getattr(vec, "gap_to_ego", 0.0), 2),
            "compound": getattr(vec, "ego_compound", "MEDIUM"),
            "tyre_age": getattr(vec, "ego_tyre_age", 0),
            "ego_predicted_lap_time": round(getattr(vec, "ego_predicted_pace", 90.0), 2),
            "ego_predicted_degradation": round(getattr(vec, "ego_predicted_deg", 0.0), 2),
            "recently_pitted": getattr(vec, "ego_recently_pitted", 0),
            "undercut_threat": getattr(vec, "ego_undercut_threat", 0),
        }
        lap_num = getattr(vec, "lap_number", 1)
    else:
        opp_physical = {
            "compound": vec.get("tyre_compound", "MEDIUM"),
            "tyre_age": vec.get("tyre_age", 0),
            "predicted_pace": round(float(vec.get("predicted_lap_time", 90.0)), 2),
            "predicted_degradation": round(float(vec.get("predicted_degradation", 0.0)), 2),
            "opponent_predicted_lap_time": round(float(vec.get("predicted_lap_time", 90.0)), 2),
            "opponent_predicted_degradation": round(float(vec.get("predicted_degradation", 0.0)), 2),
            "cost_of_staying": round(float(vec.get("cost_of_staying", 0.0)), 2),
        }
        ego_state = {
            "driver": vec.get("ego_driver", ""),
            "position": vec.get("ego_position", 0),
            "gap_to_opponent": round(float(vec.get("gap_to_ego", 0.0)), 2),
            "gap_opponent_to_ego": round(float(vec.get("gap_to_ego", 0.0)), 2),
            "compound": vec.get("ego_compound", "MEDIUM"),
            "tyre_age": vec.get("ego_tyre_age", 0),
            "ego_predicted_lap_time": round(float(vec.get("ego_predicted_pace", 90.0)), 2),
            "ego_predicted_degradation": round(float(vec.get("ego_predicted_deg", 0.0)), 2),
            "recently_pitted": vec.get("ego_recently_pitted", 0),
            "undercut_threat": vec.get("ego_undercut_threat", 0),
        }
        lap_num = int(vec.get("lap_number", 1))

    # 5. Explainability
    explanation = None
    if include_explanation:
        explanation = explain_opponent_prediction(
            model=active_model.classifier,
            opponent_state=vec,
            feature_names=active_model.feature_names,
        )

    # 4. Multi-Horizon and Operational Alert Level
    p_pit_3 = float(np.clip(1.0 - (1.0 - p_pit_1) ** 3, 1e-4, 1.0 - 1e-4))
    p_pit_5 = float(np.clip(1.0 - (1.0 - p_pit_1) ** 5, 1e-4, 1.0 - 1e-4))
    alert_level = "LOW" if p_pit_1 < 0.05 else ("MEDIUM" if p_pit_1 < 0.15 else "HIGH")

    return {
        "p_pit_next": round(p_pit_1, 4),
        "p_stay_next": round(p_stay_1, 4),
        "p_pit_3": round(p_pit_3, 4),
        "p_pit_5": round(p_pit_5, 4),
        "pit_alert_level": alert_level,
        "opponent_id": opp_code,
        "opponent_driver": opp_code,
        "ego_driver": ego_code,
        "lap": lap_num,
        "model_version": "v1.0-calibrated",
        "pit_probability": round(p_pit_1, 4),
        "stay_out_probability": round(p_stay_1, 4),
        "predicted_actions": ["STAY", "PIT"],
        "confidence": round(float(raw_pred.get("confidence", 0.5)), 4),
        "horizon": 1,
        "multi_horizon": multi_horizon,
        "opponent_physical_state": opp_physical,
        "ego_state": ego_state,
        "explanation": explanation,
    }


def explain_prediction(
    state: RaceState | OpponentStateVector | dict[str, Any],
    opponent_driver: str | None = None,
    ego_driver: str | None = None,
    model: OpponentModel | None = None,
    **kwargs,
) -> dict[str, Any]:
    """
    Explain opponent prediction using the exact same feature vector and probability (Section 19).
    Guarantees consistency between prediction and explanation.
    """
    pred = predict_opponent(
        state=state,
        opponent_driver=opponent_driver,
        ego_driver=ego_driver,
        model=model,
        include_explanation=True,
        **kwargs,
    )
    result = dict(pred)
    result["pit_probability"] = pred["pit_probability"]
    result["p_pit_next"] = pred["p_pit_next"]
    return result
