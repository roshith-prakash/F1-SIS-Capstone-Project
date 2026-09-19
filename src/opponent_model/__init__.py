"""
F1 Strategy Intelligence System (F1-SIS) - Opponent Modeling Module
===================================================================
Produces calibrated per-lap probabilities P(PIT | state_t) and P(STAY | state_t)
for opponent drivers during an F1 race to feed into the Monte Carlo Strategy Engine.
"""

from .state import OpponentStateVector, build_opponent_state
from .model import OpponentModel
from .interface import predict_opponent, get_default_opponent_model, explain_prediction
from .explainability import explain_opponent_prediction, get_global_feature_importance
from .bayesian import BayesianPitUpdater
from .baselines import (
    MajorityClassBaseline,
    HistoricalFrequencyBaseline,
    StateConditionedBaseline,
    AlwaysStayBaseline,
    AlwaysPitBaseline,
    EmpiricalBaseline,
    evaluate_all_baselines,
)
from .mc_interface import (
    MonteCarloOpponentInterface,
    run_mc_sensitivity_experiment,
    run_multi_horizon_mc_experiment,
)
from .race_replay import (
    simulate_race_with_opponent_model,
    render_opponent_race_dashboard_html,
    plot_race_opponent_analysis,
    get_driver_race_telemetry,
    render_driver_opponent_card_html,
    plot_single_driver_opponent_analysis,
)

__all__ = [
    "OpponentStateVector",
    "build_opponent_state",
    "OpponentModel",
    "predict_opponent",
    "explain_prediction",
    "get_default_opponent_model",
    "explain_opponent_prediction",
    "get_global_feature_importance",
    "BayesianPitUpdater",
    "MajorityClassBaseline",
    "HistoricalFrequencyBaseline",
    "StateConditionedBaseline",
    "AlwaysStayBaseline",
    "AlwaysPitBaseline",
    "EmpiricalBaseline",
    "evaluate_all_baselines",
    "MonteCarloOpponentInterface",
    "run_mc_sensitivity_experiment",
    "run_multi_horizon_mc_experiment",
    "simulate_race_with_opponent_model",
    "render_opponent_race_dashboard_html",
    "plot_race_opponent_analysis",
    "get_driver_race_telemetry",
    "render_driver_opponent_card_html",
    "plot_single_driver_opponent_analysis",
]
