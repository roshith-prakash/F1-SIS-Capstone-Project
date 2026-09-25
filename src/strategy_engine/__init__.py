"""
src/strategy_engine/__init__.py
===============================
F1 Strategic AI — Strategy Engine Subsystem.

Evaluates future strategy plans from the current race state, producing
probabilistic outcome distributions and multi-criteria rankings.

Components:
- CandidateStrategyGenerator (candidate_generator.py)
- RaceScenarioSimulator (simulator.py)
- StrategyEvaluator (evaluator.py)
- StrategyResultFormatter (ranking.py)
- StrategyEngine (engine.py)
"""

from .types import (
    AssumptionSource,
    RiskProfile,
    RolloutOutcome,
    SimulationResult,
    StintPlan,
    Strategy,
    StrategyEngineConfig,
    StrategyEngineResult,
    StrategyEvaluation,
)
from .candidate_generator import CandidateStrategyGenerator
from .simulator import RaceScenarioSimulator
from .evaluator import StrategyEvaluator
from .ranking import StrategyResultFormatter, StrategyResultRanker
from .engine import StrategyEngine

__all__ = [
    "AssumptionSource",
    "CandidateStrategyGenerator",
    "RaceScenarioSimulator",
    "RiskProfile",
    "RolloutOutcome",
    "SimulationResult",
    "StintPlan",
    "Strategy",
    "StrategyEngine",
    "StrategyEngineConfig",
    "StrategyEngineResult",
    "StrategyEvaluation",
    "StrategyEvaluator",
    "StrategyResultFormatter",
    "StrategyResultRanker",
]
