"""
src/strategy_engine/evaluator.py
================================
Component 3: Strategy Evaluator.

Converts stochastic Monte Carlo outcome distributions into normalized,
multi-criteria utility scores and risk assessments across the 5 strategic
risk profiles (Balanced, Aggressive, Conservative, Defensive, Opportunistic).

Strict Architectural Boundary:
- The Evaluator scores and assesses candidate strategies across multi-criteria metrics.
- It does NOT select, recommend, rank as a decision, or choose the final strategy.
- The future Decision Engine will consume these evaluated alternatives for strategic/action selection.
"""

from __future__ import annotations

import math
from typing import Any, Optional
import numpy as np

from .types import (
    RiskProfile,
    SimulationResult,
    Strategy,
    StrategyEngineConfig,
    StrategyEvaluation,
)


class StrategyEvaluator:
    """
    Multi-criteria Strategy Evaluator applying configurable risk profiles
    to Monte Carlo outcome distributions.
    """

    def __init__(
        self,
        config: StrategyEngineConfig | None = None,
        default_profile: str = "balanced",
    ):
        self.config = config or StrategyEngineConfig()
        self.default_profile = default_profile
        self.risk_profiles = self.config.risk_profiles

    def get_profile(self, profile_name: str | None) -> RiskProfile:
        """Resolve a risk profile by name (case-insensitive) with fallback to default."""
        p_name = (profile_name or self.default_profile).lower().strip()
        if p_name in self.risk_profiles:
            return self.risk_profiles[p_name]
        return self.risk_profiles.get(self.default_profile, list(self.risk_profiles.values())[0])

    def evaluate_strategy(
        self,
        strategy: Strategy,
        sim_result: SimulationResult,
        profile_name: str | None = None,
        min_race_time: float | None = None,
        max_race_time: float | None = None,
    ) -> StrategyEvaluation:
        """
        Evaluate a single candidate strategy against the active risk profile.
        Packages the simulation result into a typed StrategyEvaluation alternative
        without computing composite scores or ranking.
        """
        profile = self.get_profile(profile_name)
        return StrategyEvaluation(
            strategy=strategy,
            simulation_result=sim_result,
            risk_profile_used=profile.name,
        )

    def evaluate_all(
        self,
        strategies: list[Strategy],
        sim_results: dict[str, SimulationResult],
        profile_name: str | None = None,
    ) -> list[StrategyEvaluation]:
        """
        Evaluate an entire candidate set of simulation results under the active profile.
        Returns evaluated strategy alternatives in deterministic neutral order (S01, S02, ...).
        Does NOT sort by score, expected position, or any preference metric.
        """
        if not strategies or not sim_results:
            return []

        evaluations: list[StrategyEvaluation] = []
        for strat in strategies:
            if strat.strategy_id not in sim_results:
                continue
            sim_res = sim_results[strat.strategy_id]
            ev = self.evaluate_strategy(
                strategy=strat,
                sim_result=sim_res,
                profile_name=profile_name,
            )
            evaluations.append(ev)

        # Deterministic neutral candidate order (S01, S02, ..., S09)
        evaluations.sort(key=lambda e: e.strategy.strategy_id)
        return evaluations
