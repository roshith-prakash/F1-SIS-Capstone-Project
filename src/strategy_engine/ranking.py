"""
src/strategy_engine/ranking.py
==============================
Component 4: Strategy Result Formatter & Downstream Interface.

Processes evaluated candidate strategies into a clean, typed StrategyEngineResult
payload for downstream consumption by the Decision Engine.

Strict Architectural Boundary:
- The Strategy Engine evaluates feasible strategies and produces simulated outcome distributions.
- It does NOT select, recommend, rank, or declare a "best" strategy.
- The Decision Engine will consume these evaluated alternatives and make the strategic/action decision.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

try:
    from race_state.models import RaceState
except ImportError:
    from src.race_state.models import RaceState

from .types import StrategyEngineConfig, StrategyEngineResult, StrategyEvaluation


class StrategyResultFormatter:
    """
    Formats evaluated candidate strategies into structured result payloads
    for the downstream Decision Engine without selecting or recommending a strategy.
    """

    def __init__(self, config: StrategyEngineConfig | None = None):
        self.config = config or StrategyEngineConfig()

    def format_result(
        self,
        evaluations: list[StrategyEvaluation],
        state: RaceState,
        ego_driver: str,
        horizon_laps: int,
        profile_name: str,
        n_rollouts: int,
        seed: int = 42,
    ) -> StrategyEngineResult:
        """
        Assemble evaluated strategy candidates into a typed StrategyEngineResult.
        """
        if not evaluations:
            raise ValueError("Cannot assemble result from empty evaluation list.")

        curr_lap = state.current_lap or 1
        timestamp_str = datetime.now().isoformat()

        total_laps = getattr(state, "total_laps_expected", curr_lap + horizon_laps)
        explanation = (
            f"Evaluated {len(evaluations)} feasible strategy candidates for {ego_driver} "
            f"over full remaining-race horizon of {horizon_laps} laps ({n_rollouts} rollouts/strategy, seed={seed}) "
            f"through race completion (Lap {total_laps}) under the '{profile_name}' risk profile. "
            f"Simulated outcome distributions and risk metrics compiled for Decision Engine selection."
        )

        metadata = {
            "race_id": getattr(state, "race_id", "LIVE_SESSION"),
            "grand_prix": getattr(state, "grand_prix", "Grand Prix"),
            "current_lap": curr_lap,
            "total_laps": getattr(state, "total_laps_expected", 57),
            "ego_driver": ego_driver,
            "profile_used": profile_name,
            "rollouts_per_candidate": n_rollouts,
            "horizon_laps": horizon_laps,
            "candidate_count": len(evaluations),
            "architectural_boundary_notice": (
                "The Strategy Engine evaluates feasible strategy plans and produces simulated "
                "outcome distributions. The Decision Engine consumes these alternatives and makes "
                "the strategic and tactical choice."
            ),
        }

        simulation_metadata = {
            "rollouts_per_strategy": n_rollouts,
            "horizon_laps": horizon_laps,
            "seed": seed,
            "risk_profile": profile_name,
            "candidate_count": len(evaluations),
        }

        return StrategyEngineResult(
            timestamp=timestamp_str,
            current_lap=curr_lap,
            ego_driver=ego_driver,
            horizon_laps=horizon_laps,
            n_rollouts_per_strategy=n_rollouts,
            risk_profile=profile_name,
            candidate_count=len(evaluations),
            strategies=evaluations,
            explanation_summary=explanation,
            simulation_metadata=simulation_metadata,
            metadata=metadata,
        )

    # Backwards-compatibility alias
    rank_strategies = format_result


# Backwards-compatibility alias
StrategyResultRanker = StrategyResultFormatter
