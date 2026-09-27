"""
src/strategy_engine/engine.py
=============================
Strategy Engine Orchestration Facade.

Integrates the 4 decoupled components:
1. Candidate Strategy Generator
2. Race Scenario Simulator
3. Strategy Evaluator
4. Strategy Result Formatter & Downstream Interface

Provides the primary public API:
    StrategyEngine.evaluate_race_state(
        state, ego_driver, horizon_laps=10, n_rollouts=150, profile="balanced", seed=42
    ) -> StrategyEngineResult
"""

from __future__ import annotations

from typing import Any, Optional
from pathlib import Path

try:
    from race_state.models import RaceState, normalize_driver
except ImportError:
    from src.race_state.models import RaceState, normalize_driver

from .candidate_generator import CandidateStrategyGenerator
from .evaluator import StrategyEvaluator
from .ranking import StrategyResultFormatter, StrategyResultRanker
from .simulator import RaceScenarioSimulator
from .types import (
    SimulationResult,
    Strategy,
    StrategyEngineConfig,
    StrategyEngineResult,
    StrategyEvaluation,
)


class StrategyEngine:
    """
    Main Facade coordinating candidate generation, Monte Carlo simulation,
    multi-criteria evaluation, and downstream payload assembly.
    """

    def __init__(
        self,
        config: StrategyEngineConfig | None = None,
        candidate_generator: CandidateStrategyGenerator | None = None,
        simulator: RaceScenarioSimulator | None = None,
        evaluator: StrategyEvaluator | None = None,
        formatter: StrategyResultFormatter | None = None,
        ranker: StrategyResultRanker | None = None,
        lap_time_adapter: Any = None,
        tyre_deg_adapter: Any = None,
        sc_risk_adapter: Any = None,
        opponent_interface: Any = None,
        overtake_adapter: Any = None,
        pitstop_adapter: Any = None,
        default_horizon_laps: int | None = None,
        default_rollouts: int | None = None,
        default_profile: str = "balanced",
        random_seed: int = 42,
    ):
        self.config = config or StrategyEngineConfig()
        self.default_horizon = default_horizon_laps or self.config.default_horizon_laps
        self.default_rollouts = default_rollouts or self.config.default_n_rollouts
        self.default_profile = default_profile
        self.random_seed = random_seed
        self.pitstop_adapter = pitstop_adapter

        # Component 1: Candidate Generator
        self.candidate_generator = candidate_generator or CandidateStrategyGenerator(config=self.config)

        # Component 2: Simulator
        if simulator is not None:
            self.simulator = simulator
        else:
            self.simulator = RaceScenarioSimulator(
                config=self.config,
                lap_time_adapter=lap_time_adapter,
                tyre_deg_adapter=tyre_deg_adapter,
                sc_risk_adapter=sc_risk_adapter,
                opponent_interface=opponent_interface,
                overtake_adapter=overtake_adapter,
                pitstop_adapter=pitstop_adapter,
                default_rollouts=self.default_rollouts,
                random_seed=self.random_seed,
            )

        # Component 3: Evaluator
        self.evaluator = evaluator or StrategyEvaluator(
            config=self.config,
            default_profile=self.default_profile,
        )

        # Component 4: Result Formatter
        self.formatter = formatter or ranker or StrategyResultFormatter(config=self.config)
        self.ranker = self.formatter

    @classmethod
    def load_with_default_models(
        cls,
        config: StrategyEngineConfig | None = None,
        default_profile: str = "balanced",
        random_seed: int = 42,
    ) -> "StrategyEngine":
        """
        Factory method: Lazily loads and wires the actual repository foundational
        models and opponent model into the Strategy Engine.
        """
        cfg = config or StrategyEngineConfig()

        # Load Lap Time Adapter
        lt_adapter = None
        try:
            from lap_time.adapter import LapTimeAdapter
            lt_adapter = LapTimeAdapter()
        except Exception:
            pass

        # Load Tyre Deg Adapter
        tyre_adapter = None
        try:
            from tyre_deg.adapter import TyreDegAdapter
            tyre_adapter = TyreDegAdapter()
        except Exception:
            pass

        # Load SC Risk Adapter
        sc_adapter = None
        try:
            from sc_risk.adapter import SCRiskAdapter
            sc_prior_path = Path("models") / "SC Estimation" / "sc_vsc_historical_prior.csv"
            if sc_prior_path.exists():
                sc_adapter = SCRiskAdapter(str(sc_prior_path))
        except Exception:
            pass

        # Load Opponent Model Interface
        opp_interface = None
        try:
            from opponent_model.model import OpponentModel
            from opponent_model.mc_interface import MonteCarloOpponentInterface

            try:
                opp_model = OpponentModel.load()
            except Exception:
                opp_model = OpponentModel()
            opp_interface = MonteCarloOpponentInterface(opp_model)
        except Exception:
            pass

        # Load Overtake Adapter
        overtake_adapter = None
        try:
            from overtake.adapter import OvertakeAdapter
            overtake_adapter = OvertakeAdapter()
        except Exception:
            try:
                from src.overtake.adapter import OvertakeAdapter
                overtake_adapter = OvertakeAdapter()
            except Exception:
                pass

        # Load Pitstop Adapter
        pitstop_adapter = None
        try:
            from pitstop.adapter import PitstopAdapter
            pitstop_adapter = PitstopAdapter()
        except Exception:
            try:
                from src.pitstop.adapter import PitstopAdapter
                pitstop_adapter = PitstopAdapter()
            except Exception:
                pass

        return cls(
            config=cfg,
            lap_time_adapter=lt_adapter,
            tyre_deg_adapter=tyre_adapter,
            sc_risk_adapter=sc_adapter,
            opponent_interface=opp_interface,
            overtake_adapter=overtake_adapter,
            pitstop_adapter=pitstop_adapter,
            default_profile=default_profile,
            random_seed=random_seed,
        )

    def evaluate_race_state(
        self,
        state: RaceState,
        ego_driver: str,
        horizon_laps: int | None = None,
        n_rollouts: int | None = None,
        profile: str | None = None,
        seed: int | None = None,
    ) -> StrategyEngineResult:
        """
        Primary Strategy Engine Entry Point.
        
        Given the current committed RaceState and our driver, evaluates all
        feasible future candidate strategies through stochastic Monte Carlo
        rollouts and produces simulated outcome and risk distributions for each alternative.

        Parameters
        ----------
        state : RaceState
            Current committed race snapshot at lap t.
        ego_driver : str
            3-letter driver code for our car (e.g. "VER", "NOR").
        horizon_laps : int, optional
            Simulation horizon in laps. Defaults to full remaining race:
            (state.total_laps_expected - state.current_lap).
        n_rollouts : int, optional
            Monte Carlo rollouts per candidate strategy (default 150).
        profile : str, optional
            Risk profile: "balanced", "aggressive", "conservative", "defensive", "opportunistic".
        seed : int, optional
            Random seed for deterministic reproducibility.

        Returns
        -------
        StrategyEngineResult
            Structured typed result with evaluated candidate strategies, performance &
            risk distributions, and downstream Decision Engine payload.
        """
        curr_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        remaining_laps = max(1, total_laps - curr_lap)

        if horizon_laps is not None:
            h_laps = horizon_laps
        elif self.default_horizon is not None:
            h_laps = self.default_horizon
        else:
            h_laps = remaining_laps

        rolls = n_rollouts or self.default_rollouts
        active_profile = profile or self.default_profile
        active_seed = seed if seed is not None else self.random_seed

        # Step 1: Generate Feasible Candidate Strategies
        candidates = self.candidate_generator.generate_candidates(
            state=state,
            ego_driver=ego_driver,
            horizon_laps=h_laps,
        )

        if not candidates:
            raise RuntimeError(f"No valid candidate strategies could be generated for driver {ego_driver}.")

        # Step 2: Simulate Candidates with Monte Carlo Rollouts
        sim_results = self.simulator.simulate_all(
            state=state,
            ego_driver=ego_driver,
            strategies=candidates,
            horizon_laps=h_laps,
            n_rollouts=rolls,
            seed=active_seed,
        )

        # Step 3: Evaluate Results against Multi-Criteria Risk Profile
        evaluations = self.evaluator.evaluate_all(
            strategies=candidates,
            sim_results=sim_results,
            profile_name=active_profile,
        )

        # Step 4: Assemble Evaluated Strategy Alternatives Result
        result = self.formatter.format_result(
            evaluations=evaluations,
            state=state,
            ego_driver=ego_driver,
            horizon_laps=h_laps,
            profile_name=active_profile,
            n_rollouts=rolls,
            seed=active_seed,
        )

        return result
