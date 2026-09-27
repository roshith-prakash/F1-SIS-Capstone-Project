"""
src/strategy_engine/types.py
============================
Core type definitions, data structures, and configuration schemas for the
F1 Strategic AI Strategy Engine.

Strict Architectural Boundary:
- The Strategy Engine evaluates possible future strategies over a planning horizon.
- It produces evaluated and ranked strategy plans.
- The future Decision Engine determines the immediate tactical action.

All assumptions are explicitly centralized, categorized by origin, and configurable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class AssumptionSource(str, Enum):
    """Origin taxonomy for every parameter in the Strategy Engine."""
    EXISTING_MODEL_OUTPUT = "existing_model_output"
    PROJECT_SPECIFICATION = "project_specification"
    EMPIRICAL_PARAMETER = "empirical_parameter"
    INITIAL_ASSUMPTION = "initial_engineering_assumption"


@dataclass
class StintPlan:
    """Represents a planned tyre stint within a strategy."""
    stint_number: int
    compound: str                # "SOFT", "MEDIUM", "HARD"
    target_laps: int             # Planned length of this stint in laps
    start_lap: int               # Absolute race lap start
    end_lap: int                 # Absolute race lap end

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Strategy:
    """
    Representation of a multi-lap strategy plan.
    
    A strategy is a future sequence/plan containing planned pit windows,
    compounds, stint structures, and tactical intent.
    """
    strategy_id: str                      # e.g., "1S_M_H_L28"
    name: str                             # Human-readable title
    num_stops: int                        # 1 or 2
    pit_laps: list[int]                   # Planned pit laps e.g. [28]
    compounds: list[str]                  # Compound for each stint e.g. ["MEDIUM", "HARD"]
    stints: list[StintPlan] = field(default_factory=list)
    horizon_laps: int | None = None       # Simulation horizon for this evaluation (None = full remaining race)
    tactical_intent: str = "nominal"      # "undercut", "overcut", "nominal", "cover"
    assumptions: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["stints"] = [s.to_dict() if hasattr(s, "to_dict") else s for s in self.stints]
        return d


@dataclass
class RolloutOutcome:
    """Outcome of a single stochastic Monte Carlo rollout."""
    rollout_id: int
    final_position: int
    position_delta: int                   # Initial position - final position (+ = gained)
    total_race_time_seconds: float
    sc_deployed: bool
    vsc_deployed: bool
    opponent_pitted: bool
    final_tyre_age: int
    pit_stops_executed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SimulationResult:
    """
    Aggregated stochastic outcome distribution for a candidate strategy across rollouts.
    In production full-race simulation, all metrics represent outcomes at race completion.
    """
    strategy_id: str
    n_rollouts: int
    horizon_laps: int

    # Position distributions (at race completion)
    expected_position: float              # Expected finishing position at race completion
    median_position: float                # Median finishing position at race completion
    std_position: float                   # Standard deviation of finishing position at race completion
    position_p10: float                   # 10th percentile finishing position (optimistic)
    position_p90: float                   # 90th percentile finishing position (downside risk)
    position_distribution: dict[int, float] = field(default_factory=dict) # {position: probability}

    # Outcome probabilities (at race completion)
    p_win: float = 0.0                    # Probability of finishing P1 at race completion
    p_podium: float = 0.0                 # Probability of finishing in positions 1-3 at race completion
    p_top5: float = 0.0                   # Probability of finishing in positions 1-5 at race completion
    p_points: float = 0.0                 # Probability of finishing in positions 1-10 at race completion
    p_gain_positions: float = 0.0         # Probability of gaining positions relative to start

    # Race time distributions (at race completion)
    expected_time_seconds: float = 0.0    # Expected cumulative race time at race completion
    std_time_seconds: float = 0.0         # Standard deviation of finishing race time

    # Risk & event sensitivities
    p_sc_affected: float = 0.0
    expected_position_under_sc: float = 0.0
    expected_position_green: float = 0.0
    sc_position_advantage: float = 0.0     # under_sc - green (positive = better under SC)
    tyre_risk: float = 0.0                # Proximity to tyre degradation cliff
    traffic_risk: float = 0.0             # Risk of encountering traffic/dirty air
    robustness: float = 0.0               # Invariance to SC & opponent stochasticity

    # Raw rollouts (for vector analytics or RL consumption)
    rollouts: list[RolloutOutcome] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["rollouts"] = [r.to_dict() if hasattr(r, "to_dict") else r for r in self.rollouts]
        return d


@dataclass
class RiskProfile:
    """
    Strategic risk profile weight vector.
    
    The simulation outcomes remain identical; only the multi-criteria
    scoring weights change based on the active risk posture.
    """
    name: str                             # "aggressive", "conservative", "defensive", "opportunistic", "balanced"
    description: str
    weight_expected_position: float = 0.30
    weight_race_time: float = 0.25
    weight_downside_risk: float = 0.15
    weight_tyre_wear: float = 0.10
    weight_sc_robustness: float = 0.10
    weight_opponent_coverage: float = 0.10

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class StrategyEvaluation:
    """Evaluation of a simulated candidate strategy."""
    strategy: Strategy
    simulation_result: SimulationResult
    risk_profile_used: str = "balanced"

    def to_dict(self) -> dict[str, Any]:
        return {
            "strategy": self.strategy.to_dict(),
            "simulation_result": self.simulation_result.to_dict(),
            "risk_profile_used": self.risk_profile_used,
        }


@dataclass
class StrategyEngineResult:
    """
    Structured output returned by the Strategy Engine for downstream consumption
    by the Decision Engine, RL agent, and engineering telemetry.

    Strict Architectural Boundary:
    - The Strategy Engine evaluates feasible strategies and produces simulated outcome distributions.
    - It does NOT select, recommend, rank, or declare a 'best' strategy.
    - The Decision Engine consumes these evaluated alternatives to make the strategic/action selection.
    """
    timestamp: str
    current_lap: int
    ego_driver: str
    horizon_laps: int
    n_rollouts_per_strategy: int
    risk_profile: str
    candidate_count: int
    strategies: list[StrategyEvaluation]
    explanation_summary: str = ""
    simulation_metadata: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "current_lap": self.current_lap,
            "ego_driver": self.ego_driver,
            "horizon_laps": self.horizon_laps,
            "n_rollouts_per_strategy": self.n_rollouts_per_strategy,
            "risk_profile": self.risk_profile,
            "candidate_count": self.candidate_count,
            "strategies": [s.to_dict() for s in self.strategies],
            "explanation_summary": self.explanation_summary,
            "simulation_metadata": self.simulation_metadata,
            "metadata": self.metadata,
            "downstream_payload": self.to_decision_engine_payload(),
        }

    def to_decision_engine_payload(self) -> dict[str, Any]:
        """
        Generate typed payload formatted specifically for future Decision Engine consumption.
        Provides evaluated strategy alternatives across full-race outcome metrics without
        making a strategy selection or recommendation.
        """
        return {
            "current_lap": self.current_lap,
            "ego_driver": self.ego_driver,
            "strategies": [
                {
                    "strategy_id": ev.strategy.strategy_id,
                    "pit_laps": ev.strategy.pit_laps,
                    "compounds": ev.strategy.compounds,
                    "expected_position": round(ev.simulation_result.expected_position, 2),
                    "position_std": round(ev.simulation_result.std_position, 2),
                    "expected_race_time": round(ev.simulation_result.expected_time_seconds, 2),
                    "race_time_std": round(ev.simulation_result.std_time_seconds, 2),
                    "p_win": round(ev.simulation_result.p_win, 4),
                    "p_podium": round(ev.simulation_result.p_podium, 4),
                    "p_top5": round(ev.simulation_result.p_top5, 4),
                    "tyre_risk": round(ev.simulation_result.tyre_risk, 4),
                    "traffic_risk": round(ev.simulation_result.traffic_risk, 4),
                    "robustness": round(ev.simulation_result.robustness, 4),
                }
                for ev in self.strategies
            ],
            "simulation_metadata": {
                "rollouts_per_strategy": self.n_rollouts_per_strategy,
                "horizon_laps": self.horizon_laps,
                "seed": self.simulation_metadata.get("seed", 42),
            },
        }


# =============================================================================
# Centralized Configuration & Assumptions
# =============================================================================

@dataclass
class StrategyEngineConfig:
    """
    Centralized configuration for the Strategy Engine.
    
    Every parameter is annotated with its origin source (Project Specification,
    Existing Model Output, Empirical Parameter, or Initial Engineering Assumption).
    No parameter is silently hard-coded.
    """
    # 1. Simulation Scope & Horizon
    # Default is None -> dynamic full remaining-race horizon derived from RaceState:
    # remaining_laps = total_laps_expected - current_lap. Configurable if explicit limit desired.
    default_horizon_laps: int | None = None
    default_n_rollouts: int = 150
    default_random_seed: int = 42

    # 2. Pit Lane Time Loss Parameters
    # [Project Specification: Section 5.5 - green flag ~22s, SC ~0-12s loss]
    pit_loss_green_seconds: float = 22.0
    pit_loss_sc_seconds: float = 12.0
    pit_loss_vsc_seconds: float = 15.0

    # 3. Stint & Tyre Degradation Constraints
    # [Initial Engineering Assumption: minimum viable stint before pitting again]
    minimum_stint_length: int = 4

    # Hybrid Tyre Viability: static ages as SOFT cliff reference (NOT hard gates for candidate pruning).
    # [Empirical Parameter: Project Spec Section 5.6 median stint length table + 1 std dev]
    # Soft median 18 ± 4  -> cliff reference ~24
    max_tyre_age_soft: int = 24
    # Medium median 28 ± 6 -> cliff reference ~34
    max_tyre_age_medium: int = 34
    # Hard median 38 ± 8  -> cliff reference ~46
    max_tyre_age_hard: int = 46

    # [HYBRID APPROACH] Degradation-based stint viability (replaces hard static-age gate in candidate pruning).
    # If True: use degradation simulation beyond the cliff reference to determine if a stint is viable;
    # a stint is rejected only when the car runs significantly past the compound's cliff age,
    # causing compounding exponential pace loss BEYOND the reference cliff point.
    # Static ages are retained as cliff-reference points for the simulator's penalty calculation,
    # NOT as hard binary gates in the candidate generator.
    dynamic_tyre_viability_check: bool = True
    # Maximum pace loss (seconds) accumulated ONLY in laps run BEYOND the static cliff reference age
    # before that stint is considered unviable. This targets the tyre cliff specifically.
    # [Empirical Parameter: ~3.5s cliff excess before the stint is considered uncompetitive.]
    # E.g. HARD 39 laps (cliff @46) -> 0 laps beyond cliff -> 0s -> valid.
    # E.g. SOFT 30 laps (cliff @24) -> 6 laps beyond cliff -> ~16.9s -> invalid.
    deg_cliff_excess_threshold_seconds: float = 3.5
    # Degradation rate per lap (seconds/lap) for each compound — used for viability projection.
    # [Empirical Parameter: matching simulator._compound_deg_rate values for consistency.]
    deg_rate_soft: float = 0.085
    deg_rate_medium: float = 0.045
    deg_rate_hard: float = 0.025
    # Non-linear degradation exponent (mirrors simulator: 0.008 * age^1.3)
    deg_nonlinear_coeff: float = 0.008
    deg_nonlinear_exp: float = 1.3

    # [F1 Sporting Regulations: dry race requires at least two distinct dry compounds]
    enforce_f1_two_compound_rule: bool = True
    dry_compounds: tuple[str, ...] = ("SOFT", "MEDIUM", "HARD")

    # 4. Safety Car & VSC Dynamics
    # [Empirical Parameter: historical F1 SC duration distribution 2-5 laps]
    sc_duration_mean_laps: float = 3.2
    sc_duration_min_laps: int = 2
    sc_duration_max_laps: int = 5
    vsc_duration_mean_laps: float = 2.0
    # [Project Specification: Section 5.5 - SC lap time delta ~10-15s slower]
    sc_lap_time_delta_seconds: float = 15.0
    vsc_lap_time_delta_seconds: float = 8.0
    # [Project Specification: Section 5.5 - Field compression under SC]
    sc_field_bunching_gap_seconds: float = 0.8

    # 5. Stochastic Noise & Variability
    # [Empirical Parameter: driver consistency pace standard deviation ~0.25s]
    lap_time_noise_std: float = 0.25
    # [Initial Engineering Assumption: tyre degradation residual variance]
    tyre_deg_noise_std: float = 0.05

    # 6. Traffic, Dirty Air, & Overtaking Dynamics
    # [Fallback empirical parameter when OvertakeAdapter is absent: pace advantage delta needed to overtake]
    overtake_pace_advantage_threshold: float = 0.80
    # [Fallback empirical parameter when OvertakeAdapter is absent: dirty air pace penalty when trailing car < 1.0s]
    dirty_air_penalty_seconds: float = 0.35
    # [Empirical Parameter: DRS advantage delta]
    drs_boost_seconds: float = 0.40

    # 7. Opponent Interaction Rules
    # [Project Specification: Section 5.6 - Tier 1: P(ego-2) to P(ego+2)]
    tier1_window_ahead: int = 2
    tier1_window_behind: int = 2
    # [Project Specification: Section 5.5 & 5.6 - SC override: tyre_age > 10 pits unconditionally]
    sc_opponent_pit_tyre_age: int = 10

    # 8. Fuel Effect
    # [Existing Model Prior: TyreDegAdapter fuel weight penalty 0.065 s/lap]
    fuel_penalty_per_lap_seconds: float = 0.065

    # 9. Pre-Configured Strategic Risk Profiles
    # [Project Specification: Section 5.8 - Risk Profile Engine weight vectors]
    risk_profiles: dict[str, RiskProfile] = field(default_factory=lambda: {
        "balanced": RiskProfile(
            name="balanced",
            description="Balanced trade-off between expected position, race time, and variance.",
            weight_expected_position=0.30,
            weight_race_time=0.25,
            weight_downside_risk=0.15,
            weight_tyre_wear=0.10,
            weight_sc_robustness=0.10,
            weight_opponent_coverage=0.10,
        ),
        "aggressive": RiskProfile(
            name="aggressive",
            description="Chase position and track overtakes, accepts higher variance and tyre risk.",
            weight_expected_position=0.40,
            weight_race_time=0.15,
            weight_downside_risk=0.05,
            weight_tyre_wear=0.05,
            weight_sc_robustness=0.15,
            weight_opponent_coverage=0.20,
        ),
        "conservative": RiskProfile(
            name="conservative",
            description="Protect current position, minimize downside risk and tyre degradation.",
            weight_expected_position=0.20,
            weight_race_time=0.30,
            weight_downside_risk=0.25,
            weight_tyre_wear=0.15,
            weight_sc_robustness=0.05,
            weight_opponent_coverage=0.05,
        ),
        "defensive": RiskProfile(
            name="defensive",
            description="Priority on covering rival undercut threats and maintaining track position.",
            weight_expected_position=0.15,
            weight_race_time=0.20,
            weight_downside_risk=0.30,
            weight_tyre_wear=0.15,
            weight_sc_robustness=0.10,
            weight_opponent_coverage=0.10,
        ),
        "opportunistic": RiskProfile(
            name="opportunistic",
            description="Exploit safety car uncertainty and compound offset opportunities.",
            weight_expected_position=0.25,
            weight_race_time=0.10,
            weight_downside_risk=0.05,
            weight_tyre_wear=0.10,
            weight_sc_robustness=0.35,
            weight_opponent_coverage=0.15,
        ),
    })

    def get_max_tyre_age(self, compound: str) -> int:
        """Resolve the static cliff-reference age for a compound.
        
        In the hybrid approach this is used as a SOFT cliff reference for the
        simulator's pace penalty, NOT as a hard binary gate in candidate pruning.
        """
        comp = compound.upper().strip()
        if comp == "SOFT":
            return self.max_tyre_age_soft
        elif comp == "MEDIUM":
            return self.max_tyre_age_medium
        elif comp == "HARD":
            return self.max_tyre_age_hard
        return self.max_tyre_age_medium

    def get_deg_rate(self, compound: str) -> float:
        """Resolve per-lap degradation slope (seconds/lap) for a compound."""
        comp = compound.upper().strip()
        if comp == "SOFT":
            return self.deg_rate_soft
        elif comp == "MEDIUM":
            return self.deg_rate_medium
        elif comp == "HARD":
            return self.deg_rate_hard
        return self.deg_rate_medium

    def compute_stint_pace_loss(self, compound: str, stint_length_laps: int, start_age: int = 0) -> float:
        """
        Simulate cumulative pace loss (seconds above fresh-tyre baseline) across
        a planned stint using the same formula as the simulator.

        Formula per lap: deg_slope * tyre_age + deg_nonlinear_coeff * tyre_age^deg_nonlinear_exp

        Args:
            compound: Tyre compound (SOFT / MEDIUM / HARD).
            stint_length_laps: Number of laps in the planned stint.
            start_age: Starting tyre age at beginning of stint (0 = fresh).

        Returns:
            Total accumulated pace loss in seconds across the full stint.
        """
        deg_slope = self.get_deg_rate(compound)
        total_loss = 0.0
        for lap_offset in range(stint_length_laps):
            age = start_age + lap_offset
            lap_deg = deg_slope * age + self.deg_nonlinear_coeff * (age ** self.deg_nonlinear_exp)
            total_loss += lap_deg
        return total_loss

    def compute_stint_cliff_excess_loss(
        self,
        compound: str,
        stint_length_laps: int,
        start_age: int = 0,
    ) -> float:
        """
        Compute the pace loss accumulated ONLY in laps run BEYOND the static cliff reference age.

        This is the correct hybrid viability signal: it captures the steep, compounding
        exponential degradation that occurs when a tyre runs well past its rated life.
        Normal long stints that stay within the cliff window return 0.0 and are accepted.

        Args:
            compound: Tyre compound (SOFT / MEDIUM / HARD).
            stint_length_laps: Total planned laps in the stint.
            start_age: Starting tyre age at beginning of stint (0 = fresh).

        Returns:
            Cumulative pace loss (seconds) accumulated only in laps BEYOND the cliff age.
            Returns 0.0 if the stint stays within the rated tyre life window.
        """
        cliff_age = self.get_max_tyre_age(compound)
        deg_slope = self.get_deg_rate(compound)
        cliff_excess_loss = 0.0
        for lap_offset in range(stint_length_laps):
            age = start_age + lap_offset
            if age >= cliff_age:
                lap_deg = deg_slope * age + self.deg_nonlinear_coeff * (age ** self.deg_nonlinear_exp)
                cliff_excess_loss += lap_deg
        return cliff_excess_loss

