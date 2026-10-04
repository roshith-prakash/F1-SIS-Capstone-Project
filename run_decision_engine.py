"""
run_decision_engine.py
======================
Interactive runner and demonstration for the F1-SIS Decision Engine.

Usage:
  python run_decision_engine.py demo          # Run single-lap decision demonstration
  python run_decision_engine.py benchmark     # Run multi-race policy comparison (Aggressive vs Conservative vs Balanced)
  python run_decision_engine.py rl-demo       # Run Gym environment step using RL Decision Policy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src to sys.path
ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.models import CurrentConditions, ParticipantState, RaceState
from strategy_engine.engine import StrategyEngine
from strategy_engine.types import StrategyEngineConfig
from decision_engine.baseline import BaselineDecisionPolicy
from decision_engine.evaluator import PolicyEvaluator
from decision_engine.rl.environment import F1StrategyEnv
from decision_engine.rl.dqn import (
    CandidateConditionedQNetwork,
    DuelingCandidateConditionedQNetwork,
    RLDecisionPolicy,
)


def build_sample_race_state(current_lap: int = 20) -> RaceState:
    """Creates a sample race snapshot at Silverstone."""
    cond = CurrentConditions(
        air_temp=24.0,
        track_temp=38.0,
        has_safety_car=False,
        has_vsc=False,
    )
    state = RaceState(
        race_id="2024_British_GP",
        current_lap=current_lap,
        total_laps_expected=52,
        grand_prix="British Grand Prix",
        location="Silverstone",
        current_conditions=cond,
    )

    grid = [
        ("NOR", "McLaren", 1, 0.0, 1700.0, "MEDIUM", 20.0),
        ("VER", "Red Bull Racing", 2, 1.8, 1701.8, "MEDIUM", 20.0),
        ("HAM", "Mercedes", 3, 4.5, 1704.5, "MEDIUM", 20.0),
        ("LEC", "Ferrari", 4, 8.2, 1708.2, "HARD", 10.0),
        ("PIA", "McLaren", 5, 12.0, 1712.0, "HARD", 12.0),
    ]

    for idx, (d_code, team, pos, gap, ctime, comp, life) in enumerate(grid):
        ahead_gap = (ctime - grid[idx - 1][4]) if idx > 0 else 0.0
        behind_gap = (grid[idx + 1][4] - ctime) if idx + 1 < len(grid) else 20.0
        state.participants[d_code] = ParticipantState(
            driver=d_code,
            team=team,
            position=pos,
            gap_to_leader_seconds=gap,
            interval_to_position_ahead_seconds=ahead_gap,
            gap_behind_seconds=behind_gap,
            total_race_time_seconds=ctime,
            compound=comp,
            tyre_life=life,
            stint=1,
            pit_count=0,
            last_lap_time_seconds=85.2,
            is_active=True,
        )

    return state


def run_demo(profile: str = "balanced") -> None:
    print(f"\n=======================================================")
    print(f"  F1-SIS DECISION ENGINE: LIVE DEMONSTRATION")
    print(f"=======================================================")
    print(f"Track: Silverstone | Lap: 20/52 | Ego Driver: VER (P2)")
    print(f"Gap to Leader / Car Ahead (P1 NOR): +1.8s")
    print(f"Gap to Car Behind (P3 HAM)         : +2.7s")
    print(f"Risk Profile: {profile.upper()}\n")

    # 1. Initialize State and Strategy Engine
    state = build_sample_race_state(current_lap=20)
    cfg = StrategyEngineConfig(default_horizon_laps=10, default_n_rollouts=30, default_random_seed=42)
    engine = StrategyEngine(config=cfg)

    print("--> 1. Running Strategy Engine Monte Carlo Rollouts...")
    se_result = engine.evaluate_race_state(state=state, ego_driver="VER")
    print(f"     Generated {len(se_result.strategies)} candidate strategies.\n")

    # 2. Decision Engine selection
    print("--> 2. Evaluating candidates with Baseline Decision Policy...")
    policy = BaselineDecisionPolicy(risk_profile=profile)
    explanation = policy.select_action(se_result)

    print("\n-------------------------------------------------------")
    print("  DECISION ENGINE OUTPUT")
    print("-------------------------------------------------------")
    print(f"Selected Strategy   : {explanation.selected_strategy_id}")
    print(f"Immediate Action    : {explanation.immediate_action.value}")
    if explanation.target_pit_lap:
        print(f"Target Pit Lap      : Lap {explanation.target_pit_lap} (Compound: {explanation.target_compound})")
    print(f"\nTop Candidate Utilities:")
    for s_id, score in explanation.top_candidates_utilities.items():
        print(f"  - {s_id:25s} : Utility {score:.3f}")

    print(f"\nExplanation Summary:")
    print(f"  \"{explanation.summary}\"")
    print("-------------------------------------------------------\n")


def run_benchmark(episodes: int = 5) -> None:
    print(f"\n=======================================================")
    print(f"  F1-SIS POLICY BENCHMARK ({episodes} simulated races)")
    print(f"=======================================================")
    print("Benchmarking Aggressive vs Conservative vs Balanced...\n")

    policies = [
        ("Aggressive", BaselineDecisionPolicy(risk_profile="aggressive")),
        ("Conservative", BaselineDecisionPolicy(risk_profile="conservative")),
        ("Balanced", BaselineDecisionPolicy(risk_profile="balanced")),
    ]

    evaluator = PolicyEvaluator(
        env_factory=lambda seed: F1StrategyEnv(rollouts_per_step=10, horizon_laps=6, seed=seed)
    )

    results = evaluator.compare_policies(policies, n_episodes=episodes, base_seed=42)

    print("\n" + "=" * 65)
    print(f"{'Policy':<15} | {'Points (Avg)':<12} | {'Pos (Avg)':<10} | {'Win %':<8} | {'Podium %':<8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r.policy_name:<15} | "
            f"{r.mean_points:>5.1f} +/- {r.std_points:<4.1f} | "
            f"P{r.mean_position:>4.1f}      | "
            f"{r.win_rate * 100:>5.1f}%  | "
            f"{r.podium_rate * 100:>5.1f}%"
        )
    print("=" * 65 + "\n")


def run_rl_demo(arch: str = "dueling", use_ml_physics: bool = False) -> None:
    print(f"\n=======================================================")
    print(f"  F1-SIS REINFORCEMENT LEARNING DEMO ({arch.upper()} Architecture)")
    print(f"=======================================================")
    print(f"ML Submodel Physics: {use_ml_physics}\n")

    env = F1StrategyEnv(rollouts_per_step=10, horizon_laps=5, seed=42, use_ml_physics=use_ml_physics)
    obs, info = env.reset(seed=42)

    if arch == "dueling":
        q_net = DuelingCandidateConditionedQNetwork(state_dim=13, candidate_dim=9, embed_dim=32, hidden_dim=32)
    else:
        q_net = CandidateConditionedQNetwork(state_dim=13, candidate_dim=9, embed_dim=32, hidden_dim=32)

    rl_policy = RLDecisionPolicy(q_net=q_net, architecture=arch)

    cand_features = info["candidate_features"]
    print(f"Initial State Observation Vector (13 dims): \n  {obs.round(3)}")
    print(f"\nCandidate Strategies count: {len(info['candidates'])}")

    action_idx, q_vals = rl_policy.select_candidate_index(obs, cand_features, epsilon=0.0)
    chosen_strat = info["candidates"][action_idx]["strategy_id"]
    print(f"Q-values per candidate: {[round(float(q), 3) for q in q_vals]}")
    print(f"Selected Candidate Index: {action_idx} ({chosen_strat})")

    next_obs, reward, term, trunc, next_info = env.step(action_idx)
    print(f"\nAfter Step 1:")
    print(f"  Immediate Tactical Action : {next_info['immediate_action']}")
    print(f"  Step Reward               : {reward:.3f}")
    print(f"  New Current Lap           : {next_info['current_lap']}")
    print(f"  Ego Position              : P{next_info['ego_position']}")
    print("=======================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="F1-SIS Decision Engine Runner")
    parser.add_argument(
        "mode",
        nargs="?",
        default="demo",
        choices=["demo", "benchmark", "rl-demo"],
        help="Execution mode (default: demo)",
    )
    parser.add_argument(
        "--profile",
        default="balanced",
        choices=["balanced", "aggressive", "conservative"],
        help="Risk profile for demo mode",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes for benchmark mode",
    )
    parser.add_argument(
        "--arch",
        choices=["dueling", "standard"],
        default="dueling",
        help="Network architecture for rl-demo (default: dueling)",
    )
    parser.add_argument(
        "--ml-physics",
        action="store_true",
        help="Enable ML submodel physics (XGBoost, empirical pitstops, hazard models)",
    )
    args = parser.parse_args()

    if args.mode == "demo":
        run_demo(profile=args.profile)
    elif args.mode == "benchmark":
        run_benchmark(episodes=args.episodes)
    elif args.mode == "rl-demo":
        run_rl_demo(arch=args.arch, use_ml_physics=args.ml_physics)
