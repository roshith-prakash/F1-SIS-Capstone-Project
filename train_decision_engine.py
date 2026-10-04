"""
train_decision_engine.py
========================
Trains the Candidate-Conditioned DQN / Dueling Double-DQN Decision Policy and evaluates
it against the deterministic Baseline Policy.

Usage:
  python train_decision_engine.py --episodes 15 --warmup 5
  python train_decision_engine.py --arch dueling --ml-physics --curriculum
"""

from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from decision_engine.baseline import BaselineDecisionPolicy
from decision_engine.evaluator import PolicyEvaluator
from decision_engine.rl.trainer import DQNTrainer
from decision_engine.rl.environment import F1StrategyEnv
from decision_engine.rl.curriculum import CurriculumScheduler


def main():
    parser = argparse.ArgumentParser(description="Train F1-SIS RL Decision Engine")
    parser.add_argument("--episodes", type=int, default=15, help="Number of training episodes")
    parser.add_argument("--warmup", type=int, default=5, help="Number of baseline warm-up demonstration episodes")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--arch", choices=["dueling", "standard"], default="dueling", help="Network architecture")
    parser.add_argument("--no-double", action="store_true", help="Disable Double-DQN decoupling")
    parser.add_argument("--no-per", action="store_true", help="Disable Prioritized Experience Replay")
    parser.add_argument("--ml-physics", action="store_true", help="Enable XGBoost and empirical ML submodel physics")
    parser.add_argument("--curriculum", action="store_true", help="Enable progressive curriculum learning")
    parser.add_argument(
        "--output",
        type=str,
        default="models/Decision Engine/decision_engine_dqn.pt",
        help="Path to save trained weights",
    )
    args = parser.parse_args()

    print("\n=======================================================")
    print("  F1-SIS DECISION ENGINE: RL TRAINING PIPELINE")
    print("=======================================================")
    print(f"Architecture : {args.arch.upper()} | Double-DQN: {not args.no_double} | PER: {not args.no_per}")
    print(f"ML Physics   : {args.ml_physics} | Curriculum: {args.curriculum}")
    print(f"Episodes     : {args.episodes} | Warm-up Demos: {args.warmup} | LR: {args.lr}")
    print(f"Checkpoint   : {args.output}\n")

    env = F1StrategyEnv(
        rollouts_per_step=10,
        horizon_laps=6,
        seed=42,
        use_ml_physics=args.ml_physics,
    )

    trainer = DQNTrainer(
        env=env,
        architecture=args.arch,
        double_dqn=not args.no_double,
        use_per=not args.no_per,
        learning_rate=args.lr,
        batch_size=32,
    )

    curriculum = None
    if args.curriculum:
        curriculum = CurriculumScheduler(use_ml_physics=args.ml_physics)

    # Execute training
    trained_rl_policy = trainer.train(
        n_episodes=args.episodes,
        warm_up_episodes=args.warmup,
        save_path=ROOT / args.output,
        curriculum=curriculum,
    )

    # Head-to-head benchmark: Baseline vs Trained RL Policy
    print("\n=======================================================")
    print("  HEAD-TO-HEAD EVALUATION: Baseline vs Trained RL")
    print("=======================================================")
    evaluator = PolicyEvaluator(
        env_factory=lambda seed: F1StrategyEnv(
            rollouts_per_step=10,
            horizon_laps=6,
            seed=seed,
            use_ml_physics=args.ml_physics,
        )
    )

    comparison_policies = [
        ("Baseline (Balanced)", BaselineDecisionPolicy(risk_profile="balanced")),
        (f"Trained RL ({args.arch.capitalize()})", trained_rl_policy),
    ]

    results = evaluator.compare_policies(comparison_policies, n_episodes=3, base_seed=500)

    print("\n" + "=" * 65)
    print(f"{'Policy':<22} | {'Points (Avg)':<12} | {'Pos (Avg)':<10} | {'Win %':<8} | {'Podium %':<8}")
    print("-" * 65)
    for r in results:
        print(
            f"{r.policy_name:<22} | "
            f"{r.mean_points:>5.1f} +/- {r.std_points:<4.1f} | "
            f"P{r.mean_position:>4.1f}      | "
            f"{r.win_rate * 100:>5.1f}%  | "
            f"{r.podium_rate * 100:>5.1f}%"
        )
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
