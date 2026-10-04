"""
Policy Evaluator for comparing Decision Engine policies (Baseline vs RL vs risk profiles).
Evaluates policies across simulated races on key metrics: points, finish position, win/podium rates.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence
import numpy as np

from .rl.reward import F1_POINTS_MAP
from .rl.environment import F1StrategyEnv


class DecisionPolicyProtocol(Protocol):
    """Protocol defining the decision policy interface."""

    def select_action(
        self, payload_or_state: Any, candidates: list[dict[str, Any]] | None = None
    ) -> Any:
        ...


@dataclass
class PolicyEvaluationResult:
    """Aggregated statistical outcome of a decision policy across multiple races."""

    policy_name: str
    total_races: int
    mean_points: float
    std_points: float
    median_position: float
    mean_position: float
    win_rate: float
    podium_rate: float
    top5_rate: float
    mean_pit_stops: float
    finishing_positions: list[int] = field(default_factory=list)
    points_history: list[float] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"=== Evaluation: {self.policy_name} ({self.total_races} races) ===\n"
            f"  Mean Points: {self.mean_points:.2f} +/- {self.std_points:.2f}\n"
            f"  Mean Position: P{self.mean_position:.2f} (Median: P{self.median_position:.1f})\n"
            f"  Win Rate: {self.win_rate * 100:.1f}%\n"
            f"  Podium Rate: {self.podium_rate * 100:.1f}%\n"
            f"  Top 5 Rate: {self.top5_rate * 100:.1f}%\n"
            f"  Avg Pit Stops: {self.mean_pit_stops:.2f}\n"
        )


class PolicyEvaluator:
    """
    Evaluates and benchmarks decision policies head-to-head under identical race conditions.
    """

    def __init__(self, env_factory: Any | None = None):
        self.env_factory = env_factory or (lambda seed: F1StrategyEnv(seed=seed))

    def evaluate_policy(
        self,
        policy: Any,
        policy_name: str = "Policy",
        n_episodes: int = 20,
        base_seed: int = 100,
    ) -> PolicyEvaluationResult:
        """
        Runs n_episodes simulated races with the given policy and records performance metrics.
        """
        finish_positions: list[int] = []
        points_list: list[float] = []
        pit_counts: list[int] = []

        for ep in range(n_episodes):
            seed = base_seed + ep
            env: F1StrategyEnv = self.env_factory(seed)
            obs, info = env.reset(seed=seed)
            terminated = False
            total_pits = 0

            while not terminated:
                candidates = info.get("candidates", [])

                if hasattr(policy, "select_candidate_index"):
                    # RL Policy
                    cand_features = info.get("candidate_features", np.zeros((0, 9)))
                    action_idx, _ = policy.select_candidate_index(
                        obs, cand_features, epsilon=0.0
                    )
                else:
                    # Baseline or custom policy with select_action
                    explanation = policy.select_action(
                        env.current_state, candidates=candidates
                    )
                    selected_id = explanation.selected_strategy_id
                    action_idx = 0
                    for idx, c in enumerate(candidates):
                        if c.get("strategy_id") == selected_id:
                            action_idx = idx
                            break

                obs, reward, terminated, truncated, info = env.step(action_idx)
                if info.get("immediate_action") != "STAY_OUT":
                    total_pits += 1

            ego_pos = info.get("ego_position", 10)
            pts = F1_POINTS_MAP.get(ego_pos, 0.0)

            finish_positions.append(ego_pos)
            points_list.append(pts)
            pit_counts.append(total_pits)

        positions_arr = np.array(finish_positions)
        points_arr = np.array(points_list)

        win_count = int(np.sum(positions_arr == 1))
        podium_count = int(np.sum(positions_arr <= 3))
        top5_count = int(np.sum(positions_arr <= 5))

        return PolicyEvaluationResult(
            policy_name=policy_name,
            total_races=n_episodes,
            mean_points=float(np.mean(points_arr)),
            std_points=float(np.std(points_arr)),
            median_position=float(np.median(positions_arr)),
            mean_position=float(np.mean(positions_arr)),
            win_rate=float(win_count / n_episodes),
            podium_rate=float(podium_count / n_episodes),
            top5_rate=float(top5_count / n_episodes),
            mean_pit_stops=float(np.mean(pit_counts)),
            finishing_positions=finish_positions,
            points_history=points_list,
        )

    def compare_policies(
        self,
        policies: Sequence[tuple[str, Any]],
        n_episodes: int = 20,
        base_seed: int = 100,
    ) -> list[PolicyEvaluationResult]:
        """
        Runs head-to-head evaluation across multiple policies on the same seeds.
        """
        results = []
        for name, pol in policies:
            res = self.evaluate_policy(
                policy=pol,
                policy_name=name,
                n_episodes=n_episodes,
                base_seed=base_seed,
            )
            results.append(res)
        return results
