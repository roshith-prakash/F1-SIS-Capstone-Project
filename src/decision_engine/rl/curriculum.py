"""
Curriculum Learning Framework for F1 Strategy Decision Engine.
Implements staged progression:
Stage 1: Sprint Endgame (10 laps, 3 cars) -> Terminal reward optimization
Stage 2: Pit Window (20 laps, 5 cars)     -> Undercut/overcut pit timing
Stage 3: SC Chaos (15 laps, 5 cars, SC)   -> Opportunistic safety car pitting
Stage 4: Full Grand Prix (52 laps, 5 cars)-> Full race end-to-end policy
"""

from __future__ import annotations
from typing import Any
import numpy as np


class CurriculumScheduler:
    """
    Manages progressive training stages for RL Strategy Decision Engine.
    """

    STAGE_NAMES = {
        1: "Sprint Endgame (10 laps, 3 cars)",
        2: "Pit Window Execution (20 laps, 5 cars)",
        3: "Safety Car Chaos (15 laps, SC active)",
        4: "Full Grand Prix (52 laps, full field)",
    }

    STAGE_THRESHOLDS = {
        1: 5.0,   # Rolling average reward threshold to advance from Stage 1 -> 2
        2: 4.0,   # Stage 2 -> 3
        3: 3.0,   # Stage 3 -> 4
    }

    def __init__(
        self,
        start_stage: int = 1,
        max_stage: int = 4,
        window_size: int = 4,
        use_ml_physics: bool = False,
    ):
        self.current_stage = max(1, min(start_stage, max_stage))
        self.max_stage = max_stage
        self.window_size = window_size
        self.use_ml_physics = use_ml_physics
        self.reward_history: list[float] = []
        self._current_env: Any = None

    @property
    def stage_name(self) -> str:
        return self.STAGE_NAMES.get(self.current_stage, f"Stage {self.current_stage}")

    def maybe_advance(self, episode_reward: float) -> bool:
        """
        Records episode reward and advances curriculum stage if performance criteria met.
        """
        self.reward_history.append(episode_reward)
        if self.current_stage >= self.max_stage:
            return False

        if len(self.reward_history) < self.window_size:
            return False

        recent = self.reward_history[-self.window_size:]
        mean_reward = float(np.mean(recent))
        threshold = self.STAGE_THRESHOLDS.get(self.current_stage, 5.0)

        if mean_reward >= threshold:
            prev_stage = self.current_stage
            self.current_stage += 1
            self.reward_history.clear()
            self._current_env = None  # Force env re-creation
            print(f"\n[Curriculum] >>> ADVANCING from Stage {prev_stage} to Stage {self.current_stage}: {self.stage_name} (Mean Reward: {mean_reward:.2f} >= {threshold:.2f}) <<<\n")
            return True

        return False

    def get_env_config(self) -> dict[str, Any]:
        """Returns parameters needed to instantiate F1StrategyEnv for the current stage."""
        from .environment import F1StrategyEnv

        base_state = F1StrategyEnv.create_stage_initial_state(self.current_stage)
        horizon = 6 if self.current_stage == 1 else 8 if self.current_stage <= 2 else 10
        rollouts = 10

        return {
            "base_state": base_state,
            "horizon_laps": horizon,
            "rollouts_per_step": rollouts,
            "use_ml_physics": self.use_ml_physics,
        }

    def get_env(self) -> Any:
        """Instantiates or returns the configured environment for the current curriculum stage."""
        from .environment import F1StrategyEnv

        if self._current_env is None:
            cfg = self.get_env_config()
            self._current_env = F1StrategyEnv(**cfg)
        return self._current_env
