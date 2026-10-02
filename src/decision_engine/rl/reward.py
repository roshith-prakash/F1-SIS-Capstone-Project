"""
Reward calculation for RL Decision Engine in F1-SIS.
"""

from __future__ import annotations
from typing import Any
import numpy as np

# Official F1 points scoring system (Top 10)
F1_POINTS_MAP: dict[int, float] = {
    1: 25.0,
    2: 18.0,
    3: 15.0,
    4: 12.0,
    5: 10.0,
    6: 8.0,
    7: 6.0,
    8: 4.0,
    9: 2.0,
    10: 1.0,
}


class StrategyRewardCalculator:
    """
    Computes step-level and terminal rewards for RL policy training:
    - Intermediate reward: on-track position gains (+0.1 per position gained, -0.1 per lost)
    - Action penalty: small pit stop cost (-0.05) to discourage superfluous pitting
    - Terminal reward: normalized F1 championship points (P1 = 1.0, P2 = 0.72, etc.)
    """

    def __init__(
        self,
        pit_penalty: float = -0.05,
        position_gain_weight: float = 0.1,
        points_scale: float = 25.0,
    ):
        self.pit_penalty = pit_penalty
        self.position_gain_weight = position_gain_weight
        self.points_scale = points_scale

    def calculate_step_reward(
        self,
        prev_position: int,
        curr_position: int,
        is_pit_lap: bool,
        is_terminal: bool,
        final_position: int | None = None,
    ) -> float:
        """
        Calculates the immediate reward for a single lap transition.
        """
        reward = 0.0

        # Small penalty if pitting this lap
        if is_pit_lap:
            reward += self.pit_penalty

        # Position change (lower position number = better in racing)
        pos_delta = prev_position - curr_position
        reward += pos_delta * self.position_gain_weight

        # Terminal championship reward if race complete
        if is_terminal:
            finish_pos = final_position if final_position is not None else curr_position
            f1_pts = F1_POINTS_MAP.get(finish_pos, 0.0)
            reward += f1_pts / self.points_scale

        return float(reward)
