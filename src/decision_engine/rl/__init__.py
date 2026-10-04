"""
Reinforcement Learning module for the F1-SIS Decision Engine.
"""

from .reward import StrategyRewardCalculator, F1_POINTS_MAP
from .dqn import (
    CandidateConditionedQNetwork,
    DuelingCandidateConditionedQNetwork,
    RLDecisionPolicy,
)
from .replay_buffer import (
    VariableCandidateReplayBuffer,
    PrioritizedVariableCandidateReplayBuffer,
    SumTree,
)
from .environment import F1StrategyEnv
from .trainer import DQNTrainer
from .opponent_policy import ReactiveOpponentPolicy, create_random_opponent_policies
from .curriculum import CurriculumScheduler

__all__ = [
    "StrategyRewardCalculator",
    "F1_POINTS_MAP",
    "CandidateConditionedQNetwork",
    "DuelingCandidateConditionedQNetwork",
    "RLDecisionPolicy",
    "VariableCandidateReplayBuffer",
    "PrioritizedVariableCandidateReplayBuffer",
    "SumTree",
    "F1StrategyEnv",
    "DQNTrainer",
    "ReactiveOpponentPolicy",
    "create_random_opponent_policies",
    "CurriculumScheduler",
]
