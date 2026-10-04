"""
Decision Engine for F1-SIS: Adaptive AI Decision Intelligence Platform.
Implements consequence-selection policies (deterministic baseline and RL)
decoupled from candidate generation and simulation.
"""

from .types import ImmediateAction, DecisionExplanation
from .mapper import ActionMapper
from .state_encoder import StateEncoder, COMPOUND_LIFESPANS, SUPPORTED_COMPOUNDS
from .baseline import BaselineDecisionPolicy
from .evaluator import PolicyEvaluator, PolicyEvaluationResult
from .rl.reward import StrategyRewardCalculator, F1_POINTS_MAP
from .rl.dqn import (
    CandidateConditionedQNetwork,
    DuelingCandidateConditionedQNetwork,
    RLDecisionPolicy,
)
from .rl.replay_buffer import (
    VariableCandidateReplayBuffer,
    PrioritizedVariableCandidateReplayBuffer,
    SumTree,
)
from .rl.environment import F1StrategyEnv
from .rl.trainer import DQNTrainer
from .rl.opponent_policy import ReactiveOpponentPolicy, create_random_opponent_policies
from .rl.curriculum import CurriculumScheduler

__all__ = [
    "ImmediateAction",
    "DecisionExplanation",
    "ActionMapper",
    "StateEncoder",
    "BaselineDecisionPolicy",
    "PolicyEvaluator",
    "PolicyEvaluationResult",
    "COMPOUND_LIFESPANS",
    "SUPPORTED_COMPOUNDS",
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
