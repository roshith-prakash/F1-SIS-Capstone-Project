"""
src/overtake
============
F1-SIS Overtake Probability and Dirty Air Dynamics Modeling Package.
"""

from .adapter import OvertakeAdapter
from .dataset import OvertakeDatasetBuilder
from .features import (
    FEATURES_MODEL_A,
    FEATURES_MODEL_B,
    FEATURES_MODEL_C,
    compute_overtake_features,
    resolve_circuit_name,
)
from .train import OvertakeTrainer

__all__ = [
    "OvertakeAdapter",
    "OvertakeDatasetBuilder",
    "OvertakeTrainer",
    "FEATURES_MODEL_A",
    "FEATURES_MODEL_B",
    "FEATURES_MODEL_C",
    "compute_overtake_features",
    "resolve_circuit_name",
]
