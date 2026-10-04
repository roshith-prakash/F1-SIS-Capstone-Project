from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class ImmediateAction(Enum):
    """Immediate tactical action mapped from a multi-lap strategy."""
    STAY_OUT = "STAY_OUT"
    PIT_SOFT = "PIT_SOFT"
    PIT_MEDIUM = "PIT_MEDIUM"
    PIT_HARD = "PIT_HARD"
    PIT_INTERMEDIATE = "PIT_INTERMEDIATE"
    PIT_WET = "PIT_WET"


@dataclass
class DecisionExplanation:
    """
    Explainability payload for the decision made by the Decision Engine.
    """
    selected_strategy_id: str
    immediate_action: ImmediateAction
    target_pit_lap: int | None
    target_compound: str | None
    top_candidates_utilities: dict[str, float] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, any]:
        return {
            "selected_strategy_id": self.selected_strategy_id,
            "immediate_action": self.immediate_action.value,
            "target_pit_lap": self.target_pit_lap,
            "target_compound": self.target_compound,
            "top_candidates_utilities": self.top_candidates_utilities,
            "summary": self.summary,
        }
