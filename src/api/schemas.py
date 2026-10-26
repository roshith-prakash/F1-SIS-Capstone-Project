"""
src/api/schemas.py
==================
Pydantic request/response schemas for the F1-SIS FastAPI backend.
"""
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel


# ── Requests ──────────────────────────────────────────────────────────────────

class SetupRequest(BaseModel):
    race_file: str           # relative path under data_fastf1_v1/laps/
    driver: str              # e.g. "VER"
    profile: str = "balanced"
    rollouts: int = 15
    horizon_laps: int = 8


class SandboxRequest(BaseModel):
    action: str              # "pit_this_lap" | "pit_next_lap" | "stay_out"
    compound: str = "HARD"   # target compound if pitting
    stay_out_laps: int = 3   # only used when action == "stay_out"


# ── Responses ─────────────────────────────────────────────────────────────────

class RaceAvailableItem(BaseModel):
    year: int
    name: str             # "Italian Grand Prix"
    file: str             # relative path
    drivers: list[str]


class SetupResponse(BaseModel):
    ok: bool
    driver: str
    total_laps: int
    race_name: str


class StrategyItem(BaseModel):
    id: str
    name: str
    score: float
    rank: int
    tactical_intent: str
    expected_position: float
    p_win: float
    p_podium: float
    p_top5: float
    p_points: float
    tyre_risk: float
    traffic_risk: float
    robustness: float
    position_distribution: dict[str, float]
    expected_position_under_sc: float
    expected_position_green: float
    sc_position_advantage: float
    compounds: list[str]
    pit_laps: list[int]
    is_user_adopted: Optional[bool] = False


class RecommendationPayload(BaseModel):
    action: str
    target_compound: Optional[str]
    expected_position: float
    confidence: float
    score: float
    runner_up_name: str
    runner_up_score: float
    score_delta: float
    changed_from_last_lap: bool
    explanation: list[str]
    counterfactual_strategy: str
    counterfactual_reasons: list[str]
    target_pit_lap: Optional[int] = None
    planned_strategy_name: Optional[str] = None
    planned_compounds: Optional[list[str]] = None
    planned_pit_laps: Optional[list[int]] = None
    evaluation_time_ms: Optional[float] = None
    user_adopted_strategy_id: Optional[str] = None
    active_override: Optional[dict] = None


class SCProbabilities(BaseModel):
    p_sc_h1: float
    p_sc_h3: float
    p_sc_h5: float
    p_sc_h10: float
    p_vsc_h1: float
    p_vsc_h3: float
    p_vsc_h5: float
    p_vsc_h10: float


class EventProbabilities(BaseModel):
    rain_5l: float
    undercut_success: float
    overtake_this_lap: float


class ParticipantPayload(BaseModel):
    driver: str
    team: str
    position: int
    compound: str
    tyre_life: float
    pit_count: int
    last_lap_time: Optional[float]
    gap_to_leader: Optional[float]
    interval_ahead: Optional[float]
    is_pit_in_lap: bool
    is_pit_out_lap: bool
    pit_probability: Optional[float] = None
    threat_level: Optional[str] = None
    likely_action: Optional[str] = None


class ConditionsPayload(BaseModel):
    has_safety_car: bool
    has_vsc: bool
    has_red_flag: bool
    track_temp: Optional[float]
    air_temp: Optional[float]
    rainfall: bool


class LapCompletePayload(BaseModel):
    lap: int
    total_laps: int
    race_complete: bool
    conditions: ConditionsPayload
    participants: dict[str, ParticipantPayload]
    recommendation: RecommendationPayload
    strategies: list[StrategyItem]
    sc_probabilities: SCProbabilities
    event_probabilities: EventProbabilities
    evaluation_time_ms: Optional[float] = None
    active_override: Optional[dict] = None
    adopted_strategy_id: Optional[str] = None


class SandboxResult(BaseModel):
    action: str
    compound: str
    expected_position: float
    p_win: float
    p_podium: float
    score: float
    vs_ai_score: float
    vs_ai_delta: float
    vs_ai_position: float
    verdict: str    # "AI is better" | "Your strategy is better" | "Equivalent"


class HistoryLap(BaseModel):
    lap: int
    recommendation_action: str
    recommendation_score: float
    actual_compound: str
    actual_tyre_life: float
    actual_position: int
    actual_lap_time: Optional[float]


class OverrideRequest(BaseModel):
    action: str  # "pit_next_lap", "pit_this_lap", "stay_out", "adopt_strategy", "clear"
    compound: Optional[str] = None
    strategy_id: Optional[str] = None
