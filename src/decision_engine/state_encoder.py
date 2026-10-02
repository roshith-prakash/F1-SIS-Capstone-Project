from __future__ import annotations

from typing import Any, Sequence
import numpy as np

# Typical nominal compound lifespans in laps used for normalizing tyre life fraction
COMPOUND_LIFESPANS: dict[str, float] = {
    "SOFT": 20.0,
    "MEDIUM": 32.0,
    "HARD": 45.0,
    "INTERMEDIATE": 30.0,
    "WET": 30.0,
}

SUPPORTED_COMPOUNDS = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]


class StateEncoder:
    """
    Encodes and normalizes RaceState and Strategy Candidates into bounded
    numerical vectors for RL and policy evaluation.
    """

    def __init__(self, max_grid_size: int = 20, max_gap_seconds: float = 20.0):
        self.max_grid_size = float(max_grid_size)
        self.max_gap_seconds = float(max_gap_seconds)

    def encode_state(self, race_state: Any, ego_driver: str | None = None) -> np.ndarray:
        """
        Extracts and normalizes features from a RaceState object or dictionary.

        Vector components:
        1. race_progress_pct [0, 1]
        2. is_safety_car (0 or 1)
        3. is_vsc (0 or 1)
        4. track_temp_norm [0, 1] (scaled between 20°C and 60°C)
        5. position_norm [0, 1] (1st = 0.95, 20th = 0.0)
        6. tyre_life_fraction [0, 1.5] (clipped)
        7-11. one-hot compound (SOFT, MEDIUM, HARD, INTERMEDIATE, WET)
        12. gap_ahead_norm [0, 1] (clipped to max_gap_seconds)
        13. gap_behind_norm [0, 1] (clipped to max_gap_seconds)

        Total dimensions: 13
        """
        # Global features
        if hasattr(race_state, "current_lap"):
            current_lap = float(race_state.current_lap or 1)
            total_laps = float(race_state.total_laps_expected or 50)
            conditions = race_state.current_conditions
            is_sc = 1.0 if getattr(conditions, "has_safety_car", False) else 0.0
            is_vsc = 1.0 if getattr(conditions, "has_vsc", False) else 0.0
            track_temp = getattr(conditions, "track_temp", None)
            participants = race_state.participants
        else:
            # Dict fallback
            current_lap = float(race_state.get("current_lap", 1))
            total_laps = float(race_state.get("total_laps_expected", 50))
            cond = race_state.get("current_conditions", {})
            is_sc = 1.0 if cond.get("has_safety_car", False) else 0.0
            is_vsc = 1.0 if cond.get("has_vsc", False) else 0.0
            track_temp = cond.get("track_temp", None)
            participants = race_state.get("participants", {})

        progress = np.clip(current_lap / max(total_laps, 1.0), 0.0, 1.0)

        # Track temperature normalized between 20C and 60C
        if track_temp is None:
            track_temp_norm = 0.5
        else:
            track_temp_norm = np.clip((float(track_temp) - 20.0) / 40.0, 0.0, 1.0)

        # Ego driver features
        ego_p = None
        if ego_driver:
            if isinstance(participants, dict):
                ego_p = participants.get(ego_driver) or participants.get(ego_driver.upper())

        # Fallback if ego_p not directly indexed or not found: find first participant
        if ego_p is None and isinstance(participants, dict) and participants:
            ego_p = next(iter(participants.values()))

        if ego_p is not None:
            if hasattr(ego_p, "position"):
                pos = float(ego_p.position or 10)
                tyre_age = float(ego_p.tyre_life or 0.0)
                compound_raw = str(ego_p.compound or "MEDIUM").strip().upper()
                gap_ahead = getattr(ego_p, "interval_to_position_ahead_seconds", None)
                gap_behind = getattr(ego_p, "gap_behind_seconds", None)
            else:
                pos = float(ego_p.get("position", 10))
                tyre_age = float(ego_p.get("tyre_life", 0.0))
                compound_raw = str(ego_p.get("compound", "MEDIUM")).strip().upper()
                gap_ahead = ego_p.get("interval_to_position_ahead_seconds")
                gap_behind = ego_p.get("gap_behind_seconds")
        else:
            pos = 10.0
            tyre_age = 5.0
            compound_raw = "MEDIUM"
            gap_ahead = 5.0
            gap_behind = 5.0

        pos_norm = np.clip(1.0 - (pos / self.max_grid_size), 0.0, 1.0)

        nominal_life = COMPOUND_LIFESPANS.get(compound_raw, 30.0)
        tyre_life_frac = np.clip(tyre_age / nominal_life, 0.0, 1.5)

        # One-hot compound
        compound_one_hot = [
            1.0 if compound_raw == c else 0.0 for c in SUPPORTED_COMPOUNDS
        ]

        # Gaps
        ahead_val = float(gap_ahead) if gap_ahead is not None else self.max_gap_seconds
        behind_val = float(gap_behind) if gap_behind is not None else self.max_gap_seconds
        gap_ahead_norm = np.clip(ahead_val / self.max_gap_seconds, 0.0, 1.0)
        gap_behind_norm = np.clip(behind_val / self.max_gap_seconds, 0.0, 1.0)

        feature_vector = np.array([
            progress,
            is_sc,
            is_vsc,
            track_temp_norm,
            pos_norm,
            tyre_life_frac,
            *compound_one_hot,
            gap_ahead_norm,
            gap_behind_norm,
        ], dtype=np.float32)

        return feature_vector

    def encode_candidate(self, candidate: dict[str, Any], current_lap: int = 1) -> np.ndarray:
        """
        Encodes a single candidate strategy into a normalized numerical feature vector.

        Candidate vector components:
        1. exp_pos_norm [0, 1] (1st = 0.95, 20th = 0.0)
        2. p_win [0, 1]
        3. p_podium [0, 1]
        4. p_top5 [0, 1]
        5. tyre_risk [0, 1]
        6. traffic_risk [0, 1]
        7. robustness [0, 1]
        8. is_immediate_pit (1.0 if current_lap in pit_laps else 0.0)
        9. total_stops (number of pit stops / 4.0, normalized)

        Total dimensions: 9
        """
        exp_pos = float(candidate.get("expected_position", 10.0))
        pos_norm = np.clip(1.0 - (exp_pos / self.max_grid_size), 0.0, 1.0)

        p_win = np.clip(float(candidate.get("p_win", 0.0)), 0.0, 1.0)
        p_podium = np.clip(float(candidate.get("p_podium", 0.0)), 0.0, 1.0)
        p_top5 = np.clip(float(candidate.get("p_top5", 0.0)), 0.0, 1.0)
        tyre_risk = np.clip(float(candidate.get("tyre_risk", 0.0)), 0.0, 1.0)
        traffic_risk = np.clip(float(candidate.get("traffic_risk", 0.0)), 0.0, 1.0)
        robustness = np.clip(float(candidate.get("robustness", 0.5)), 0.0, 1.0)

        pit_laps = candidate.get("pit_laps", [])
        is_immediate_pit = 1.0 if current_lap in pit_laps else 0.0
        total_stops = np.clip(len(pit_laps) / 4.0, 0.0, 1.0)

        return np.array([
            pos_norm,
            p_win,
            p_podium,
            p_top5,
            tyre_risk,
            traffic_risk,
            robustness,
            is_immediate_pit,
            total_stops,
        ], dtype=np.float32)

    def encode_candidates_batch(
        self, candidates: Sequence[dict[str, Any]], current_lap: int = 1
    ) -> np.ndarray:
        """
        Encodes a list of candidate strategies into a 2D array of shape [K, 9].
        """
        if not candidates:
            return np.zeros((0, 9), dtype=np.float32)
        return np.stack([self.encode_candidate(c, current_lap) for c in candidates], axis=0)
