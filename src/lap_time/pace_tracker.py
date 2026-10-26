from __future__ import annotations

import math
from typing import Any, Optional
import numpy as np

try:
    from race_state.models import ParticipantState, RaceState, normalize_driver
except ImportError:
    from src.race_state.models import ParticipantState, RaceState, normalize_driver

# Canonical 2024/2025 Ground-Effect Era Team Relative Pace Priors (seconds/lap relative to field median)
# Negative = faster than field median, Positive = slower than field median
DEFAULT_TEAM_PACE_PRIORS: dict[str, float] = {
    "McLaren": -0.95,
    "Ferrari": -0.80,
    "Red Bull Racing": -0.75,
    "Mercedes": -0.65,
    "Aston Martin": 0.10,
    "Racing Bulls": 0.25,
    "Haas F1 Team": 0.35,
    "Alpine": 0.45,
    "Williams": 0.55,
    "Kick Sauber": 0.85,
}

CANONICAL_TEAM_ALIASES: dict[str, str] = {
    "red bull": "Red Bull Racing",
    "red bull racing": "Red Bull Racing",
    "ferrari": "Ferrari",
    "scuderia ferrari": "Ferrari",
    "mclaren": "McLaren",
    "mclaren f1 team": "McLaren",
    "mercedes": "Mercedes",
    "mercedes amg": "Mercedes",
    "mercedes-amg": "Mercedes",
    "aston martin": "Aston Martin",
    "aston martin f1 team": "Aston Martin",
    "alpine": "Alpine",
    "alpine f1 team": "Alpine",
    "williams": "Williams",
    "williams racing": "Williams",
    "haas": "Haas F1 Team",
    "haas f1 team": "Haas F1 Team",
    "alfa romeo": "Kick Sauber",
    "alfa romeo racing": "Kick Sauber",
    "sauber": "Kick Sauber",
    "kick sauber": "Kick Sauber",
    "alphatauri": "Racing Bulls",
    "scuderia alphatauri": "Racing Bulls",
    "rb": "Racing Bulls",
    "vcarb": "Racing Bulls",
    "racing bulls": "Racing Bulls",
}


class DynamicTeamPaceTracker:
    """
    Dynamic Exponential Moving Average (EMA) Team Pace Tracker for F1-SIS.
    
    Eliminates intra-season and inter-season development drift (e.g. 2024/2025 McLaren surge,
    Red Bull balance shifts, team mid-season upgrades) by continuously updating each constructor's
    relative pace offset using observed clean-lap telemetry.
    
    Formula:
        delta_team(t) = (1 - alpha) * delta_team(t-1) + alpha * observed_residual(t)
    """

    def __init__(
        self,
        alpha: float = 0.20,
        priors: dict[str, float] | None = None,
        residual_clip: tuple[float, float] = (-1.5, 1.5),
        min_laps_for_full_ema: int = 3,
    ) -> None:
        self.alpha = float(np.clip(alpha, 0.02, 0.80))
        self.residual_clip = residual_clip
        self.min_laps_for_full_ema = max(1, min_laps_for_full_ema)

        self._priors: dict[str, float] = dict(priors or DEFAULT_TEAM_PACE_PRIORS)
        # Current active team pace offsets
        self._current_offsets: dict[str, float] = dict(self._priors)
        # Number of clean laps observed per team in current session
        self._observed_lap_counts: dict[str, int] = {t: 0 for t in self._priors}
        # In-memory history of residuals per team
        self._residuals_history: dict[str, list[float]] = {t: [] for t in self._priors}
        self._last_processed_lap: int = 0

    def canonicalize_team(self, team_name: str | None) -> str:
        """Resolve team string to canonical constructor name."""
        if not team_name:
            return "Red Bull Racing"
        cleaned = str(team_name).strip()
        if cleaned in self._current_offsets:
            return cleaned
        low = cleaned.lower()
        if low in CANONICAL_TEAM_ALIASES:
            return CANONICAL_TEAM_ALIASES[low]
        for canonical in self._current_offsets:
            if canonical.lower() in low or low in canonical.lower():
                return canonical
        return "Red Bull Racing"

    def get_team_offset(self, team_name: str | None) -> float:
        """
        Get the current dynamic pace offset for a team (seconds/lap).
        Negative indicates faster than field baseline, positive indicates slower.
        """
        canon = self.canonicalize_team(team_name)
        return float(self._current_offsets.get(canon, 0.0))

    def get_all_offsets(self) -> dict[str, float]:
        """Return a copy of all current team pace offsets."""
        return dict(self._current_offsets)

    def reset(self) -> None:
        """Reset offsets back to initialized priors."""
        self._current_offsets = dict(self._priors)
        self._observed_lap_counts = {t: 0 for t in self._priors}
        self._residuals_history = {t: [] for t in self._priors}
        self._last_processed_lap = 0

    def observe_lap_residual(self, team_name: str | None, raw_residual: float) -> float:
        """
        Update a single team's pace offset with an observed clean-lap pace residual.
        raw_residual = observed_lap_time - expected_baseline_lap_time
        Returns the updated offset.
        """
        canon = self.canonicalize_team(team_name)
        if canon not in self._current_offsets:
            self._current_offsets[canon] = 0.0
            self._observed_lap_counts[canon] = 0
            self._residuals_history[canon] = []

        clipped_res = float(np.clip(raw_residual, self.residual_clip[0], self.residual_clip[1]))
        self._residuals_history[canon].append(clipped_res)
        self._observed_lap_counts[canon] += 1

        prev_offset = self._current_offsets[canon]
        # Dynamic EMA weighting
        updated_offset = (1.0 - self.alpha) * prev_offset + self.alpha * clipped_res
        self._current_offsets[canon] = float(updated_offset)
        return self._current_offsets[canon]

    def update_from_race_state(
        self,
        state: RaceState,
        predicted_lap_times: dict[str, float] | None = None,
        circuit_base: float = 85.0,
    ) -> dict[str, float]:
        """
        Process a committed RaceState snapshot at lap t, extracting clean laps
        for all active runners and updating constructor offsets via EMA.
        
        Freezes updates during Caution periods (SC, VSC, Red Flag).
        """
        cond = getattr(state, "current_conditions", None)
        if cond and (cond.has_safety_car or cond.has_vsc or cond.has_red_flag):
            return self.get_all_offsets()

        current_lap = state.current_lap or 1
        if current_lap <= self._last_processed_lap:
            return self.get_all_offsets()
        self._last_processed_lap = current_lap

        # Group valid driver residuals by canonical team
        team_residuals: dict[str, list[float]] = {}

        for driver_code, p in state.participants.items():
            if not p.is_active or p.is_pit_in_lap or p.is_pit_out_lap:
                continue
            lt = p.last_lap_time_seconds
            if lt is None or (isinstance(lt, float) and (math.isnan(lt) or lt <= 0)):
                continue

            # Ensure lap time is within reasonable racing pace (15% of circuit baseline)
            if not (circuit_base * 0.82 <= lt <= circuit_base * 1.25):
                continue

            canon_team = self.canonicalize_team(p.team)

            # Compute raw pace residual relative to model prediction or circuit base
            if predicted_lap_times and driver_code in predicted_lap_times:
                expected_lt = predicted_lap_times[driver_code]
                raw_res = float(lt - expected_lt)
            else:
                # Approximate baseline without team offset
                compound_delta = {"SOFT": -0.4, "MEDIUM": 0.0, "HARD": 0.4}.get(
                    str(p.compound or "MEDIUM").upper(), 0.0
                )
                raw_res = float(lt - (circuit_base + compound_delta))

            team_residuals.setdefault(canon_team, []).append(raw_res)

        # Apply robust median residual per constructor
        for canon_team, res_list in team_residuals.items():
            if not res_list:
                continue
            team_median_res = float(np.median(res_list))
            self.observe_lap_residual(canon_team, team_median_res)

        return self.get_all_offsets()
