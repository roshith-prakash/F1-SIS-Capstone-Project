"""
src/sc_risk/adapter.py
======================
SCRiskAdapter
-------------
Bridges the live RaceState (produced by RaceStateManager) to the
SC_Risk_Estimation model's ``get_fcy_probabilities()`` interface.

All 6 live-race-evidence features and both historical logit priors are
derived purely from the RaceState object and its lap_history, with no
dependency on the raw CSV data.

Usage
-----
>>> from src.sc_risk.adapter import SCRiskAdapter
>>> adapter = SCRiskAdapter(
...     circ_prog_csv='models/SC Estimation/sc_vsc_historical_prior.csv',
... )
>>> H_t = adapter.build_H_t(state)
>>> probs = get_fcy_probabilities(H_t, logit_sc, logit_vsc, X_cols_sc, X_cols_vsc)
"""

from __future__ import annotations

import math
from typing import Any, Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Constants matching SC_Risk_Estimation.ipynb
# ---------------------------------------------------------------------------
PROGRESS_BINS   = [0, 0.01, 0.20, 0.40, 0.60, 0.80, 1.01]
PROGRESS_LABELS = ['Lap1', '2-20', '20-40', '40-60', '60-80', '80-100']
CLOSE_BATTLE_GAP_S = 1.0  # seconds — same threshold as the CSV adapter


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    """Logit transform with clamping to avoid +/-inf."""
    p = max(eps, min(1 - eps, float(p)))
    return math.log(p / (1.0 - p))


def _median(values: list) -> Optional[float]:
    """Return the median of a list, ignoring None values."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    mid = len(vals) // 2
    return vals[mid] if len(vals) % 2 else (vals[mid - 1] + vals[mid]) / 2.0


def _progress_bin(race_progress: float) -> str:
    """Map a race_progress fraction [0, 1] to the matching PROGRESS_LABEL."""
    for i, (lo, hi) in enumerate(zip(PROGRESS_BINS[:-1], PROGRESS_BINS[1:])):
        if lo <= race_progress < hi:
            return PROGRESS_LABELS[i]
    return PROGRESS_LABELS[-1]


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class SCRiskAdapter:
    """
    Converts a live RaceState into the H_t feature dict expected by
    ``get_fcy_probabilities()`` in SC_Risk_Estimation.ipynb.

    Parameters
    ----------
    circ_prog_csv : str or path-like
        Path to ``models/SC Estimation/sc_vsc_historical_prior.csv`` produced
        by SC_Risk_Estimation.ipynb (Cell 22 serialization step).
    sector_anomaly_window : int
        Number of *preceding* laps used to compute the rolling sector-time
        baseline for sector_anomaly_score.  Defaults to 5 (matching CSV adapter).
    retirement_window : int
        Number of preceding laps over which to count car-count drops.
        Defaults to 3 (matching CSV adapter).
    yellow_window : int
        Number of preceding laps over which to sum yellow-flag occurrences.
        Defaults to 3.
    """

    def __init__(
        self,
        circ_prog_csv: str,
        sector_anomaly_window: int = 5,
        retirement_window: int = 3,
        yellow_window: int = 3,
    ) -> None:
        self._circ_prog = pd.read_csv(circ_prog_csv)
        self._sector_anomaly_window = sector_anomaly_window
        self._retirement_window = retirement_window
        self._yellow_window = yellow_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_H_t(self, state: Any) -> dict:
        """
        Build the H_t feature dict from a ``RaceState`` object.

        Parameters
        ----------
        state : RaceState
            The committed race state for the current lap, including its
            ``lap_history`` which must have entries from earlier laps
            (populated automatically by RaceStateManager.commit_lap() after
            the manager.py lap_history extension).

        Returns
        -------
        dict
            Keys: ``logit_p_sc_hist``, ``logit_p_vsc_hist``,
            ``close_battles``, ``field_spread``, ``sector_anomaly_score``,
            ``recent_retirements``, ``yellow_last3``, ``sc_count_cumul``.
            All values are floats; NaN / None replaced with 0.0.
        """
        history = list(state.lap_history)  # ordered list of committed laps
        current_lap = state.current_lap or 0

        H_t: dict = {}

        # Historical logit priors
        H_t['logit_p_sc_hist']  = self._get_logit_hist(state, 'p_sc_historical')
        H_t['logit_p_vsc_hist'] = self._get_logit_hist(state, 'p_vsc_historical')

        # Live race-evidence features
        H_t['close_battles']        = self._compute_close_battles(state)
        H_t['field_spread']         = self._compute_field_spread(state)
        H_t['sector_anomaly_score'] = self._compute_sector_anomaly(history)
        H_t['recent_retirements']   = self._compute_recent_retirements(history)
        H_t['yellow_last3']         = self._compute_yellow_last3(history)
        H_t['sc_count_cumul']       = self._compute_sc_count_cumul(history, current_lap)

        # Replace any remaining NaN / None with 0
        return {
            k: (v if (v is not None and not math.isnan(float(v))) else 0.0)
            for k, v in H_t.items()
        }

    # ------------------------------------------------------------------
    # Historical prior
    # ------------------------------------------------------------------
    def _get_logit_hist(self, state: Any, col: str) -> float:
        """
        Look up the historical per-circuit per-progress-bin prior from the
        circ_prog table and return its logit transform.

        Falls back to the global progress-bin rate if:
          - the circuit has no row for this bin, OR
          - the circuit prior is exactly 0.0 (e.g. Silverstone had zero SC events
            in 2018-2023 training data, so using 0 would always produce logit(eps)).
        """
        location = getattr(state, 'location', None)
        if location is None:
            return 0.0

        # Compute race progress fraction
        planned = getattr(state, 'total_laps_expected', None)
        current = getattr(state, 'current_lap', None)
        if planned and current:
            race_progress = min(1.0, current / planned)
        elif getattr(state, 'race_completion_pct', None) is not None:
            race_progress = min(1.0, state.race_completion_pct / 100.0)
        else:
            race_progress = 0.0

        prog_bin = _progress_bin(race_progress)
        mask_bin = self._circ_prog['progress_bin'].astype(str) == prog_bin

        # Try circuit-specific prior first
        mask_circ = (
            self._circ_prog['Location'].str.lower() == str(location).lower()
        )
        circ_row = self._circ_prog[mask_circ & mask_bin]

        p_circ = float(circ_row[col].iloc[0]) if (not circ_row.empty and col in circ_row.columns) else None

        # If circuit prior is missing OR is exactly 0.0 (no events ever recorded),
        # fall back to the global rate for this progress bin.
        if p_circ is None or p_circ == 0.0:
            global_row = self._circ_prog[mask_bin]
            if not global_row.empty and col in global_row.columns:
                # Use the mean across all circuits in this bin as the global rate
                p = float(global_row[col].mean())
            else:
                p = 0.0
        else:
            p = p_circ

        return _safe_logit(p)

    # ------------------------------------------------------------------
    # Live features
    # ------------------------------------------------------------------
    def _compute_close_battles(self, state: Any) -> float:
        """
        Count of active cars whose interval to the car ahead is < 1.0 s.
        Matches CSV adapter: (df['IntervalToPositionAheadSeconds'] < 1.0).sum()
        """
        count = 0
        for p in state.participants.values():
            if not p.is_active:
                continue
            interval = p.interval_to_position_ahead_seconds
            if interval is not None and interval < CLOSE_BATTLE_GAP_S:
                count += 1
        return float(count)

    def _compute_field_spread(self, state: Any) -> float:
        """
        Std dev of gap-to-leader across all active classified cars.
        Matches CSV adapter: df['GapToLeaderSeconds'].std()
        """
        gaps = [
            p.gap_to_leader_seconds
            for p in state.participants.values()
            if p.is_active and p.gap_to_leader_seconds is not None
        ]
        if len(gaps) < 2:
            return 0.0
        n = len(gaps)
        mean = sum(gaps) / n
        variance = sum((g - mean) ** 2 for g in gaps) / (n - 1)
        return math.sqrt(variance)

    def _compute_sector_anomaly(self, history: list) -> float:
        """
        Max across 3 sectors of (current_median / rolling-window-median - 1),
        clipped >= 0.

        Uses the ``sector_medians`` key added to lap_history by the
        extended manager.py.

        Matches CSV adapter:
            agg['s1_anom'] = s1_med / s1_med.shift(1).rolling(5).median() - 1
            sector_anomaly_score = max(s1_anom, s2_anom, s3_anom).clip(lower=0)
        """
        if len(history) < 2:
            return 0.0

        current = history[-1]
        cur_s = current.get('sector_medians', {})

        # Rolling baseline: up to `window` laps *before* the current one (shift(1))
        window = history[-1 - self._sector_anomaly_window: -1]
        if not window:
            return 0.0

        anomalies = []
        for key in ('s1', 's2', 's3'):
            cur_val = cur_s.get(key)
            baseline_vals = [h.get('sector_medians', {}).get(key) for h in window]
            baseline_med = _median(baseline_vals)
            if cur_val is not None and baseline_med and baseline_med > 0:
                anomaly = (cur_val / baseline_med) - 1.0
                anomalies.append(max(0.0, anomaly))

        return float(max(anomalies)) if anomalies else 0.0

    def _compute_recent_retirements(self, history: list) -> float:
        """
        Retirements that occurred in the last ``retirement_window`` laps,
        using the same peak-car logic as the CSV adapter:

            retirements_total = (peak_cars_so_far - n_cars).clip(lower=0)
            recent_retirements = retirements_total[t] - retirements_total[t-3]

        Crucially, ``retirements_total`` uses the RUNNING MAXIMUM of car count
        (peak so far), not a simple diff.  This prevents race-end car-count
        drops (lapped cars not reported, finishers going inactive) from being
        misread as retirements.
        """
        if len(history) < 2:
            return 0.0

        # Build retirements_total series (same as CSV adapter's cummax logic)
        peak = 0
        ret_total = []  # one value per lap in history
        for entry in history:
            n = entry.get('n_active_cars')
            if n is None:
                ret_total.append(ret_total[-1] if ret_total else 0)
                continue
            if n > peak:
                peak = n
            ret_total.append(max(0, peak - n))

        # recent_retirements = diff over window laps (clipped >= 0)
        current_ret = ret_total[-1]
        lookback_idx = max(0, len(ret_total) - 1 - self._retirement_window)
        past_ret = ret_total[lookback_idx]
        return float(max(0.0, current_ret - past_ret))

    def _compute_yellow_last3(self, history: list) -> float:
        """
        Sum of yellow-flag laps over the last `yellow_window` laps *before*
        the current lap.

        Matches CSV adapter:
            agg['yellow_last3'] = yellow.astype(int).shift(1).rolling(3).sum()
        """
        # Exclude the current lap (shift(1)) — look at the laps immediately before
        preceding = history[-1 - self._yellow_window: -1]
        total = 0.0
        for entry in preceding:
            cond = entry.get('current_conditions', {})
            has_yellow = cond.get('has_yellow', False)
            total += 1.0 if has_yellow else 0.0
        return total

    def _compute_sc_count_cumul(self, history: list, current_lap: int) -> float:
        """
        Cumulative count of Safety Car *deployments* (SC start events) up to
        but NOT including the current lap.

        Matches CSV adapter:
            df_feat['sc_count_cumul'] = sc_started.cumsum().shift(1).fillna(0)

        An SC deployment is a False->True transition in ``has_safety_car``
        across consecutive lap_history entries.
        """
        sc_count = 0.0
        prev_sc = False
        for entry in history:
            if entry.get('lap_number', 0) >= current_lap:
                break  # lag-by-one: do not count the current lap
            cond = entry.get('current_conditions', {})
            cur_sc = bool(cond.get('has_safety_car', False))
            if cur_sc and not prev_sc:
                sc_count += 1.0
            prev_sc = cur_sc
        return sc_count
