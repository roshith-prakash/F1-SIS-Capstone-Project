"""
src/opponent_model/dataset.py
=============================
Master dataset generator for Opponent Modeling (Task 29, Phase A3 & A4).

Iterates over historical race CSVs, builds race state chronologically,
evaluates foundational models (LapTime, TyreDeg, SCRisk), computes
derived strategic features, extracts pit targets on lap t+1,
and saves train/validation/test parquet splits.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional
import concurrent.futures

import numpy as np
import pandas as pd
import joblib

from .features import compute_derived_features
from .targets import compute_pit_targets_and_filters

try:
    from race_state.manager import RaceStateManager
    from lap_time.adapter import LapTimeAdapter
    from tyre_deg.adapter import TyreDegAdapter
    from sc_risk.adapter import SCRiskAdapter
except ImportError:
    from src.race_state.manager import RaceStateManager
    from src.lap_time.adapter import LapTimeAdapter
    from src.tyre_deg.adapter import TyreDegAdapter
    from src.sc_risk.adapter import SCRiskAdapter

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data_fastf1_v1"
LAPS_DIR = DATA_DIR / "laps"
DEFAULT_OUTPUT_DIR = DATA_DIR / "opponent_model"


class OpponentDatasetBuilder:
    """
    Builds and caches opponent modeling training, validation, and test datasets.
    """

    def __init__(
        self,
        laps_dir: str | Path = LAPS_DIR,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    ):
        self.laps_dir = Path(laps_dir)
        self.output_dir = Path(output_dir)
        self.cache_dir = self.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Lazy model references
        self._lt_adapter: Optional[LapTimeAdapter] = None
        self._tyre_adapter: Optional[TyreDegAdapter] = None
        self._sc_adapter: Optional[SCRiskAdapter] = None
        self._sc_models: Optional[tuple] = None

    def _init_models(self) -> None:
        """Initialize all three foundational models."""
        if self._lt_adapter is None:
            self._lt_adapter = LapTimeAdapter()
        if self._tyre_adapter is None:
            self._tyre_adapter = TyreDegAdapter()
        if self._sc_adapter is None:
            sc_prior_csv = PROJECT_ROOT / "models" / "SC Estimation" / "sc_vsc_historical_prior.csv"
            self._sc_adapter = SCRiskAdapter(str(sc_prior_csv))

            logit_sc = joblib.load(PROJECT_ROOT / "models" / "SC Estimation" / "sc_risk_logit_model_sc.joblib")
            logit_vsc = joblib.load(PROJECT_ROOT / "models" / "SC Estimation" / "sc_risk_logit_model_vsc.joblib")
            X_sc = joblib.load(PROJECT_ROOT / "models" / "SC Estimation" / "sc_risk_logit_features_sc.joblib")
            X_vsc = joblib.load(PROJECT_ROOT / "models" / "SC Estimation" / "sc_risk_logit_features_vsc.joblib")
            self._sc_models = (logit_sc, logit_vsc, X_sc, X_vsc)

    def process_race(self, race_csv_path: Path) -> pd.DataFrame:
        """
        Process a single race CSV into opponent model feature rows.
        Uses cached parquet if available.
        """
        year = int(race_csv_path.parent.name)
        grand_prix_raw = race_csv_path.stem
        race_id = f"{year}_{grand_prix_raw}"
        cache_file = self.cache_dir / f"{race_id}.parquet"

        if cache_file.exists():
            return pd.read_parquet(cache_file)

        self._init_models()
        assert self._lt_adapter is not None
        assert self._tyre_adapter is not None
        assert self._sc_adapter is not None
        assert self._sc_models is not None

        logit_sc, logit_vsc, X_sc, X_vsc = self._sc_models

        df_raw = pd.read_csv(race_csv_path)
        df_filtered = compute_pit_targets_and_filters(df_raw)

        # Pre-index valid targets
        valid_dict: dict[tuple[str, int], tuple[bool, int]] = {}
        for _, row in df_filtered.iterrows():
            d_code = str(row["Driver"]).strip().upper()
            l_num = int(row["LapNumber"])
            valid_dict[(d_code, l_num)] = (
                bool(row["is_valid_sample"]),
                int(row["target_pits_next_lap"]),
            )

        total_laps = int(df_filtered["LapNumber"].max()) if not df_filtered.empty else 55
        manager = RaceStateManager({
            "year": year,
            "grand_prix": grand_prix_raw.replace("_", " "),
            "total_laps_expected": total_laps,
            "race_id": race_id,
        })

        laps = sorted(df_filtered["LapNumber"].unique())
        race_rows: list[dict[str, Any]] = []

        for lap in laps:
            lap_rows = df_filtered[df_filtered["LapNumber"] == lap].to_dict("records")
            manager.update_from_rows(lap_rows)
            manager.commit_lap(lap)
            state = manager.state

            # 1. SC Risk batch (once per lap)
            try:
                H_t = self._sc_adapter.build_H_t(state)
                X_curr_sc = pd.DataFrame([H_t], columns=X_sc).fillna(0)
                X_curr_vsc = pd.DataFrame([H_t], columns=X_vsc).fillna(0)
                h_t_sc = float(logit_sc.predict_proba(X_curr_sc)[0, 1])
                h_t_vsc = float(logit_vsc.predict_proba(X_curr_vsc)[0, 1])
                p_sc_h1 = float(1.0 - (1.0 - h_t_sc))
                p_vsc_h1 = float(1.0 - (1.0 - h_t_vsc))
                p_sc_h3 = float(1.0 - (1.0 - h_t_sc) ** 3)
            except Exception:
                p_sc_h1, p_vsc_h1, p_sc_h3 = 0.02, 0.01, 0.05

            # 2. Lap Time batch (once per lap)
            try:
                pred_lts = self._lt_adapter.predict_all(state)
            except Exception:
                pred_lts = {}

            # 3. Tyre Deg batch (once per lap)
            try:
                pred_degs = self._tyre_adapter.predict_all(state)
                self._tyre_adapter.observe_lap(state)
            except Exception:
                pred_degs = {}

            # Field median pace
            valid_lts = [v for v in pred_lts.values() if v > 40.0]
            field_median_lt = float(np.median(valid_lts)) if valid_lts else 85.0

            # 4. Driver iteration
            cond = state.current_conditions
            for driver, p in state.participants.items():
                key = (driver, lap)
                if key not in valid_dict:
                    continue
                is_valid, target = valid_dict[key]
                if not is_valid:
                    continue

                comp = str(p.compound or "MEDIUM").strip().upper()
                tyre_age = float(p.tyre_life or 0)
                gap_ahead = float(p.interval_to_position_ahead_seconds) if p.interval_to_position_ahead_seconds is not None else 0.0
                gap_behind = float(p.gap_behind_seconds) if p.gap_behind_seconds is not None else 50.0
                pos = int(p.position or 10)
                lt_pred = float(pred_lts.get(driver, p.rolling_3_lap_avg or 85.0))
                deg_pred = float(pred_degs.get(driver, 0.0))

                deg_pred_t1 = deg_pred + 0.05
                lt_pred_t1 = lt_pred + 0.05

                derived = compute_derived_features(
                    tyre_compound=comp,
                    tyre_age=tyre_age,
                    gap_ahead=gap_ahead,
                    gap_behind=gap_behind,
                    position=pos,
                    predicted_lap_time=lt_pred,
                    field_median_lap_time=field_median_lt,
                    predicted_degradation=deg_pred,
                    predicted_degradation_t1=deg_pred_t1,
                    p_sc_h1=p_sc_h1,
                )

                row_dict = {
                    "race_id": race_id,
                    "year": year,
                    "grand_prix": grand_prix_raw.replace("_", " "),
                    "driver": driver,
                    "lap_number": lap,
                    "remaining_laps": max(0, total_laps - lap),
                    "race_progress_fraction": float(lap / max(1, total_laps)),
                    "position": pos,
                    "tyre_compound": comp,
                    "compound_SOFT": 1 if comp == "SOFT" else 0,
                    "compound_MEDIUM": 1 if comp == "MEDIUM" else 0,
                    "compound_HARD": 1 if comp == "HARD" else 0,
                    "tyre_age": int(tyre_age),
                    "tyre_age_squared": float(tyre_age ** 2),
                    "laps_since_last_pit": int(p.laps_since_last_pit or tyre_age),
                    "pit_count": int(p.pit_count or 0),
                    "gap_ahead": gap_ahead,
                    "gap_behind": gap_behind,
                    "last_lap_time": float(p.last_lap_time_seconds or 85.0),
                    "rolling_3_lap_avg": float(p.rolling_3_lap_avg or 85.0),
                    "is_safety_car": 1 if getattr(cond, "has_safety_car", False) else 0,
                    "is_vsc": 1 if getattr(cond, "has_vsc", False) else 0,
                    "track_temp": float(getattr(cond, "track_temp", 30.0) or 30.0),
                    "air_temp": float(getattr(cond, "air_temp", 25.0) or 25.0),
                    "rainfall": 1 if getattr(cond, "rainfall", False) else 0,
                    "predicted_lap_time": lt_pred,
                    "predicted_degradation": deg_pred,
                    "predicted_lap_time_t1": lt_pred_t1,
                    "predicted_degradation_t1": deg_pred_t1,
                    "p_sc_h1": p_sc_h1,
                    "p_vsc_h1": p_vsc_h1,
                    "p_sc_h3": p_sc_h3,
                    **derived,
                    "target": target,
                }
                race_rows.append(row_dict)

        df_out = pd.DataFrame(race_rows)
        if not df_out.empty:
            df_out.to_parquet(cache_file, index=False)
        return df_out

    def build_dataset_for_seasons(self, seasons: list[int], max_races_per_season: int | None = None) -> pd.DataFrame:
        """
        Build or load concatenated dataset for a list of seasons.
        """
        dfs = []
        for season in seasons:
            season_dir = self.laps_dir / str(season)
            if not season_dir.exists():
                continue
            race_files = sorted(list(season_dir.glob("*.csv")))
            if max_races_per_season is not None:
                race_files = race_files[:max_races_per_season]

            for rf in race_files:
                df_race = self.process_race(rf)
                if not df_race.empty:
                    dfs.append(df_race)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def generate_splits(
        self,
        train_seasons: list[int] = [2022, 2023],
        val_seasons: list[int] = [2024],
        test_seasons: list[int] = [2025],
        max_races_per_season: int | None = None,
        force_rebuild: bool = False,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Generate and persist train, validation, and test datasets.
        """
        train_path = self.output_dir / "train_dataset.parquet"
        val_path = self.output_dir / "val_dataset.parquet"
        test_path = self.output_dir / "test_dataset.parquet"

        if not force_rebuild and train_path.exists() and val_path.exists() and test_path.exists():
            df_train = pd.read_parquet(train_path)
            df_val = pd.read_parquet(val_path)
            df_test = pd.read_parquet(test_path)
            return df_train, df_val, df_test

        df_train = self.build_dataset_for_seasons(train_seasons, max_races_per_season=max_races_per_season)
        df_val = self.build_dataset_for_seasons(val_seasons, max_races_per_season=max_races_per_season)
        df_test = self.build_dataset_for_seasons(test_seasons, max_races_per_season=max_races_per_season)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not df_train.empty:
            df_train.to_parquet(train_path, index=False)
        if not df_val.empty:
            df_val.to_parquet(val_path, index=False)
        if not df_test.empty:
            df_test.to_parquet(test_path, index=False)

        return df_train, df_val, df_test
