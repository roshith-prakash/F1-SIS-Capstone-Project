"""
src/overtake/dataset.py
=======================
Battle extraction, label engineering, circuit difficulty derivation,
and train/validation/test split generation for the Overtake Model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .features import (
    CIRCUIT_ALIASES,
    CIRCUIT_PROPERTIES,
    DEFAULT_CIRCUIT_DIFFICULTY,
    FEATURES_MODEL_C,
    compute_overtake_features,
    resolve_circuit_name,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data_fastf1_v1"
LAPS_DIR = DATA_DIR / "laps"
OPPONENT_CACHE_DIR = DATA_DIR / "opponent_model" / "cache"
DEFAULT_OUTPUT_DIR = DATA_DIR / "overtake_model"
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models" / "Overtake Model"


class OvertakeDatasetBuilder:
    """
    Extracts wheel-to-wheel battles (< 1.5s gap), derives ground-effect
    circuit difficulty scores, and builds parquet splits.
    """

    def __init__(
        self,
        laps_dir: str | Path = LAPS_DIR,
        opponent_cache_dir: str | Path = OPPONENT_CACHE_DIR,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        model_dir: str | Path = DEFAULT_MODEL_DIR,
    ):
        self.laps_dir = Path(laps_dir)
        self.opponent_cache_dir = Path(opponent_cache_dir)
        self.output_dir = Path(output_dir)
        self.model_dir = Path(model_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.circuit_difficulty_map: dict[str, float] = {}

    def extract_battles_from_race(
        self,
        race_csv_path: str | Path,
        gap_threshold: float = 1.5,
    ) -> pd.DataFrame:
        """
        Extract all valid wheel-to-wheel battles and target labels from a single race CSV.
        """
        csv_path = Path(race_csv_path)
        if not csv_path.exists():
            return pd.DataFrame()

        try:
            df_raw = pd.read_csv(csv_path)
        except Exception:
            return pd.DataFrame()

        if df_raw.empty or "LapNumber" not in df_raw.columns:
            return pd.DataFrame()

        # Parse identifiers
        year = int(df_raw["Year"].iloc[0]) if "Year" in df_raw.columns else int(csv_path.parent.name)
        gp_name = str(df_raw["GrandPrix"].iloc[0]) if "GrandPrix" in df_raw.columns else csv_path.stem.replace("_", " ")
        location = str(df_raw["Location"].iloc[0]) if "Location" in df_raw.columns else gp_name
        circuit = resolve_circuit_name(location)
        race_id = f"{year}_{csv_path.stem}"

        # Check for pre-built foundational cache
        cache_path = self.opponent_cache_dir / f"{year}_{csv_path.stem}.parquet"
        cache_map: dict[tuple[str, int], dict[str, Any]] = {}
        if cache_path.exists():
            try:
                df_cache = pd.read_parquet(cache_path)
                for r in df_cache.to_dict("records"):
                    d_code = str(r.get("driver", "")).strip().upper()
                    l_num = int(r.get("lap_number", 0))
                    cache_map[(d_code, l_num)] = r
            except Exception:
                cache_map = {}

        max_lap = int(df_raw["LapNumber"].max())
        if max_lap < 5:
            return pd.DataFrame()

        # Detect safety car restart laps
        sc_status_per_lap: dict[int, int] = {}
        vsc_status_per_lap: dict[int, int] = {}
        for lap in range(1, max_lap + 1):
            sub = df_raw[df_raw["LapNumber"] == lap]
            has_sc = int((sub["HasSafetyCar"].fillna(0) == 1).any()) if "HasSafetyCar" in sub.columns else 0
            has_vsc = int((sub["HasVSC"].fillna(0) == 1).any()) if "HasVSC" in sub.columns else 0
            sc_status_per_lap[lap] = has_sc
            vsc_status_per_lap[lap] = has_vsc

        restart_laps: set[int] = set()
        for lap in range(2, max_lap + 1):
            if sc_status_per_lap.get(lap - 1, 0) == 1 and sc_status_per_lap.get(lap, 0) == 0:
                restart_laps.add(lap)
                restart_laps.add(lap + 1)

        battles: list[dict[str, Any]] = []

        # Iterate laps: exclude lap 1 (start chaos) and final lap (no outcome lap t+1)
        for lap in range(2, max_lap):
            lap_df = df_raw[df_raw["LapNumber"] == lap].copy()
            next_lap_df = df_raw[df_raw["LapNumber"] == lap + 1].copy()

            if lap_df.empty or next_lap_df.empty:
                continue

            lap_df = lap_df.sort_values("Position")
            next_lap_map = next_lap_df.set_index("Driver")["Position"].to_dict()
            next_pit_map = next_lap_df.set_index("Driver")["PitInTime"].notna().to_dict() if "PitInTime" in next_lap_df.columns else {}
            curr_pit_map = lap_df.set_index("Driver")["PitInTime"].notna().to_dict() if "PitInTime" in lap_df.columns else {}

            positions = sorted(lap_df["Position"].dropna().unique())
            for pos in positions:
                if pos <= 1:
                    continue

                car_b_rows = lap_df[lap_df["Position"] == pos]
                car_a_rows = lap_df[lap_df["Position"] == pos - 1]
                if car_b_rows.empty or car_a_rows.empty:
                    continue

                car_b = car_b_rows.iloc[0]
                car_a = car_a_rows.iloc[0]

                driver_b = str(car_b["Driver"]).strip().upper()
                driver_a = str(car_a["Driver"]).strip().upper()

                # Filter out cars currently pitting on lap t
                if curr_pit_map.get(driver_a, False) or curr_pit_map.get(driver_b, False):
                    continue

                gap = car_b.get("IntervalToPositionAheadSeconds")
                if pd.isna(gap) or gap >= gap_threshold or gap <= 0:
                    continue

                pos_b_next = next_lap_map.get(driver_b)
                pos_a_next = next_lap_map.get(driver_a)
                if pos_b_next is None or pos_a_next is None:
                    continue  # DNF or missing lap t+1

                # Exclude pit stops on lap t+1 (strategic pit cycle / undercut, not on-track pass)
                if next_pit_map.get(driver_a, False) or next_pit_map.get(driver_b, False):
                    continue

                # Target label: Driver B overtook Driver A on-track on lap t+1
                overtake = 1 if pos_b_next < pos_a_next else 0

                # Feature extraction: retrieve enriched features from cache if available
                cache_b = cache_map.get((driver_b, lap), {})
                cache_a = cache_map.get((driver_a, lap), {})

                lt_b = float(cache_b.get("last_lap_time", car_b.get("LapTimeSeconds", 85.0) or 85.0))
                lt_a = float(cache_a.get("last_lap_time", car_a.get("LapTimeSeconds", 85.0) or 85.0))
                if lt_b <= 40.0 or lt_b > 180.0:
                    lt_b = 85.0
                if lt_a <= 40.0 or lt_a > 180.0:
                    lt_a = 85.0
                pace_delta = float(lt_a - lt_b)

                r3_b = float(cache_b.get("rolling_3_lap_avg", lt_b))
                r3_a = float(cache_a.get("rolling_3_lap_avg", lt_a))
                r3_delta = float(r3_a - r3_b)

                comp_b = str(car_b.get("Compound", "MEDIUM")).strip().upper()
                comp_a = str(car_a.get("Compound", "MEDIUM")).strip().upper()

                age_b = float(car_b.get("TyreLife", 0) or 0)
                age_a = float(car_a.get("TyreLife", 0) or 0)
                tyre_age_delta = float(age_a - age_b)
                is_fresh_b = 1 if age_b <= 2 else 0

                is_sc = 1 if sc_status_per_lap.get(lap, 0) == 1 else 0
                is_vsc = 1 if vsc_status_per_lap.get(lap, 0) == 1 else 0
                is_sc_restart = 1 if lap in restart_laps else 0

                pred_lt_b = float(cache_b.get("predicted_lap_time", lt_b))
                pred_lt_a = float(cache_a.get("predicted_lap_time", lt_a))
                pred_deg_b = float(cache_b.get("predicted_degradation", 0.05))
                pred_deg_a = float(cache_a.get("predicted_degradation", 0.05))

                feats = compute_overtake_features(
                    gap_seconds=float(gap),
                    pace_delta=pace_delta,
                    tyre_age_delta=tyre_age_delta,
                    compound_behind=comp_b,
                    compound_ahead=comp_a,
                    is_fresh_tyre_behind=is_fresh_b,
                    circuit=circuit,
                    is_sc_restart=is_sc_restart,
                    is_sc=is_sc,
                    is_vsc=is_vsc,
                    rolling_3lap_pace_delta=r3_delta,
                    predicted_lap_time_behind=pred_lt_b,
                    predicted_lap_time_ahead=pred_lt_a,
                    predicted_deg_behind=pred_deg_b,
                    predicted_deg_ahead=pred_deg_a,
                    circuit_difficulty_map=self.circuit_difficulty_map,
                )

                row = {
                    "race_id": race_id,
                    "year": year,
                    "grand_prix": gp_name,
                    "circuit": circuit,
                    "lap_number": lap,
                    "driver_behind": driver_b,
                    "driver_ahead": driver_a,
                    "position_behind": pos,
                    "position_ahead": pos - 1,
                    **feats,
                    "target": overtake,
                }
                battles.append(row)

        return pd.DataFrame(battles)

    def build_raw_season_dataset(self, seasons: list[int]) -> pd.DataFrame:
        """Collect all battles across specified seasons."""
        dfs = []
        for season in seasons:
            season_dir = self.laps_dir / str(season)
            if not season_dir.exists():
                continue
            race_files = sorted(list(season_dir.glob("*.csv")))
            for rf in race_files:
                df_race = self.extract_battles_from_race(rf)
                if not df_race.empty:
                    dfs.append(df_race)

        if not dfs:
            return pd.DataFrame()
        return pd.concat(dfs, ignore_index=True)

    def derive_circuit_difficulty_scores(self, df_train: pd.DataFrame) -> dict[str, float]:
        """
        Derive Phase 3 Circuit Overtake Difficulty Score from 2022-2023 training data only:
        difficulty = 1.0 - (overtake_success_rate_per_circuit)
        """
        if df_train.empty:
            return {c: DEFAULT_CIRCUIT_DIFFICULTY for c in CIRCUIT_PROPERTIES}

        # Calculate success rate per circuit
        grouped = df_train.groupby("circuit")["target"].agg(["sum", "count"])
        diff_dict: dict[str, float] = {}

        global_rate = float(df_train["target"].mean())
        global_difficulty = float(1.0 - global_rate)

        for circuit, row in grouped.iterrows():
            count = int(row["count"])
            if count >= 15:
                rate = float(row["sum"] / count)
                # Bound between [0.4, 0.98] to prevent degenerate extremes
                difficulty = float(np.clip(1.0 - rate, 0.40, 0.98))
            else:
                difficulty = global_difficulty
            diff_dict[str(circuit)] = round(difficulty, 4)

        # Ensure all canonical circuits have an entry
        for c in CIRCUIT_PROPERTIES:
            if c not in diff_dict:
                diff_dict[c] = round(global_difficulty, 4)

        # Save to models directory
        diff_path = self.model_dir / "circuit_overtake_difficulty.json"
        with open(diff_path, "w", encoding="utf-8") as f:
            json.dump(diff_dict, f, indent=2)

        self.circuit_difficulty_map = diff_dict
        return diff_dict

    def generate_splits(
        self,
        train_seasons: list[int] = [2022, 2023],
        val_seasons: list[int] = [2024],
        test_seasons: list[int] = [2025],
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

        print("Building training dataset from raw laps...")
        df_train = self.build_raw_season_dataset(train_seasons)

        print("Deriving circuit difficulty table from 2022-2023 training data...")
        diff_map = self.derive_circuit_difficulty_scores(df_train)

        # Update train dataset with the derived circuit difficulties
        if not df_train.empty:
            df_train["circuit_overtake_difficulty"] = df_train["circuit"].map(
                lambda c: diff_map.get(c, DEFAULT_CIRCUIT_DIFFICULTY)
            )

        print("Building validation dataset (2024)...")
        df_val = self.build_raw_season_dataset(val_seasons)
        if not df_val.empty:
            df_val["circuit_overtake_difficulty"] = df_val["circuit"].map(
                lambda c: diff_map.get(c, DEFAULT_CIRCUIT_DIFFICULTY)
            )

        print("Building test dataset (2025)...")
        df_test = self.build_raw_season_dataset(test_seasons)
        if not df_test.empty:
            df_test["circuit_overtake_difficulty"] = df_test["circuit"].map(
                lambda c: diff_map.get(c, DEFAULT_CIRCUIT_DIFFICULTY)
            )

        # Save to parquets
        if not df_train.empty:
            df_train.to_parquet(train_path, index=False)
        if not df_val.empty:
            df_val.to_parquet(val_path, index=False)
        if not df_test.empty:
            df_test.to_parquet(test_path, index=False)

        return df_train, df_val, df_test
