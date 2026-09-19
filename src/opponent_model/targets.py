"""
src/opponent_model/targets.py
=============================
Target generation and valid sample filtering for Opponent Modeling.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def compute_pit_targets_and_filters(df_race: pd.DataFrame) -> pd.DataFrame:
    """
    Process a single race CSV to compute:
    1. Target: 1 if driver pits on lap t+1 (PitInTimeSeconds notna on t+1), else 0
    2. Valid sample mask according to Task 7 exclusions:
       - Exclude lap 1 (formation / opening lap)
       - Exclude final lap of the race
       - Exclude pit-in lap (driver already pitting on lap t)
       - Exclude pit-out lap (driver just pitted)
       - Exclude wet/intermediate tyre compounds
       - Exclude retirements (driver does not complete lap t+1)
       - Exclude laps with NaN LapTimeSeconds
    """
    df = df_race.copy()

    # Ensure sorting
    sort_cols = ["Driver", "LapNumber"]
    if "Driver" in df.columns and "LapNumber" in df.columns:
        df = df.sort_values(sort_cols).reset_index(drop=True)

    # Detect total laps
    max_lap = df["LapNumber"].max() if "LapNumber" in df.columns else 0

    # Pit on lap t+1 detection
    # If PitInTimeSeconds on t+1 is notna, driver pits on next lap
    if "PitInTimeSeconds" in df.columns:
        df["target_pits_next_lap"] = (
            df.groupby("Driver")["PitInTimeSeconds"]
            .shift(-1)
            .notna()
            .astype(int)
        )
    else:
        df["target_pits_next_lap"] = 0

    # Next lap existence (detect retirement before lap t+1)
    df["next_lap_number"] = df.groupby("Driver")["LapNumber"].shift(-1)
    has_next_lap = df["next_lap_number"] == (df["LapNumber"] + 1)

    # Exclusions
    is_not_first_lap = df["LapNumber"] > 1
    is_not_final_lap = df["LapNumber"] < max_lap

    is_not_pit_in = True
    if "PitInTimeSeconds" in df.columns:
        is_not_pit_in = df["PitInTimeSeconds"].isna()

    is_not_pit_out = True
    if "PitOutTimeSeconds" in df.columns:
        is_not_pit_out = df["PitOutTimeSeconds"].isna()

    # Dry compounds only
    is_dry = True
    if "Compound" in df.columns:
        is_dry = ~df["Compound"].astype(str).str.upper().isin(["WET", "INTERMEDIATE", "NAN", "NONE", "UNKNOWN"])

    # Valid lap time
    has_lap_time = True
    if "LapTimeSeconds" in df.columns:
        has_lap_time = df["LapTimeSeconds"].notna() & (df["LapTimeSeconds"] > 40.0)

    # Combined valid mask
    df["is_valid_sample"] = (
        is_not_first_lap
        & is_not_final_lap
        & is_not_pit_in
        & is_not_pit_out
        & has_next_lap
        & is_dry
        & has_lap_time
    )

    return df
