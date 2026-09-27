"""
src/opponent_model/leakage_check.py
==================================
Verification suite to ensure zero data leakage across features and splits.
"""

from __future__ import annotations

import pandas as pd


FORBIDDEN_FEATURE_KEYWORDS = [
    "future",
    "pitted_next",
    "target",
    "final_position",
    "total_race_time",
    "next_compound",
    "pitintimeseconds",
    "pitouttimeseconds",
]


def verify_feature_leakage(feature_columns: list[str]) -> bool:
    """
    Ensure no future-looking columns or target variables exist in the feature set.
    """
    leaked_cols = []
    for col in feature_columns:
        col_lower = col.lower()
        for kw in FORBIDDEN_FEATURE_KEYWORDS:
            if kw in col_lower:
                leaked_cols.append((col, kw))

    if leaked_cols:
        raise ValueError(f"Data leakage detected! Forbidden features found: {leaked_cols}")
    return True


def verify_temporal_split_disjointness(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    race_id_col: str = "race_id",
) -> bool:
    """
    Verify that train, validation, and test splits are strictly disjoint by race/season.
    """
    train_races = set(df_train[race_id_col].unique())
    val_races = set(df_val[race_id_col].unique())
    test_races = set(df_test[race_id_col].unique())

    train_val_overlap = train_races.intersection(val_races)
    train_test_overlap = train_races.intersection(test_races)
    val_test_overlap = val_races.intersection(test_races)

    if train_val_overlap:
        raise ValueError(f"Leakage: Overlap between Train and Validation races: {train_val_overlap}")
    if train_test_overlap:
        raise ValueError(f"Leakage: Overlap between Train and Test races: {train_test_overlap}")
    if val_test_overlap:
        raise ValueError(f"Leakage: Overlap between Validation and Test races: {val_test_overlap}")

    return True


def run_all_leakage_checks(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, bool]:
    """Run all leakage assertions and return status report."""
    feature_ok = verify_feature_leakage(feature_cols)
    split_ok = verify_temporal_split_disjointness(df_train, df_val, df_test)

    # Check that target is binary 0 or 1
    for name, df in [("Train", df_train), ("Val", df_val), ("Test", df_test)]:
        if "target" in df.columns:
            vals = set(df["target"].dropna().unique())
            if not vals.issubset({0, 1}):
                raise ValueError(f"Target in {name} split is not strictly binary {0, 1}: {vals}")

    return {
        "features_clean": feature_ok,
        "splits_disjoint": split_ok,
        "targets_binary": True,
    }
