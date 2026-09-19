"""
src/opponent_model/evaluate.py
==============================
Comprehensive evaluation suite for Opponent Modeling (Tasks 12–18).
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    log_loss,
    brier_score_loss,
)


def compute_calibration_errors(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> tuple[float, float, pd.DataFrame]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)
    with n_bins equal-width bins in [0, 1].

    Returns: (ece, mce, bin_df)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_data = []
    total_samples = len(y_true)
    weighted_error_sum = 0.0
    max_error = 0.0

    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= low) & (y_prob <= high)
        else:
            mask = (y_prob >= low) & (y_prob < high)

        count = int(np.sum(mask))
        if count > 0:
            conf = float(np.mean(y_prob[mask]))
            acc = float(np.mean(y_true[mask]))
            err = abs(acc - conf)
            weighted_error_sum += (count / total_samples) * err
            max_error = max(max_error, err)
        else:
            conf, acc, err = 0.0, 0.0, 0.0

        bin_data.append({
            "bin": f"[{low:.1f}, {high:.1f}]",
            "bin_center": (low + high) / 2.0,
            "count": count,
            "confidence": conf,
            "accuracy": acc,
            "calibration_error": err,
        })

    bin_df = pd.DataFrame(bin_data)
    ece = float(weighted_error_sum)
    mce = float(max_error)
    return ece, mce, bin_df


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Compute full evaluation metrics (Tasks 12–18).
    """
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Classification metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Class-specific precisions/recalls
    stay_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else 0.0
    stay_rec = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

    # ROC-AUC (handle single-class edge case)
    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.5

    # Probability quality metrics
    brier = brier_score_loss(y_true, y_prob)
    # Clamp for log loss
    eps = 1e-15
    y_prob_clamped = np.clip(y_prob, eps, 1.0 - eps)
    try:
        ll = float(log_loss(y_true, y_prob_clamped, labels=[0, 1]))
    except Exception:
        ll = float(-np.mean(y_true * np.log(y_prob_clamped) + (1 - y_true) * np.log(1 - y_prob_clamped)))
    prob_mae = float(np.mean(np.abs(y_prob - y_true)))
    prob_rmse = float(np.sqrt(np.mean((y_prob - y_true) ** 2)))

    # Calibration
    ece, mce, bin_df = compute_calibration_errors(y_true, y_prob, n_bins=10)

    # Action error metrics
    n = len(y_true)
    action_error = float((fp + fn) / n) if n > 0 else 0.0
    false_pit_rate = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    missed_pit_rate = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(prec),
        "recall": float(rec),
        "roc_auc": float(auc),
        "brier_score": float(brier),
        "log_loss": float(ll),
        "prob_mae": float(prob_mae),
        "prob_rmse": float(prob_rmse),
        "ece": float(ece),
        "mce": float(mce),
        "action_error": float(action_error),
        "false_pit_rate": float(false_pit_rate),
        "missed_pit_rate": float(missed_pit_rate),
        "pit_precision": float(prec),
        "pit_recall": float(rec),
        "stay_precision": float(stay_prec),
        "stay_recall": float(stay_rec),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "calibration_bins": bin_df,
    }


def compute_error_breakdowns(
    df: pd.DataFrame,
    y_true_col: str = "target",
    y_prob_col: str = "y_prob",
) -> dict[str, pd.DataFrame]:
    """
    Compute fine-grained error breakdowns (Task 16):
    - By tyre age bins: [0-5], [6-10], [11-15], [16-20], [21+]
    - By race phase: Early (0-33%), Middle (33-66%), Late (66-100%)
    - By compound: SOFT, MEDIUM, HARD
    - By track/SC status: Normal, VSC, SC
    - By predicted probability bin
    - By driver (Task 18)
    """
    df = df.copy()

    # 1. Tyre Age Bins
    age_bins = [-1, 5, 10, 15, 20, 999]
    age_labels = ["0-5", "6-10", "11-15", "16-20", "21+"]
    df["tyre_age_bin"] = pd.cut(df["tyre_age"], bins=age_bins, labels=age_labels)

    # 2. Race Phase
    phase_bins = [-0.01, 0.33, 0.66, 1.01]
    phase_labels = ["Early (0-33%)", "Middle (33-66%)", "Late (66-100%)"]
    df["race_phase"] = pd.cut(df["race_progress_fraction"], bins=phase_bins, labels=phase_labels)

    # 3. Track / SC Status
    def sc_status(row):
        if row.get("is_safety_car", 0) == 1:
            return "Safety Car (SC)"
        elif row.get("is_vsc", 0) == 1:
            return "Virtual Safety Car (VSC)"
        return "Normal Flag"

    df["sc_status"] = df.apply(sc_status, axis=1)

    # 4. Prob Bins
    prob_bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    prob_labels = ["[0.0-0.2]", "[0.2-0.4]", "[0.4-0.6]", "[0.6-0.8]", "[0.8-1.0]"]
    df["prob_bin"] = pd.cut(df[y_prob_col], bins=prob_bins, labels=prob_labels, right=False)

    # Aggregate function
    def agg_metrics(sub_df: pd.DataFrame) -> dict:
        if len(sub_df) == 0:
            return {}
        y_t = sub_df[y_true_col].values
        y_p = sub_df[y_prob_col].values
        m = compute_all_metrics(y_t, y_p)
        return {
            "samples": len(sub_df),
            "pit_count": int(np.sum(y_t)),
            "brier_score": m["brier_score"],
            "log_loss": m["log_loss"],
            "accuracy": m["accuracy"],
            "f1": m["f1"],
            "action_error": m["action_error"],
            "ece": m["ece"],
        }

    breakdowns = {}

    # By Tyre Age
    tyre_rows = []
    for lbl in age_labels:
        sub = df[df["tyre_age_bin"] == lbl]
        res = agg_metrics(sub)
        res["tyre_age_bin"] = lbl
        tyre_rows.append(res)
    breakdowns["by_tyre_age"] = pd.DataFrame(tyre_rows).set_index("tyre_age_bin")

    # By Race Phase
    phase_rows = []
    for lbl in phase_labels:
        sub = df[df["race_phase"] == lbl]
        res = agg_metrics(sub)
        res["race_phase"] = lbl
        phase_rows.append(res)
    breakdowns["by_race_phase"] = pd.DataFrame(phase_rows).set_index("race_phase")

    # By Compound
    comp_rows = []
    for comp in ["SOFT", "MEDIUM", "HARD"]:
        sub = df[df["tyre_compound"].str.upper() == comp]
        res = agg_metrics(sub)
        res["compound"] = comp
        comp_rows.append(res)
    breakdowns["by_compound"] = pd.DataFrame(comp_rows).set_index("compound")

    # By SC Status
    sc_rows = []
    for st in ["Normal Flag", "Virtual Safety Car (VSC)", "Safety Car (SC)"]:
        sub = df[df["sc_status"] == st]
        res = agg_metrics(sub)
        res["sc_status"] = st
        sc_rows.append(res)
    breakdowns["by_sc_status"] = pd.DataFrame(sc_rows).set_index("sc_status")

    # By Prob Bin
    prob_rows = []
    for pb in prob_labels:
        sub = df[df["prob_bin"] == pb]
        res = agg_metrics(sub)
        res["prob_bin"] = pb
        prob_rows.append(res)
    breakdowns["by_prob_bin"] = pd.DataFrame(prob_rows).set_index("prob_bin")

    # By Driver (Task 18)
    driver_rows = []
    driver_counts = df["driver"].value_counts()
    top_drivers = driver_counts[driver_counts >= 30].index.tolist()

    df_driver = df.copy()
    df_driver["driver_group"] = df_driver["driver"].apply(lambda d: d if d in top_drivers else "Other")

    for d in top_drivers + ["Other"]:
        sub = df_driver[df_driver["driver_group"] == d]
        if len(sub) > 0:
            res = agg_metrics(sub)
            res["driver"] = d
            driver_rows.append(res)
    breakdowns["by_driver"] = pd.DataFrame(driver_rows).set_index("driver")

    return breakdowns


def evaluate_multi_horizon(
    df: pd.DataFrame,
    p_calib_h1: np.ndarray,
    horizons: list[int] | None = None,
    thresholds: list[float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate multi-lap pit window probabilities across horizons (e.g. H=1, 3, 5 laps).

    Converts single-lap hazard h_t into multi-lap survival pit window probability:
        P(pit within H laps) = 1 - (1 - h_t)^H

    Computes ground truth pit occurrence within H laps on the held-out test set
    and returns:
    1. Summary DataFrame comparing H=1, H=3, H=5 at default (0.50) and optimal F1 thresholds.
    2. Detailed threshold sweep DataFrame showing Precision, Recall, and F1 across thresholds.
    """
    if horizons is None:
        horizons = [1, 3, 5]
    if thresholds is None:
        thresholds = [0.50, 0.40, 0.35, 0.30, 0.25, 0.20, 0.15, 0.10]

    df_eval = df.copy()
    df_eval["orig_idx"] = np.arange(len(df_eval))

    # Identify all pit signal laps (target == 1)
    pit_laps = df_eval[df_eval["target"] == 1][["race_id", "driver", "lap_number"]].rename(
        columns={"lap_number": "pit_signal_lap"}
    )

    # Align each lap with the next pit signal lap for that driver in that race
    merged = pd.merge_asof(
        df_eval.sort_values("lap_number"),
        pit_laps.sort_values("pit_signal_lap"),
        by=["race_id", "driver"],
        left_on="lap_number",
        right_on="pit_signal_lap",
        direction="forward",
    )

    # Restore original row ordering
    merged = merged.sort_values("orig_idx").reset_index(drop=True)
    merged["laps_to_pit_signal"] = merged["pit_signal_lap"] - merged["lap_number"]

    summary_rows = []
    sweep_rows = []

    p_h1 = np.asarray(p_calib_h1, dtype=float)

    for h in horizons:
        # Ground truth: pit occurs within next h laps (0 <= laps_to_pit_signal <= h - 1)
        y_true_h = (
            (merged["laps_to_pit_signal"] >= 0) & (merged["laps_to_pit_signal"] <= (h - 1))
        ).astype(int).values

        # Predicted multi-horizon probability via survival hazard formula
        p_h = 1.0 - (1.0 - p_h1) ** h

        # ROC-AUC and Brier Score
        try:
            auc = roc_auc_score(y_true_h, p_h)
        except ValueError:
            auc = 0.5
        brier = brier_score_loss(y_true_h, p_h)
        base_rate = float(np.mean(y_true_h))

        best_f1 = -1.0
        best_th = 0.5
        best_prec = 0.0
        best_rec = 0.0

        for th in thresholds:
            y_pred = (p_h >= th).astype(int)
            prec = precision_score(y_true_h, y_pred, zero_division=0)
            rec = recall_score(y_true_h, y_pred, zero_division=0)
            f1 = f1_score(y_true_h, y_pred, zero_division=0)
            acc = accuracy_score(y_true_h, y_pred)

            if f1 > best_f1:
                best_f1 = f1
                best_th = th
                best_prec = prec
                best_rec = rec

            sweep_rows.append({
                "Horizon": f"H = {h} ({h} Laps)",
                "Threshold": th,
                "Precision": prec,
                "Recall": rec,
                "F1 Score": f1,
                "Accuracy": acc,
            })

        # Record default threshold (0.50)
        y_pred_def = (p_h >= 0.50).astype(int)
        prec_def = precision_score(y_true_h, y_pred_def, zero_division=0)
        rec_def = recall_score(y_true_h, y_pred_def, zero_division=0)
        f1_def = f1_score(y_true_h, y_pred_def, zero_division=0)
        acc_def = accuracy_score(y_true_h, y_pred_def)

        summary_rows.append({
            "Horizon": f"{h}-Lap Window (H={h})",
            "Pit Base Rate": f"{base_rate:.2%}",
            "ROC-AUC": f"{auc:.4f}",
            "Brier Score": f"{brier:.4f}",
            "Def Prec (0.50)": f"{prec_def:.3f}",
            "Def Rec (0.50)": f"{rec_def:.3f}",
            "Def F1 (0.50)": f"{f1_def:.3f}",
            "Optimal Threshold": f"{best_th:.2f}",
            "Max F1 Score": f"{best_f1:.3f}",
            "Optimal Precision": f"{best_prec:.3f}",
            "Optimal Recall": f"{best_rec:.3f}",
            "Action Accuracy": f"{acc_def:.2%}",
        })

    df_summary = pd.DataFrame(summary_rows).set_index("Horizon")
    df_sweep = pd.DataFrame(sweep_rows)

    return df_summary, df_sweep
