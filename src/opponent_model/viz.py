"""
src/opponent_model/viz.py
=========================
Visualization suite for Opponent Modeling (Task 28).
Produces all 14 required figures with sleek F1 dark-mode styling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix

# Premium dark theme styling
plt.style.use("dark_background")
F1_BG_COLOR = "#0e1117"
F1_PANEL_COLOR = "#161b26"
F1_RED = "#ff1801"
F1_CYAN = "#00f0ff"
F1_AMBER = "#ffb800"
F1_GREEN = "#00e676"
F1_PURPLE = "#b388ff"
F1_TEXT = "#e0e6ed"


def set_dark_f1_theme(ax: plt.Axes) -> None:
    """Apply F1 telemetry aesthetic to a matplotlib Axes."""
    ax.set_facecolor(F1_PANEL_COLOR)
    ax.tick_params(colors=F1_TEXT, which="both")
    for spine in ax.spines.values():
        spine.set_color("#2d3748")
        spine.set_linewidth(1.2)
    ax.grid(True, color="#2d3748", linestyle="--", alpha=0.5)


# 1. Confusion Matrix
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=F1_BG_COLOR)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    set_dark_f1_theme(ax)
    ax.set_title("1. Confusion Matrix (Test Set: 2025)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred: STAY", "Pred: PIT"], color=F1_TEXT)
    ax.set_yticklabels(["True: STAY", "True: PIT"], color=F1_TEXT)

    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            color = "white" if count > cm.max() / 2 else F1_CYAN
            ax.text(j, i, f"{count:,}", ha="center", va="center", color=color, fontsize=14, fontweight="bold")
    return ax


# 2. ROC Curve
def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, roc_auc: float, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=F1_BG_COLOR)
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    ax.plot(fpr, tpr, color=F1_CYAN, lw=2.5, label=f"Calibrated XGBoost (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="#718096", lw=1.5, linestyle="--", label="Random Baseline (AUC = 0.500)")
    set_dark_f1_theme(ax)
    ax.set_title("2. ROC Curve (Discrimination Ability)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("False Positive Rate", color=F1_TEXT)
    ax.set_ylabel("True Positive Rate (Recall)", color=F1_TEXT)
    ax.legend(loc="lower right", facecolor=F1_PANEL_COLOR, edgecolor="#2d3748")
    return ax


# 3. Precision-Recall Curve
def plot_pr_curve(y_true: np.ndarray, y_prob: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=F1_BG_COLOR)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ax.plot(recall, precision, color=F1_RED, lw=2.5, label="Precision-Recall Curve")
    base_rate = float(np.mean(y_true))
    ax.axhline(base_rate, color="#718096", linestyle="--", label=f"Empirical Prior ({base_rate:.1%})")
    set_dark_f1_theme(ax)
    ax.set_title("3. Precision-Recall Curve (Imbalanced PIT Class)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Recall (Coverage)", color=F1_TEXT)
    ax.set_ylabel("Precision (Accuracy of Pit Alerts)", color=F1_TEXT)
    ax.legend(loc="upper right", facecolor=F1_PANEL_COLOR, edgecolor="#2d3748")
    return ax


# 4. Calibration / Reliability Diagram
def plot_calibration_diagram(
    raw_bins: pd.DataFrame,
    calib_bins: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=F1_BG_COLOR)
    ax.plot([0, 1], [0, 1], color="#718096", linestyle="--", label="Perfect Calibration")
    ax.plot(raw_bins["confidence"], raw_bins["accuracy"], marker="o", color=F1_AMBER, lw=2, label="Raw XGBoost")
    ax.plot(calib_bins["confidence"], calib_bins["accuracy"], marker="s", color=F1_GREEN, lw=2.5, label="Platt Calibrated")
    set_dark_f1_theme(ax)
    ax.set_title("4. Reliability Diagram (Calibration Curve)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Mean Predicted Probability", color=F1_TEXT)
    ax.set_ylabel("Observed Pit Frequency", color=F1_TEXT)
    ax.legend(loc="upper left", facecolor=F1_PANEL_COLOR, edgecolor="#2d3748")
    return ax


# 5. Probability Distribution
def plot_probability_distribution(y_true: np.ndarray, y_prob: np.ndarray, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=F1_BG_COLOR)
    ax.hist(y_prob[y_true == 0], bins=30, alpha=0.6, color=F1_CYAN, label="True STAY", density=True)
    ax.hist(y_prob[y_true == 1], bins=30, alpha=0.7, color=F1_RED, label="True PIT", density=True)
    set_dark_f1_theme(ax)
    ax.set_title("5. Predicted Probability Distribution by Class", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Predicted P(PIT)", color=F1_TEXT)
    ax.set_ylabel("Density", color=F1_TEXT)
    ax.legend(loc="upper right", facecolor=F1_PANEL_COLOR, edgecolor="#2d3748")
    return ax


# 6. Brier Score Comparison
def plot_brier_comparison(scores_dict: dict[str, float], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=F1_BG_COLOR)
    models = list(scores_dict.keys())
    scores = list(scores_dict.values())
    colors = [F1_AMBER if "Baseline" in m else F1_GREEN for m in models]
    bars = ax.barh(models, scores, color=colors, height=0.55)
    set_dark_f1_theme(ax)
    ax.set_title("6. Brier Score Comparison (Lower is Better)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Brier Score", color=F1_TEXT)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.001, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", color=F1_TEXT, fontsize=10)
    return ax


# 7. Log Loss Comparison
def plot_log_loss_comparison(log_loss_dict: dict[str, float], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=F1_BG_COLOR)
    models = list(log_loss_dict.keys())
    losses = list(log_loss_dict.values())
    colors = [F1_AMBER if "Baseline" in m else F1_CYAN for m in models]
    bars = ax.barh(models, losses, color=colors, height=0.55)
    set_dark_f1_theme(ax)
    ax.set_title("7. Log Loss Comparison (Cross-Entropy)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Log Loss", color=F1_TEXT)
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.4f}", va="center", color=F1_TEXT, fontsize=10)
    return ax


# 8. Ablation Study Comparison
def plot_ablation_comparison(ablation_results: dict[str, dict[str, float]], ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5), facecolor=F1_BG_COLOR)
    models = list(ablation_results.keys())
    roc_aucs = [ablation_results[m]["roc_auc"] for m in models]
    f1s = [ablation_results[m]["f1"] for m in models]

    x = np.arange(len(models))
    width = 0.35
    b1 = ax.bar(x - width / 2, roc_aucs, width, label="ROC-AUC", color=F1_CYAN)
    b2 = ax.bar(x + width / 2, f1s, width, label="F1 Score", color=F1_PURPLE)

    set_dark_f1_theme(ax)
    ax.set_title("8. Feature Ablation: Model A vs Model B vs Model C", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(["Model A\n(Race State)", "Model B\n(+ Foundational)", "Model C\n(+ Derived)"], color=F1_TEXT)
    ax.legend(facecolor=F1_PANEL_COLOR, edgecolor="#2d3748")
    return ax


# 9. Bayesian vs Base Comparison
def plot_bayesian_comparison(
    y_true: np.ndarray,
    p_base: np.ndarray,
    p_bayes: np.ndarray,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5), facecolor=F1_BG_COLOR)
    brier_base = float(np.mean((p_base - y_true) ** 2))
    brier_bayes = float(np.mean((p_bayes - y_true) ** 2))

    labels = ["Base Calibrated ML", "+ Bayesian Layer"]
    vals = [brier_base, brier_bayes]
    bars = ax.bar(labels, vals, color=[F1_AMBER, F1_GREEN], width=0.45)
    set_dark_f1_theme(ax)
    ax.set_title("9. Bayesian Layer Impact on Brier Score", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Brier Score (Lower is Better)", color=F1_TEXT)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.0005, f"{h:.5f}", ha="center", color=F1_TEXT, fontsize=11, fontweight="bold")
    return ax


# 10. Error by Tyre Age
def plot_error_by_tyre_age(df_tyre_age: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=F1_BG_COLOR)
    bars = ax.bar(df_tyre_age.index, df_tyre_age["brier_score"], color=F1_CYAN, width=0.55)
    set_dark_f1_theme(ax)
    ax.set_title("10. Brier Score by Tyre Age Bins", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Tyre Age (Laps on Set)", color=F1_TEXT)
    ax.set_ylabel("Brier Score", color=F1_TEXT)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f"{h:.3f}", ha="center", color=F1_TEXT, fontsize=10)
    return ax


# 11. Error by Race Phase
def plot_error_by_race_phase(df_phase: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=F1_BG_COLOR)
    bars = ax.bar(df_phase.index, df_phase["brier_score"], color=F1_PURPLE, width=0.5)
    set_dark_f1_theme(ax)
    ax.set_title("11. Brier Score by Race Phase", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Race Phase", color=F1_TEXT)
    ax.set_ylabel("Brier Score", color=F1_TEXT)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f"{h:.3f}", ha="center", color=F1_TEXT, fontsize=10)
    return ax


# 12. Error by SC Status
def plot_error_by_sc_status(df_sc: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 4), facecolor=F1_BG_COLOR)
    bars = ax.bar(df_sc.index, df_sc["brier_score"], color=F1_AMBER, width=0.45)
    set_dark_f1_theme(ax)
    ax.set_title("12. Brier Score by Safety Car / Flag Status", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Brier Score", color=F1_TEXT)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.001, f"{h:.3f}", ha="center", color=F1_TEXT, fontsize=10)
    return ax


# 13. Error by Driver
def plot_error_by_driver(df_driver: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 4), facecolor=F1_BG_COLOR)
    df_sorted = df_driver.sort_values("brier_score")
    bars = ax.bar(df_sorted.index, df_sorted["brier_score"], color=F1_GREEN, width=0.6)
    set_dark_f1_theme(ax)
    ax.set_title("13. Driver-Level Brier Score (Top Drivers vs Other)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_ylabel("Brier Score", color=F1_TEXT)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    return ax


# 14. Monte Carlo Sensitivity Curve
def plot_mc_sensitivity(df_mc: pd.DataFrame, ax: Optional[plt.Axes] = None) -> plt.Axes:
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 5), facecolor=F1_BG_COLOR)
    assumed_p = df_mc.index.values
    dec_err = [float(v.replace("%", "")) for v in df_mc["Decision Error Rate"].values]

    ax.plot(assumed_p, dec_err, marker="o", color=F1_RED, lw=2.5, label="Decision Error Rate (%)")
    ax.axvline(0.70, color=F1_GREEN, linestyle="--", lw=2, label="True Opponent P(PIT) = 0.70")
    set_dark_f1_theme(ax)
    ax.set_title("14. Monte Carlo Sensitivity: Decision Error vs Probability Error", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Assumed Opponent P(PIT)", color=F1_TEXT)
    ax.set_ylabel("Optimal Strategy Decision Error Rate (%)", color=F1_TEXT)
    ax.legend(loc="upper left", facecolor=F1_PANEL_COLOR, edgecolor="#2d3748")
    return ax


def generate_all_14_figures(
    y_test: np.ndarray,
    raw_probs: np.ndarray,
    calib_probs: np.ndarray,
    bayes_probs: np.ndarray,
    ablation_results: dict[str, dict[str, float]],
    baselines_brier: dict[str, float],
    baselines_logloss: dict[str, float],
    breakdowns: dict[str, pd.DataFrame],
    df_mc: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """
    Assemble and display all 14 figures in a cohesive multi-panel dashboard.
    """
    fig = plt.figure(figsize=(24, 28), facecolor=F1_BG_COLOR)
    gs = fig.add_gridspec(5, 3, hspace=0.35, wspace=0.25)

    y_pred = (calib_probs >= 0.5).astype(int)
    from .evaluate import compute_calibration_errors
    _, _, raw_bins = compute_calibration_errors(y_test, raw_probs)
    _, _, calib_bins = compute_calibration_errors(y_test, calib_probs)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_confusion_matrix(y_test, y_pred, ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    from sklearn.metrics import roc_auc_score
    plot_roc_curve(y_test, calib_probs, roc_auc_score(y_test, calib_probs), ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_pr_curve(y_test, calib_probs, ax3)

    ax4 = fig.add_subplot(gs[1, 0])
    plot_calibration_diagram(raw_bins, calib_bins, ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    plot_probability_distribution(y_test, calib_probs, ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    plot_brier_comparison(baselines_brier, ax6)

    ax7 = fig.add_subplot(gs[2, 0])
    plot_log_loss_comparison(baselines_logloss, ax7)

    ax8 = fig.add_subplot(gs[2, 1])
    plot_ablation_comparison(ablation_results, ax8)

    ax9 = fig.add_subplot(gs[2, 2])
    plot_bayesian_comparison(y_test, calib_probs, bayes_probs, ax9)

    ax10 = fig.add_subplot(gs[3, 0])
    plot_error_by_tyre_age(breakdowns["by_tyre_age"], ax10)

    ax11 = fig.add_subplot(gs[3, 1])
    plot_error_by_race_phase(breakdowns["by_race_phase"], ax11)

    ax12 = fig.add_subplot(gs[3, 2])
    plot_error_by_sc_status(breakdowns["by_sc_status"], ax12)

    ax13 = fig.add_subplot(gs[4, 0:2])
    plot_error_by_driver(breakdowns["by_driver"], ax13)

    ax14 = fig.add_subplot(gs[4, 2])
    plot_mc_sensitivity(df_mc, ax14)

    if save_dir:
        s_path = Path(save_dir)
        s_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(s_path / "opponent_model_14_visualizations.png", dpi=150, bbox_inches="tight", facecolor=F1_BG_COLOR)

    return fig


def plot_multi_horizon_analysis(
    df_sweep: pd.DataFrame,
    df_mc_multi: pd.DataFrame,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """
    Plot 3-panel dashboard for Multi-Horizon Pit Window Analysis:
    1. F1 Score vs Decision Threshold across H=1, 3, 5 laps.
    2. Precision vs Recall Tradeoff across H=1, 3, 5 laps.
    3. Monte Carlo Strategy Engine Outcomes (Undercut Success & Time Regret across H=1, 3, 5).
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 5.5), facecolor=F1_BG_COLOR)
    plt.subplots_adjust(wspace=0.28)

    horizon_colors = {
        "H = 1 (1 Laps)": F1_RED,
        "H = 3 (3 Laps)": F1_AMBER,
        "H = 5 (5 Laps)": F1_GREEN,
    }

    # Panel 1: F1 Score vs Threshold
    for h_label, color in horizon_colors.items():
        sub = df_sweep[df_sweep["Horizon"] == h_label].sort_values("Threshold")
        if len(sub) > 0:
            ax1.plot(
                sub["Threshold"],
                sub["F1 Score"],
                marker="o",
                lw=2.5,
                color=color,
                label=f"{h_label}",
            )
    set_dark_f1_theme(ax1)
    ax1.set_title("A. F1 Score vs Threshold (H=1, 3, 5 Laps)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax1.set_xlabel("Decision Threshold", color=F1_TEXT)
    ax1.set_ylabel("F1 Score", color=F1_TEXT)
    ax1.legend(facecolor=F1_PANEL_COLOR, edgecolor="#2d3748")

    # Panel 2: Precision-Recall Tradeoff
    for h_label, color in horizon_colors.items():
        sub = df_sweep[df_sweep["Horizon"] == h_label].sort_values("Recall")
        if len(sub) > 0:
            ax2.plot(
                sub["Recall"],
                sub["Precision"],
                marker="s",
                lw=2.5,
                color=color,
                label=f"{h_label}",
            )
    set_dark_f1_theme(ax2)
    ax2.set_title("B. Precision vs Recall Curve by Horizon", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax2.set_xlabel("Recall (Pit Detection Rate)", color=F1_TEXT)
    ax2.set_ylabel("Precision (Pit Alert Accuracy)", color=F1_TEXT)
    ax2.legend(facecolor=F1_PANEL_COLOR, edgecolor="#2d3748")

    # Panel 3: Monte Carlo Strategy Outcomes
    horizons = df_mc_multi.index.tolist()
    win_rates = [float(v.replace("%", "")) for v in df_mc_multi["Undercut Success Rate"].values]
    regrets = [float(v.replace(" s", "")) for v in df_mc_multi["Avg Time Regret vs Optimal"].values]

    x = np.arange(len(horizons))
    width = 0.35

    ax3_twin = ax3.twinx()
    b1 = ax3.bar(x - width / 2, win_rates, width, label="Undercut Success (%)", color=F1_CYAN)
    b2 = ax3_twin.bar(x + width / 2, regrets, width, label="Time Regret (s)", color=F1_PURPLE)

    set_dark_f1_theme(ax3)
    ax3_twin.tick_params(colors=F1_TEXT)
    for spine in ax3_twin.spines.values():
        spine.set_color("#2d3748")

    ax3.set_title("C. Monte Carlo Strategy Engine Outcomes", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(["H=1\n(1-Lap)", "H=3\n(3-Lap)", "H=5\n(5-Lap)"], color=F1_TEXT)
    ax3.set_ylabel("Undercut Defense Success Rate (%)", color=F1_CYAN, fontweight="bold")
    ax3_twin.set_ylabel("Avg Time Regret vs Optimal (s)", color=F1_PURPLE, fontweight="bold")

    # Value labels
    for bar in b1:
        h = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2, h + 1.0, f"{h:.1f}%", ha="center", color=F1_CYAN, fontsize=10, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax3_twin.text(bar.get_x() + bar.get_width() / 2, h + 0.1, f"{h:.2f}s", ha="center", color=F1_PURPLE, fontsize=10, fontweight="bold")

    if save_dir:
        s_path = Path(save_dir)
        s_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(s_path / "opponent_model_multi_horizon_analysis.png", dpi=150, bbox_inches="tight", facecolor=F1_BG_COLOR)

    return fig
