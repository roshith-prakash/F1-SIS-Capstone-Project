"""
src/opponent_model/race_replay.py
=================================
Chronological race simulation and visual replay integrating RaceStateManager
with OpponentModel and foundational models.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.race_state.manager import RaceStateManager
from src.race_state.replay import load_csv_rows
from src.race_state.models import parse_lap_number, RaceState
from src.opponent_model.state import build_opponent_state
from src.opponent_model.model import OpponentModel

# F1 dark-theme styling palette
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


def simulate_race_with_opponent_model(
    csv_path: str | Path,
    opponent_model: OpponentModel | None = None,
    lt_adapter: Any = None,
    tyre_adapter: Any = None,
    sc_adapter: Any = None,
    sc_models: tuple[Any, Any, list[str], list[str]] | None = None,
    max_laps: int | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]], RaceStateManager]:
    """
    Chronologically replay a full race CSV through RaceStateManager while querying
    OpponentModel and foundational models on each completed lap.

    Returns:
    --------
    df_replay : pd.DataFrame
        Telemetry and prediction records for all drivers across all laps.
    snapshots : list[dict]
        Per-lap race state and opponent prediction snapshots.
    manager : RaceStateManager
        The final race state manager instance.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"Race CSV file not found: {csv_file}")

    if opponent_model is None:
        opponent_model = OpponentModel.load(enable_bayesian=True)

    rows = list(load_csv_rows(csv_file))
    grouped: dict[int, list[dict[str, Any]]] = {}
    for r in rows:
        lap = parse_lap_number(r.get("LapNumber"))
        if lap is not None:
            grouped.setdefault(lap, []).append(r)

    sorted_laps = sorted(grouped.keys())
    if max_laps is not None and max_laps > 0:
        sorted_laps = sorted_laps[:max_laps]

    manager = RaceStateManager({
        "total_laps_expected": len(grouped),
        "race_id": csv_file.stem,
    })

    records: list[dict[str, Any]] = []
    snapshots: list[dict[str, Any]] = []

    for lap in sorted_laps:
        # Ingest telemetry strictly for current lap
        manager.update_from_rows(grouped[lap])
        state = manager.commit_lap(lap)

        # Build H_t and SC probabilities if available
        cached_sc_probs = None
        if sc_adapter is not None and sc_models is not None:
            try:
                logit_sc, logit_vsc, X_sc, X_vsc = sc_models
                H_t = sc_adapter.build_H_t(state)
                X_curr_sc = pd.DataFrame([H_t], columns=X_sc).fillna(0)
                X_curr_vsc = pd.DataFrame([H_t], columns=X_vsc).fillna(0)
                h_t_sc = float(logit_sc.predict_proba(X_curr_sc)[0, 1])
                h_t_vsc = float(logit_vsc.predict_proba(X_curr_vsc)[0, 1])
                cached_sc_probs = {
                    "SC": {1: 1.0 - (1.0 - h_t_sc), 3: 1.0 - (1.0 - h_t_sc) ** 3},
                    "VSC": {1: 1.0 - (1.0 - h_t_vsc), 3: 1.0 - (1.0 - h_t_vsc) ** 3},
                }
            except Exception:
                cached_sc_probs = None

        lap_predictions: dict[str, dict[str, Any]] = {}

        # Query models for all active drivers
        for driver_code, participant in state.participants.items():
            if not participant.is_active:
                continue

            opp_state = build_opponent_state(
                state=state,
                driver=driver_code,
                lt_adapter=lt_adapter,
                tyre_adapter=tyre_adapter,
                sc_adapter=sc_adapter,
                sc_models=sc_models,
                cached_sc_probs=cached_sc_probs,
            )

            if opp_state is not None:
                pred = opponent_model.predict(opp_state)
                lap_predictions[driver_code] = pred

                # Determine tactical status
                p_next = pred["p_pit_next_lap"]
                p_3l = pred["p_pit_window_3laps"]
                if participant.is_pit_in_lap:
                    status = "PITTING (IN-LAP)"
                elif p_3l >= 0.50 or p_next >= 0.30:
                    status = "BOX BOX / IMMINENT"
                elif p_3l >= 0.25:
                    status = "PIT WINDOW OPEN"
                else:
                    status = "STAY"

                records.append({
                    "lap": lap,
                    "driver": driver_code,
                    "team": participant.team or "Unknown",
                    "position": participant.position,
                    "compound": participant.compound or "MEDIUM",
                    "tyre_age": participant.tyre_life or 0,
                    "stint": participant.stint or 1,
                    "pit_count": participant.pit_count or 0,
                    "gap_to_leader": participant.gap_to_leader_seconds,
                    "interval_ahead": participant.interval_to_position_ahead_seconds,
                    "last_lap_time": participant.last_lap_time_seconds,
                    "rolling_3_lap_avg": participant.rolling_3_lap_avg,
                    "is_pit_in_lap": participant.is_pit_in_lap,
                    "is_pit_out_lap": participant.is_pit_out_lap,
                    "p_pit_next": p_next,
                    "p_pit_3laps": p_3l,
                    "p_pit_5laps": pred["p_pit_window_5laps"],
                    "predicted_pace": pred["predicted_pace"],
                    "predicted_degradation": pred["predicted_degradation"],
                    "confidence": pred["confidence"],
                    "tactical_status": status,
                })

        # Observe lap for tyre adapter residual calibration
        if tyre_adapter is not None:
            try:
                tyre_adapter.observe_lap(state)
            except Exception:
                pass

        snapshots.append({
            "lap": lap,
            "state": state,
            "opponent_predictions": lap_predictions,
            "cached_sc_probs": cached_sc_probs,
        })

    df_replay = pd.DataFrame(records)
    return df_replay, snapshots, manager


def render_opponent_race_dashboard_html(
    state: RaceState,
    opponent_predictions: dict[str, dict[str, Any]],
    limit: int = 10,
) -> str:
    """
    Render a rich F1 dark-mode HTML dashboard snapshot displaying live race state
    integrated with Opponent Model pit probabilities and tactical statuses.
    """
    cond = state.current_conditions
    total_expected = f"/{state.total_laps_expected}" if state.total_laps_expected else ""
    rain_badge = "<b style='color:#63b3ed;'>RAIN</b>" if getattr(cond, "rainfall", False) else "<span style='color:#a0aec0;'>Dry</span>"
    sc_badge = "<b style='color:#ecc94b;'>ACTIVE</b>" if getattr(cond, "has_safety_car", False) else "<span style='color:#a0aec0;'>None</span>"

    entries = state.classification
    if limit > 0:
        entries = entries[:limit]

    def prob_bar(p: float, color: str = "#00f0ff") -> str:
        pct = min(100, max(0, p * 100))
        return (
            f'<span style="display:inline-block;width:45px;text-align:right;font-weight:600;">{p:.1%}</span>&nbsp;'
            f'<span style="display:inline-block;width:{pct * 0.6:.0f}px;min-width:3px;height:7px;'
            f'background:{color};border-radius:2px;"></span>'
        )

    def status_badge(status: str) -> str:
        if "BOX" in status or "PITTING" in status:
            return f'<span style="background:#dc2626;color:#ffffff;padding:2px 8px;border-radius:4px;font-weight:700;font-size:11px;">{status}</span>'
        elif "OPEN" in status:
            return f'<span style="background:#d97706;color:#ffffff;padding:2px 8px;border-radius:4px;font-weight:600;font-size:11px;">{status}</span>'
        return f'<span style="background:#1e293b;color:#94a3b8;padding:2px 8px;border-radius:4px;font-size:11px;">{status}</span>'

    rows_html = []
    for entry in entries:
        driver = entry.driver
        p = state.participants.get(driver)
        pred = opponent_predictions.get(driver, {})

        pos = str(entry.position) if entry.position is not None else "-"
        team = p.team or entry.team or "Unknown"
        compound = p.compound or "MEDIUM"
        tyre_life = f"{int(p.tyre_life or 0)}L"
        gap = f"+{entry.gap_to_leader_seconds:.1f}s" if entry.gap_to_leader_seconds and entry.position != 1 else ("Leader" if entry.position == 1 else "-")

        p_next = pred.get("p_pit_next_lap", 0.0)
        p_3l = pred.get("p_pit_window_3laps", 0.0)
        p_5l = pred.get("p_pit_window_5laps", 0.0)

        if p and p.is_pit_in_lap:
            status = "PITTING (IN-LAP)"
        elif p_3l >= 0.50 or p_next >= 0.30:
            status = "BOX BOX / IMMINENT"
        elif p_3l >= 0.25:
            status = "PIT WINDOW OPEN"
        else:
            status = "STAY"

        rows_html.append(f"""
        <tr>
            <td style="padding:6px 10px;font-weight:700;color:#f6ad55;">{pos}</td>
            <td style="padding:6px 10px;font-weight:700;color:#ffffff;">{driver}</td>
            <td style="padding:6px 10px;color:#94a3b8;">{team}</td>
            <td style="padding:6px 10px;">{compound} ({tyre_life})</td>
            <td style="padding:6px 10px;color:#cbd5e1;">{gap}</td>
            <td style="padding:6px 10px;">{prob_bar(p_next, '#ff1801')}</td>
            <td style="padding:6px 10px;">{prob_bar(p_3l, '#ffb800')}</td>
            <td style="padding:6px 10px;">{prob_bar(p_5l, '#00e676')}</td>
            <td style="padding:6px 10px;">{status_badge(status)}</td>
        </tr>
        """)

    table_body = "".join(rows_html)

    html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#12141a;color:#e0e0e0;padding:16px;border-radius:10px;margin-bottom:20px;box-shadow:0 4px 14px rgba(0,0,0,0.5);">
        <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:2px solid #e10600;padding-bottom:10px;margin-bottom:14px;">
            <div>
                <div style="font-size:20px;font-weight:700;color:#ffffff;">{state.grand_prix or 'Grand Prix'} ({state.year or ''}) &mdash; Live Opponent Strategy State</div>
                <div style="font-size:13px;color:#94a3b8;margin-top:3px;">
                    Tracked Drivers: <b>{len(state.participants)}</b> &bull; Track Status: <b>{cond.track_status or '1'}</b> &bull; Rain: {rain_badge} &bull; Safety Car: {sc_badge}
                </div>
            </div>
            <div style="background:#1e222d;padding:6px 14px;border-radius:6px;font-size:14px;font-weight:600;border:1px solid #2e3546;">
                LAP <span style="font-size:18px;color:#e10600;font-weight:800;">{state.current_lap or '-'}{total_expected}</span>
            </div>
        </div>

        <div style="display:flex;gap:10px;margin-bottom:12px;font-size:11px;">
            <span style="background:#1e293b;color:#ff1801;padding:3px 8px;border-radius:4px;border:1px solid #ff180155;">
                &#x1F4CA; <b>P(PIT Next)</b>: Single-Lap Hazard
            </span>
            <span style="background:#1e293b;color:#ffb800;padding:3px 8px;border-radius:4px;border:1px solid #ffb80055;">
                &#x23F3; <b>P(Pit 3L)</b>: 3-Lap Window Prob
            </span>
            <span style="background:#1e293b;color:#00e676;padding:3px 8px;border-radius:4px;border:1px solid #00e67655;">
                &#x1F52E; <b>P(Pit 5L)</b>: 5-Lap Strategic Prob
            </span>
        </div>

        <div style="overflow-x:auto;">
            <table style="width:100%;border-collapse:collapse;font-size:12.5px;text-align:left;background:#151821;">
                <thead>
                    <tr style="background:#1f2430;color:#94a3b8;border-bottom:1px solid #2d3748;">
                        <th style="padding:7px 10px;">Pos</th>
                        <th style="padding:7px 10px;">Driver</th>
                        <th style="padding:7px 10px;">Team</th>
                        <th style="padding:7px 10px;">Tyre (Life)</th>
                        <th style="padding:7px 10px;">Gap to Leader</th>
                        <th style="padding:7px 10px;color:#ff1801;">P(PIT Next)</th>
                        <th style="padding:7px 10px;color:#ffb800;">P(Pit 3L)</th>
                        <th style="padding:7px 10px;color:#00e676;">P(Pit 5L)</th>
                        <th style="padding:7px 10px;">Tactical Status</th>
                    </tr>
                </thead>
                <tbody>
                    {table_body}
                </tbody>
            </table>
        </div>
    </div>
    """
    return html


def plot_race_opponent_analysis(
    df_replay: pd.DataFrame,
    top_drivers: list[str] | None = None,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """
    Generate a 4-panel telemetry visualization of the entire race:
    1. 3-Lap Pit Window Probability Evolution (with actual pit stop markers)
    2. Track Position & Gap Evolution
    3. Tyre Age vs Pit Probability (The Tyre Cliff)
    4. Grid-Wide Pit Window Heatmap
    """
    if top_drivers is None:
        top_drivers = ["LEC", "PIA", "NOR", "SAI", "HAM", "VER"]

    # Filter to top drivers present
    available = df_replay["driver"].unique()
    drivers = [d for d in top_drivers if d in available]
    if not drivers:
        drivers = list(available[:6])

    driver_colors = {
        "LEC": "#e8002d",  # Ferrari Red
        "SAI": "#ff5722",  # Ferrari Amber
        "PIA": "#ff8700",  # McLaren Papaya
        "NOR": "#ffb703",  # McLaren Yellow
        "VER": "#1e40af",  # Red Bull Navy
        "HAM": "#00f0ff",  # Mercedes Cyan
        "RUS": "#00e676",  # Mercedes Green
    }

    fig = plt.figure(figsize=(22, 16), facecolor=F1_BG_COLOR)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.20)

    # -------------------------------------------------------------
    # Panel 1: Pit Window Probability Evolution
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    set_dark_f1_theme(ax1)
    ax1.set_title("A. Opponent Pit Window Probability P(Pit in 3 Laps) Evolution", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)

    for d in drivers:
        sub = df_replay[df_replay["driver"] == d].sort_values("lap")
        color = driver_colors.get(d, F1_TEXT)
        ax1.plot(sub["lap"], sub["p_pit_3laps"], label=d, color=color, lw=2.2)

        # Mark actual pit stops
        pit_laps = sub[sub["is_pit_in_lap"] == True]["lap"].values
        for pl in pit_laps:
            ax1.axvline(pl, color=color, linestyle=":", alpha=0.6, lw=1.2)
            ax1.scatter([pl], [sub[sub["lap"] == pl]["p_pit_3laps"].values[0]], color=color, s=80, zorder=5, edgecolors="#ffffff")

    ax1.axhline(0.25, color=F1_AMBER, linestyle="--", alpha=0.7, label="Pit Window Open Threshold (0.25)")
    ax1.axhline(0.50, color=F1_RED, linestyle="--", alpha=0.7, label="Imminent Pit Threshold (0.50)")
    ax1.set_xlabel("Race Lap", color=F1_TEXT)
    ax1.set_ylabel("P(Pit within 3 Laps)", color=F1_TEXT)
    ax1.legend(facecolor=F1_PANEL_COLOR, edgecolor="#2d3748", loc="upper left")

    # -------------------------------------------------------------
    # Panel 2: Track Position & Gap Evolution
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    set_dark_f1_theme(ax2)
    ax2.set_title("B. Track Position Progression (Lower is Better)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)

    for d in drivers:
        sub = df_replay[df_replay["driver"] == d].sort_values("lap")
        color = driver_colors.get(d, F1_TEXT)
        ax2.plot(sub["lap"], sub["position"], label=d, color=color, lw=2.2)

        # Mark pit in laps
        pit_laps = sub[sub["is_pit_in_lap"] == True]["lap"].values
        for pl in pit_laps:
            ax2.scatter([pl], [sub[sub["lap"] == pl]["position"].values[0]], color=color, s=80, zorder=5, marker="v")

    ax2.invert_yaxis()
    ax2.set_xlabel("Race Lap", color=F1_TEXT)
    ax2.set_ylabel("Position (P1 - P20)", color=F1_TEXT)
    ax2.legend(facecolor=F1_PANEL_COLOR, edgecolor="#2d3748", loc="center left")

    # -------------------------------------------------------------
    # Panel 3: Tyre Age vs Pit Probability (The Tyre Cliff)
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    set_dark_f1_theme(ax3)
    ax3.set_title("C. Tyre Age vs Pit Probability (The Degradation Cliff)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)

    for d in drivers:
        sub = df_replay[df_replay["driver"] == d]
        color = driver_colors.get(d, F1_TEXT)
        ax3.scatter(sub["tyre_age"], sub["p_pit_3laps"], label=d, color=color, alpha=0.6, s=30)

    # Trendline across all top drivers
    all_sub = df_replay[df_replay["driver"].isin(drivers)]
    age_bins = np.linspace(1, all_sub["tyre_age"].max(), 25)
    mean_probs = [all_sub[(all_sub["tyre_age"] >= a - 1) & (all_sub["tyre_age"] <= a + 1)]["p_pit_3laps"].mean() for a in age_bins]
    valid_mask = ~np.isnan(mean_probs)
    ax3.plot(age_bins[valid_mask], np.array(mean_probs)[valid_mask], color="#ffffff", lw=3.0, label="Field Trend")

    ax3.set_xlabel("Tyre Age (Laps on Set)", color=F1_TEXT)
    ax3.set_ylabel("P(Pit within 3 Laps)", color=F1_TEXT)
    ax3.legend(facecolor=F1_PANEL_COLOR, edgecolor="#2d3748", loc="upper left")

    # -------------------------------------------------------------
    # Panel 4: Grid-Wide Pit Window Heatmap
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    set_dark_f1_theme(ax4)
    ax4.set_title("D. Grid-Wide Pit Window Heatmap (Lap vs Driver)", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)

    # Pivot table: Driver x Lap with P(Pit 3L)
    pivot = df_replay[df_replay["driver"].isin(drivers)].pivot(index="driver", columns="lap", values="p_pit_3laps").fillna(0.0)
    pivot = pivot.reindex(drivers)

    im = ax4.imshow(pivot.values, aspect="auto", cmap="YlOrRd", interpolation="nearest", vmin=0.0, vmax=0.6)
    ax4.set_yticks(np.arange(len(drivers)))
    ax4.set_yticklabels(drivers, color=F1_TEXT, fontweight="bold")
    ax4.set_xlabel("Race Lap", color=F1_TEXT)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax4, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors=F1_TEXT)
    cbar.set_label("P(Pit in 3 Laps)", color=F1_TEXT)

    if save_dir:
        s_path = Path(save_dir)
        s_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(s_path / "opponent_model_race_simulation_replay.png", dpi=150, bbox_inches="tight", facecolor=F1_BG_COLOR)

    return fig


def get_driver_race_telemetry(df_replay: pd.DataFrame, driver_code: str) -> pd.DataFrame:
    """Extract and sort chronological telemetry and prediction records for an individual driver."""
    sub = df_replay[df_replay["driver"].str.upper() == driver_code.strip().upper()].sort_values("lap").reset_index(drop=True)
    if sub.empty:
        raise ValueError(f"Driver '{driver_code}' not found in replay dataframe.")
    return sub


def render_driver_opponent_card_html(
    df_driver: pd.DataFrame,
    lap: int,
) -> str:
    """
    Render a high-end telemetry and pit strategy card for an individual driver at lap t,
    displaying live position, compound, gaps, and Opponent Model pit probabilities.
    """
    row = df_driver[df_driver["lap"] == lap]
    if row.empty:
        return f"<div style='color:#fc8181;'>Lap {lap} not found for driver.</div>"
    r = row.iloc[0]

    def prob_bar(p: float, color: str = "#00f0ff") -> str:
        pct = min(100, max(0, p * 100))
        return (
            f'<span style="display:inline-block;width:50px;text-align:right;font-weight:600;">{p:.1%}</span>&nbsp;'
            f'<span style="display:inline-block;width:{pct * 0.8:.0f}px;min-width:3px;height:8px;'
            f'background:{color};border-radius:2px;"></span>'
        )

    status_color = "#dc2626" if "BOX" in r["tactical_status"] or "PITTING" in r["tactical_status"] else (
        "#d97706" if "OPEN" in r["tactical_status"] else "#10b981"
    )

    card_html = f"""
    <div style="font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;background:#12141a;color:#e0e0e0;padding:16px;border-radius:10px;margin-bottom:16px;box-shadow:0 4px 14px rgba(0,0,0,0.5);border-left:4px solid {status_color};">
        <div style="display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #2d3748;padding-bottom:8px;margin-bottom:12px;">
            <div>
                <span style="font-size:22px;font-weight:800;color:#ffffff;">{r['driver']}</span>
                <span style="font-size:14px;color:#94a3b8;margin-left:8px;">{r['team']} &bull; Stint {int(r['stint'])}</span>
            </div>
            <div style="text-align:right;">
                <span style="background:#1e222d;padding:4px 10px;border-radius:4px;font-size:13px;border:1px solid #2e3546;">
                    LAP <b style="color:#e10600;font-size:15px;">{int(r['lap'])}</b>
                </span>
                <span style="background:#1e222d;padding:4px 10px;border-radius:4px;font-size:13px;color:#f6ad55;font-weight:700;margin-left:6px;border:1px solid #2e3546;">
                    P{int(r['position'])}
                </span>
            </div>
        </div>

        <div style="display:grid;grid-template-columns:repeat(auto-fit, minmax(200px, 1fr));gap:12px;">
            <!-- Pit Probabilities -->
            <div style="background:#1b202c;padding:10px;border-radius:6px;border:1px solid #273042;">
                <div style="font-size:11px;font-weight:700;color:#e10600;text-transform:uppercase;margin-bottom:6px;border-bottom:1px solid #2a3346;padding-bottom:3px;">
                    Opponent Pit Probabilities
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">P(PIT Next)</span>
                    <span>{prob_bar(r['p_pit_next'], '#ff1801')}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">P(Pit 3L)</span>
                    <span>{prob_bar(r['p_pit_3laps'], '#ffb800')}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">P(Pit 5L)</span>
                    <span>{prob_bar(r['p_pit_5laps'], '#00e676')}</span>
                </div>
                <div style="margin-top:6px;text-align:right;">
                    <span style="background:{status_color}22;color:{status_color};border:1px solid {status_color};padding:2px 8px;border-radius:4px;font-weight:700;font-size:11px;">
                        {r['tactical_status']}
                    </span>
                </div>
            </div>

            <!-- Stint & Tyre Status -->
            <div style="background:#1b202c;padding:10px;border-radius:6px;border:1px solid #273042;">
                <div style="font-size:11px;font-weight:700;color:#00f0ff;text-transform:uppercase;margin-bottom:6px;border-bottom:1px solid #2a3346;padding-bottom:3px;">
                    Tyre &amp; Degradation Status
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">Compound</span>
                    <span style="font-weight:700;color:#ffffff;">{r['compound']}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">Tyre Age</span>
                    <span style="font-weight:700;color:#ffffff;">{int(r['tyre_age'])} laps</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">Pred Deg Deficit</span>
                    <span style="font-weight:700;color:#f43f5e;">+{r['predicted_degradation']:.2f} s</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">Pits Completed</span>
                    <span style="font-weight:700;color:#ffffff;">{int(r['pit_count'])}</span>
                </div>
            </div>

            <!-- Gaps & Track Position -->
            <div style="background:#1b202c;padding:10px;border-radius:6px;border:1px solid #273042;">
                <div style="font-size:11px;font-weight:700;color:#ffb800;text-transform:uppercase;margin-bottom:6px;border-bottom:1px solid #2a3346;padding-bottom:3px;">
                    Track Gaps &amp; Pace
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">Last Lap</span>
                    <span style="font-weight:700;color:#68d391;">{f"{r['last_lap_time']:.3f}s" if r['last_lap_time'] else "-"}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">Pred Next Pace</span>
                    <span style="font-weight:700;color:#38bdf8;">{f"{r['predicted_pace']:.3f}s" if r['predicted_pace'] else "-"}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">Gap Ahead</span>
                    <span style="font-weight:700;color:#ffffff;">{f"+{r['interval_ahead']:.1f}s" if r['interval_ahead'] and r['position'] != 1 else ("Leader" if r['position'] == 1 else "-")}</span>
                </div>
                <div style="display:flex;justify-content:space-between;font-size:12px;padding:2px 0;">
                    <span style="color:#94a3b8;">Gap to Leader</span>
                    <span style="font-weight:700;color:#ffffff;">{f"+{r['gap_to_leader']:.1f}s" if r['gap_to_leader'] and r['position'] != 1 else ("Leader" if r['position'] == 1 else "-")}</span>
                </div>
            </div>
        </div>
    </div>
    """
    return card_html


def plot_single_driver_opponent_analysis(
    df_replay: pd.DataFrame,
    driver_code: str = "PIA",
    grand_prix: str = "Italian Grand Prix",
    year: int = 2024,
    save_dir: str | Path | None = None,
) -> plt.Figure:
    """
    Generate a 4-panel telemetry visualization for an individual driver's race:
    1. Pit Window Probabilities P(Next), P(3L), P(5L) across all laps with pit stop markers.
    2. Actual Lap Pace vs AI Predicted Pace & Pure Tyre Degradation Deficit.
    3. Track Position & Gaps (Gap Ahead & Gap Behind) with Undercut Danger Zones.
    4. Stint Progression: Tyre Age vs P(Pit 3L) showing the cliff on each stint.
    """
    df_driver = get_driver_race_telemetry(df_replay, driver_code)

    fig = plt.figure(figsize=(22, 15), facecolor=F1_BG_COLOR)
    gs = fig.add_gridspec(2, 2, hspace=0.28, wspace=0.22)

    pit_in_rows = df_driver[df_driver["is_pit_in_lap"] == True]
    pit_laps = pit_in_rows["lap"].tolist()

    # -------------------------------------------------------------
    # Panel 1: Pit Window Probabilities Evolution
    # -------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    set_dark_f1_theme(ax1)
    ax1.set_title(f"A. {driver_code} — Pit Window Probabilities Evolution ({year} {grand_prix})", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)

    ax1.plot(df_driver["lap"], df_driver["p_pit_next"], label="P(PIT Next Lap)", color=F1_RED, lw=2.2)
    ax1.plot(df_driver["lap"], df_driver["p_pit_3laps"], label="P(Pit in 3 Laps)", color=F1_AMBER, lw=2.5)
    ax1.plot(df_driver["lap"], df_driver["p_pit_5laps"], label="P(Pit in 5 Laps)", color=F1_GREEN, lw=2.0, linestyle=":")

    ax1.axhline(0.25, color=F1_AMBER, linestyle="--", alpha=0.6, label="Window Open (0.25)")
    ax1.axhline(0.50, color=F1_RED, linestyle="--", alpha=0.6, label="Imminent Pit (0.50)")

    # Mark actual pit stops
    for idx, pl in enumerate(pit_laps):
        p_row = df_driver[df_driver["lap"] == pl].iloc[0]
        ax1.axvline(pl, color="#ffffff", linestyle="--", lw=1.5, alpha=0.8)
        ax1.scatter([pl], [p_row["p_pit_3laps"]], color=F1_RED, s=120, zorder=6, edgecolors="#ffffff", lw=1.5)
        ax1.text(pl + 0.8, p_row["p_pit_3laps"] + 0.03, f"Pit Stop {idx+1}\n(Lap {pl})", color="#ffffff", fontsize=10, fontweight="bold")

    ax1.set_xlabel("Race Lap", color=F1_TEXT)
    ax1.set_ylabel("Probability P(PIT)", color=F1_TEXT)
    ax1.set_ylim(-0.02, max(0.85, df_driver["p_pit_5laps"].max() + 0.10))
    ax1.legend(facecolor=F1_PANEL_COLOR, edgecolor="#2d3748", loc="upper left")

    # -------------------------------------------------------------
    # Panel 2: Pace, AI Forecast & Tyre Degradation Deficit
    # -------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    set_dark_f1_theme(ax2)
    ax2.set_title(f"B. {driver_code} — Lap Pace vs AI Forecast & Tyre Degradation Deficit", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)

    # Filter out in/out lap times for pace plotting
    valid_pace = df_driver[(df_driver["is_pit_in_lap"] == False) & (df_driver["is_pit_out_lap"] == False) & (df_driver["lap"] > 1)]

    l1 = ax2.plot(valid_pace["lap"], valid_pace["last_lap_time"], label="Actual Lap Time (s)", color=F1_CYAN, lw=2.0)
    l2 = ax2.plot(valid_pace["lap"], valid_pace["predicted_pace"], label="AI Predicted Pace (s)", color="#ffffff", lw=2.0, linestyle="--")

    ax2_twin = ax2.twinx()
    l3 = ax2_twin.plot(df_driver["lap"], df_driver["predicted_degradation"], label="Pure Tyre Deg Deficit (s)", color=F1_PURPLE, lw=2.2)

    set_dark_f1_theme(ax2_twin)
    for spine in ax2_twin.spines.values():
        spine.set_color("#2d3748")

    for pl in pit_laps:
        ax2.axvline(pl, color="#ffffff", linestyle="--", lw=1.2, alpha=0.5)

    ax2.set_xlabel("Race Lap", color=F1_TEXT)
    ax2.set_ylabel("Lap Time (Seconds)", color=F1_CYAN, fontweight="bold")
    ax2_twin.set_ylabel("Tyre Deg Deficit (s)", color=F1_PURPLE, fontweight="bold")

    lines = l1 + l2 + l3
    labels = [l.get_label() for l in lines]
    ax2.legend(lines, labels, facecolor=F1_PANEL_COLOR, edgecolor="#2d3748", loc="upper left")

    # -------------------------------------------------------------
    # Panel 3: Track Position & Tactical Gaps
    # -------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    set_dark_f1_theme(ax3)
    ax3.set_title(f"C. {driver_code} — Track Position & Tactical Gap to Leader", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)

    l4 = ax3.plot(df_driver["lap"], df_driver["position"], color=F1_AMBER, lw=2.5, label="Track Position")
    ax3.invert_yaxis()

    ax3_twin = ax3.twinx()
    l5 = ax3_twin.plot(df_driver["lap"], df_driver["gap_to_leader"].fillna(0.0), color=F1_CYAN, lw=2.0, linestyle="-.", label="Gap to Leader (s)")

    set_dark_f1_theme(ax3_twin)
    for spine in ax3_twin.spines.values():
        spine.set_color("#2d3748")

    for pl in pit_laps:
        ax3.axvline(pl, color="#ffffff", linestyle="--", lw=1.2, alpha=0.5)

    ax3.set_xlabel("Race Lap", color=F1_TEXT)
    ax3.set_ylabel("Position (P1 - P20)", color=F1_AMBER, fontweight="bold")
    ax3_twin.set_ylabel("Gap to Leader (Seconds)", color=F1_CYAN, fontweight="bold")

    lines2 = l4 + l5
    labels2 = [l.get_label() for l in lines2]
    ax3.legend(lines2, labels2, facecolor=F1_PANEL_COLOR, edgecolor="#2d3748", loc="center left")

    # -------------------------------------------------------------
    # Panel 4: Stint Breakdown: Tyre Age vs Pit Probability
    # -------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    set_dark_f1_theme(ax4)
    ax4.set_title(f"D. {driver_code} — Stint Progression & The Tyre Age Cliff", color=F1_TEXT, fontsize=12, fontweight="bold", pad=12)

    stints = df_driver["stint"].unique()
    stint_colors = ["#ff1801", "#ffb800", "#00f0ff", "#00e676"]

    for idx, s in enumerate(stints):
        sub_s = df_driver[df_driver["stint"] == s]
        comp = sub_s["compound"].iloc[0] if len(sub_s) > 0 else "UNKNOWN"
        col = stint_colors[idx % len(stint_colors)]
        ax4.plot(
            sub_s["tyre_age"],
            sub_s["p_pit_3laps"],
            marker="o",
            lw=2.2,
            color=col,
            label=f"Stint {int(s)} ({comp}, Laps {int(sub_s['lap'].min())}-{int(sub_s['lap'].max())})",
        )

    ax4.axhline(0.25, color=F1_AMBER, linestyle="--", alpha=0.6, label="Window Open (0.25)")
    ax4.set_xlabel("Tyre Age on Set (Laps)", color=F1_TEXT)
    ax4.set_ylabel("P(Pit within 3 Laps)", color=F1_TEXT)
    ax4.legend(facecolor=F1_PANEL_COLOR, edgecolor="#2d3748", loc="upper left")

    if save_dir:
        s_path = Path(save_dir)
        s_path.mkdir(parents=True, exist_ok=True)
        fig.savefig(s_path / f"opponent_model_single_driver_{driver_code.lower()}_analysis.png", dpi=150, bbox_inches="tight", facecolor=F1_BG_COLOR)

    return fig
