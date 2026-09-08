"""
Helper module for Jupyter Notebook visual rendering of F1 Race State.
Provides rich HTML-formatted tables and driver detail cards for interactive display.
"""

from __future__ import annotations

from typing import Any
import pandas as pd

from .models import RaceState, ParticipantState
from .display import (
    format_lap_time,
    format_total_time,
    format_gap,
    format_pos_delta,
    format_tyre_info,
    format_in_out,
    format_pits,
    format_speeds,
)

F1_THEME_CSS = """
<style>
.f1-container {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #e0e0e0;
    background-color: #12141a;
    padding: 16px;
    border-radius: 10px;
    margin-bottom: 20px;
    box-shadow: 0 4px 14px rgba(0, 0, 0, 0.5);
}

.f1-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 2px solid #e10600;
    padding-bottom: 10px;
    margin-bottom: 14px;
}

.f1-title {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.5px;
}

.f1-badge {
    background-color: #1e222d;
    padding: 6px 14px;
    border-radius: 6px;
    font-size: 14px;
    font-weight: 600;
    border: 1px solid #2e3546;
}

.f1-conditions-bar {
    display: flex;
    gap: 16px;
    background: #191d26;
    padding: 8px 14px;
    border-radius: 6px;
    font-size: 12px;
    margin-bottom: 14px;
    color: #a0aec0;
}

.f1-conditions-bar span b {
    color: #ffffff;
}

.f1-section-title {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    color: #e10600;
    letter-spacing: 1px;
    margin: 12px 0 6px 0;
}

.f1-table-wrapper {
    overflow-x: auto;
    margin-bottom: 14px;
}

.f1-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12.5px;
    text-align: left;
    background-color: #151821;
}

.f1-table th {
    background-color: #1f2430;
    color: #94a3b8;
    font-weight: 600;
    padding: 7px 10px;
    border-bottom: 1px solid #2d3748;
    white-space: nowrap;
}

.f1-table td {
    padding: 6px 10px;
    border-bottom: 1px solid #1e2330;
    color: #cbd5e1;
    white-space: nowrap;
}

.f1-table tr:hover {
    background-color: #212634;
}

.pos-1 { color: #f6ad55 !important; font-weight: 700; }
.pos-2 { color: #cbd5e0 !important; font-weight: 700; }
.pos-3 { color: #ed8936 !important; font-weight: 700; }

.chg-gain { color: #48bb78 !important; font-weight: 600; }
.chg-loss { color: #f56565 !important; font-weight: 600; }
.chg-same { color: #718096; }

/* Driver Card */
.driver-card {
    background: #151821;
    border: 1px solid #2d3748;
    border-left: 4px solid #e10600;
    border-radius: 8px;
    padding: 16px;
    color: #e2e8f0;
}

.driver-card-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-bottom: 1px solid #2d3748;
    padding-bottom: 10px;
    margin-bottom: 12px;
}

.driver-name {
    font-size: 22px;
    font-weight: 700;
    color: #ffffff;
}

.driver-team {
    font-size: 14px;
    color: #94a3b8;
}

.card-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
    gap: 12px;
}

.card-block {
    background: #1b202c;
    padding: 12px;
    border-radius: 6px;
    border: 1px solid #273042;
}

.card-block-title {
    font-size: 11px;
    font-weight: 700;
    color: #e10600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 8px;
    border-bottom: 1px solid #2a3346;
    padding-bottom: 4px;
}

.stat-row {
    display: flex;
    justify-content: space-between;
    font-size: 12.5px;
    padding: 3px 0;
}

.stat-label {
    color: #94a3b8;
}

.stat-val {
    font-weight: 600;
    color: #f1f5f9;
}
</style>
"""


def build_grid_summary_dataframes(
    state: RaceState,
    limit: int | None = 10,
    lt_preds: dict[str, float] | None = None,
    tyre_preds: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build pandas DataFrames for Table 1 (Timing) and Table 2 (Strategy/Telemetry),
    optionally enriched with Lap Time and Tyre Degradation model predictions."""
    entries = state.classification
    if limit is not None and limit > 0:
        entries = entries[:limit]

    t1_rows: list[dict[str, Any]] = []
    t2_rows: list[dict[str, Any]] = []

    for entry in entries:
        driver = entry.driver
        p = state.participants.get(driver, ParticipantState(driver=driver))

        # Pos styling
        pos_display = str(entry.position) if entry.position is not None else "-"
        chg_display = format_pos_delta(p.position_change)

        # Table 1: Classification & Timing
        row_t1: dict[str, Any] = {
            "Pos": pos_display,
            "Chg": chg_display,
            "Driver": driver,
            "Team": p.team or entry.team or "Unknown",
            "Last Lap": format_lap_time(p.last_lap_time_seconds),
        }

        # Add Lap Time prediction if available
        if lt_preds is not None:
            pred_lt = lt_preds.get(driver)
            if pred_lt is not None:
                row_t1["Pred Next"] = format_lap_time(pred_lt)
            elif p.compound in ("INTERMEDIATE", "WET"):
                row_t1["Pred Next"] = "Wet (N/A)"
            else:
                row_t1["Pred Next"] = "-"

        row_t1.update({
            "Total Time": format_total_time(p.total_race_time_seconds),
            "S1": f"{p.sector_1_time_seconds:.2f}" if p.sector_1_time_seconds is not None else "-",
            "S2": f"{p.sector_2_time_seconds:.2f}" if p.sector_2_time_seconds is not None else "-",
            "S3": f"{p.sector_3_time_seconds:.2f}" if p.sector_3_time_seconds is not None else "-",
            "Gap (s)": format_gap(entry.gap_to_leader_seconds, is_leader=(entry.position == 1)),
            "Int (s)": format_gap(entry.interval_to_position_ahead_seconds, is_leader=(entry.position == 1)),
            "Behind (s)": format_gap(entry.gap_behind_seconds or p.gap_behind_seconds),
        })
        t1_rows.append(row_t1)

        # Table 2: Tyre, Strategy & Telemetry
        recent_formatted = [format_lap_time(t) for t in p.recent_lap_times]
        recent_str = f"[{', '.join(recent_formatted)}]" if recent_formatted else "[]"

        row_t2: dict[str, Any] = {
            "Pos": pos_display,
            "Driver": driver,
            "Tyre (Life)": format_tyre_info(p.compound, p.tyre_life, p.fresh_tyre),
        }

        # Add Tyre Degradation pure deficit prediction if available
        if tyre_preds is not None:
            pred_deg = tyre_preds.get(driver)
            if pred_deg is not None:
                row_t2["Pred Deg"] = f"+{pred_deg:.2f}s" if pred_deg >= 0 else f"{pred_deg:.2f}s"
            else:
                row_t2["Pred Deg"] = "-"

        row_t2.update({
            "Stint": f"S{int(p.stint)}" if p.stint is not None else "S1",
            "Pits (Last)": format_pits(p.pit_count, p.last_pit_lap),
            "In/Out": format_in_out(p.is_pit_in_lap, p.is_pit_out_lap),
            "Roll 3L": format_lap_time(p.rolling_3_lap_avg),
            "Speeds (I1/I2/FL/ST)": format_speeds(p.speed_i1, p.speed_i2, p.speed_fl, p.speed_st),
            "Recent Laps (Last 5)": recent_str,
        })
        t2_rows.append(row_t2)

    return pd.DataFrame(t1_rows), pd.DataFrame(t2_rows)


def render_grid_summary_html(
    state: RaceState,
    limit: int | None = 10,
    lt_preds: dict[str, float] | None = None,
    tyre_preds: dict[str, float] | None = None,
) -> str:
    """Render the full grid summary view with both Table 1 and Table 2 in rich HTML."""
    df_timing, df_strategy = build_grid_summary_dataframes(
        state, limit=limit, lt_preds=lt_preds, tyre_preds=tyre_preds
    )
    cond = state.current_conditions

    total_expected = f"/{state.total_laps_expected}" if state.total_laps_expected else ""
    rain_badge = "<b style='color:#63b3ed;'>RAIN</b>" if cond.rainfall else "<span style='color:#a0aec0;'>Dry</span>"
    sc_badge = "<b style='color:#ecc94b;'>ACTIVE</b>" if cond.has_safety_car else "<span style='color:#a0aec0;'>None</span>"

    header_html = f"""
    <div class="f1-container">
        <div class="f1-header">
            <div>
                <div class="f1-title">{state.grand_prix or 'Grand Prix'} ({state.year or ''})</div>
                <div style="font-size: 13px; color: #94a3b8; margin-top: 3px;">
                    Tracked Drivers: <b>{len(state.participants)}</b>
                </div>
            </div>
            <div class="f1-badge">
                LAP <span style="font-size: 18px; color: #e10600; font-weight: 800;">{state.current_lap or '-'}{total_expected}</span>
            </div>
        </div>

        <div class="f1-conditions-bar">
            <span>Track Status: <b>{cond.track_status or '1'}</b></span>
            <span>Air: <b>{cond.air_temp or '-'}°C</b></span>
            <span>Track: <b>{cond.track_temp or '-'}°C</b></span>
            <span>Rainfall: {rain_badge}</span>
            <span>Safety Car: {sc_badge}</span>
        </div>

        <div class="f1-section-title">Table 1: Classification & Timing {'(with Lap Time Model Predictions)' if lt_preds else ''}</div>
        <div class="f1-table-wrapper">
            {df_timing.to_html(index=False, classes='f1-table')}
        </div>

        <div class="f1-section-title">Table 2: Tyre, Strategy & Telemetry {'(with Tyre Deg Model Predictions)' if tyre_preds else ''}</div>
        <div class="f1-table-wrapper">
            {df_strategy.to_html(index=False, classes='f1-table')}
        </div>
    </div>
    """
    return F1_THEME_CSS + header_html


def _prob_bar(p: float, color: str) -> str:
    """Helper to render an inline colored probability progress bar."""
    pct = min(100, max(0, p * 100))
    return (
        f'<span style="display:inline-block;width:60px;text-align:right;font-weight:600;">'
        f'{p:.1%}</span>&nbsp;'
        f'<span style="display:inline-block;width:{pct:.0f}px;min-width:3px;height:9px;'
        f'background:{color};border-radius:3px;"></span>'
    )


def render_multi_model_dashboard_html(
    state: RaceState,
    sc_probs: dict[str, dict[int, float]] | None = None,
    H_t: dict[str, Any] | None = None,
    lt_preds: dict[str, float] | None = None,
    tyre_preds: dict[str, float] | None = None,
    limit: int | None = 10,
) -> str:
    """Render the unified multi-model AI dashboard displaying Safety Car risks,
    Lap Time predictions, and pure Tyre Degradation pace deficits alongside live race state.
    """
    df_timing, df_strategy = build_grid_summary_dataframes(
        state, limit=limit, lt_preds=lt_preds, tyre_preds=tyre_preds
    )
    cond = state.current_conditions

    total_expected = f"/{state.total_laps_expected}" if state.total_laps_expected else ""
    rain_badge = "<b style='color:#63b3ed;'>RAIN</b>" if cond.rainfall else "<span style='color:#a0aec0;'>Dry</span>"
    sc_active = cond.has_safety_car
    vsc_active = cond.has_vsc
    flag_str = "SAFETY CAR" if sc_active else ("VSC" if vsc_active else ("RAIN" if cond.rainfall else "GREEN"))
    flag_color = "#e53935" if sc_active else ("#fb8c00" if vsc_active else ("#38bdf8" if cond.rainfall else "#10b981"))

    # Safety Car & VSC probability rows
    sc_panel_html = ""
    if sc_probs and "SC" in sc_probs and "VSC" in sc_probs:
        horizons = sorted(sc_probs["SC"].keys())
        sc_rows = "".join(
            f"<tr>"
            f"<td style='padding:2px 8px;color:#94a3b8;'>Next {n} Lap{'s' if n > 1 else ''}</td>"
            f"<td style='padding:2px 8px;'>{_prob_bar(sc_probs['SC'][n], '#ef4444')}</td>"
            f"<td style='padding:2px 8px;'>{_prob_bar(sc_probs['VSC'][n], '#f59e0b')}</td>"
            f"</tr>"
            for n in horizons
        )
        features_str = ""
        if H_t:
            features_str = " &bull; ".join(
                f"<span style='color:#94a3b8;'>{k}=<b style='color:#f1f5f9;'>{v:.2f}</b></span>"
                for k, v in H_t.items()
                if k in ("close_battles", "field_spread", "sector_anomaly_score", "recent_retirements")
            )

        sc_panel_html = f"""
        <div style="background:#181c26;border-radius:8px;padding:10px 14px;margin-bottom:14px;border-left:4px solid {flag_color};">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                <div style="font-size:13px;font-weight:700;letter-spacing:0.5px;color:#ffffff;">
                    SAFETY CAR &amp; VSC RISK ESTIMATION <span style="margin-left:8px;padding:2px 8px;border-radius:4px;font-size:11px;background:{flag_color}22;color:{flag_color};border:1px solid {flag_color};">{flag_str}</span>
                </div>
                <div style="font-size:11px;color:#94a3b8;">
                    {features_str}
                </div>
            </div>
            <table style="border-collapse:collapse;font-size:12px;font-family:monospace;margin-top:4px;">
                <tr>
                    <th style="text-align:left;padding:2px 8px;color:#94a3b8;">Forecast Horizon</th>
                    <th style="text-align:left;padding:2px 8px;color:#ef4444;">P(Safety Car)</th>
                    <th style="text-align:left;padding:2px 8px;color:#f59e0b;">P(Virtual Safety Car)</th>
                </tr>
                {sc_rows}
            </table>
        </div>
        """

    # AI Model Status Badges
    model_badges = """
    <div style="display:flex;gap:8px;margin-bottom:12px;font-size:11px;">
        <span style="background:#1e293b;color:#38bdf8;padding:3px 8px;border-radius:4px;border:1px solid #0284c7;">
            &#x1F3CE; Lap Time Model: <b>XGBoost (Dry Base + Delta)</b>
        </span>
        <span style="background:#1e293b;color:#f43f5e;padding:3px 8px;border-radius:4px;border:1px solid #e11d48;">
            &#x1FA99; Tyre Deg Model: <b>XGBoost Causal Deg Regressor</b>
        </span>
        <span style="background:#1e293b;color:#fbbf24;padding:3px 8px;border-radius:4px;border:1px solid #d97706;">
            &#x1F6A8; SC Risk Model: <b>Logistic Hazard Ensemble</b>
        </span>
    </div>
    """

    header_html = f"""
    <div class="f1-container">
        <div class="f1-header">
            <div>
                <div class="f1-title">{state.grand_prix or 'Grand Prix'} ({state.year or ''}) &mdash; Multi-Model Race Intelligence</div>
                <div style="font-size: 13px; color: #94a3b8; margin-top: 3px;">
                    Tracked Drivers: <b>{len(state.participants)}</b> &bull; Track Status: <b>{cond.track_status or '1'}</b>
                </div>
            </div>
            <div class="f1-badge">
                LAP <span style="font-size: 18px; color: #e10600; font-weight: 800;">{state.current_lap or '-'}{total_expected}</span>
            </div>
        </div>

        {model_badges}

        {sc_panel_html}

        <div class="f1-conditions-bar">
            <span>Air Temp: <b>{cond.air_temp or '-'}°C</b></span>
            <span>Track Temp: <b>{cond.track_temp or '-'}°C</b></span>
            <span>Rainfall: {rain_badge}</span>
            <span>Active Safety Car: <b style="color:{flag_color};">{flag_str}</b></span>
        </div>

        <div class="f1-section-title">Table 1: Classification, Timing &amp; Lap Time Forecast</div>
        <div class="f1-table-wrapper">
            {df_timing.to_html(index=False, classes='f1-table')}
        </div>

        <div class="f1-section-title">Table 2: Tyre Degradation, Strategy &amp; Telemetry</div>
        <div class="f1-table-wrapper">
            {df_strategy.to_html(index=False, classes='f1-table')}
        </div>
    </div>
    """
    return F1_THEME_CSS + header_html


def render_driver_card_html(
    state: RaceState,
    driver_code: str,
    lt_pred: float | None = None,
    tyre_pred: float | None = None,
    sc_probs: dict[str, dict[int, float]] | None = None,
) -> str:
    """Render a high-end telemetry and pit strategy card for an individual driver,
    optionally augmented with AI predictions for Lap Time, Tyre Degradation, and SC risk."""
    code = driver_code.strip().upper()
    participant = state.participants.get(code)
    if participant is None:
        return f"<div style='color: #fc8181;'>Driver '{code}' not found in current race state (Lap {state.current_lap}).</div>"

    total_time_str = (
        format_total_time(participant.total_race_time_seconds)
        if participant.total_race_time_seconds is not None
        else "-"
    )
    total_expected = f"/{state.total_laps_expected}" if state.total_laps_expected else ""
    stint_num = int(participant.stint) if participant.stint is not None else 1
    recent_laps_str = ", ".join([format_lap_time(t) for t in participant.recent_lap_times]) or "-"

    speed_i1 = f"{participant.speed_i1:.0f} km/h" if participant.speed_i1 else "-"
    speed_i2 = f"{participant.speed_i2:.0f} km/h" if participant.speed_i2 else "-"
    speed_fl = f"{participant.speed_fl:.0f} km/h" if participant.speed_fl else "-"
    speed_st = f"{participant.speed_st:.0f} km/h" if participant.speed_st else "-"

    fresh_str = "Yes (Fresh)" if participant.fresh_tyre is True else "No (Used)" if participant.fresh_tyre is False else "Unknown"
    pos_delta = format_pos_delta(participant.position_change)

    # Multi-Model AI Prediction stats
    pred_lt_str = (
        format_lap_time(lt_pred)
        if lt_pred is not None
        else ("Wet (N/A)" if participant.compound in ("INTERMEDIATE", "WET") else "-")
    )
    if lt_pred is not None and participant.last_lap_time_seconds:
        delta = lt_pred - participant.last_lap_time_seconds
        pace_delta_str = f"{delta:+.2f}s"
    else:
        pace_delta_str = "-"

    pred_deg_str = f"+{tyre_pred:.2f}s deficit" if tyre_pred is not None else "-"
    sc_risk_3l = f"{sc_probs['SC'].get(3, 0):.1%}" if sc_probs and "SC" in sc_probs else "-"

    ai_block_html = f"""
    <!-- AI Forecast & Strategy -->
    <div class="card-block" style="border-left: 3px solid #38bdf8;">
        <div class="card-block-title" style="color: #38bdf8;">AI Strategy &amp; Forecast</div>
        <div class="stat-row"><span class="stat-label">Pred Next Lap</span><span class="stat-val" style="color:#38bdf8;">{pred_lt_str}</span></div>
        <div class="stat-row"><span class="stat-label">Pred vs Last Delta</span><span class="stat-val">{pace_delta_str}</span></div>
        <div class="stat-row"><span class="stat-label">Pure Tyre Deg Deficit</span><span class="stat-val" style="color:#f43f5e;">{pred_deg_str}</span></div>
        <div class="stat-row"><span class="stat-label">Track SC Risk (3L)</span><span class="stat-val" style="color:#fbbf24;">{sc_risk_3l}</span></div>
        <div class="stat-row"><span class="stat-label">Stint Tyre Age</span><span class="stat-val">{int(participant.tyre_life or 0)} laps</span></div>
    </div>
    """

    card_html = f"""
    <div class="f1-container">
        <div class="driver-card">
            <div class="driver-card-header">
                <div>
                    <div class="driver-name">{participant.driver_full_name or code} <span style="color:#e10600;">#{participant.driver_number or ''}</span></div>
                    <div class="driver-team">{participant.team or 'Unknown Constructor'} &bull; Status: <b style="color: #48bb78;">{'ACTIVE' if participant.is_active else 'RETIRED'}</b></div>
                </div>
                <div class="f1-badge" style="text-align: right;">
                    <div>LAP <b>{state.current_lap or '-'}{total_expected}</b></div>
                    <div style="color: #f6ad55; font-size: 16px;">P{participant.position or '-'} ({pos_delta})</div>
                </div>
            </div>

            <div class="card-grid">
                <!-- Track Position & Gaps -->
                <div class="card-block">
                    <div class="card-block-title">Track Position &amp; Gaps</div>
                    <div class="stat-row"><span class="stat-label">Position</span><span class="stat-val">P{participant.position or '-'}</span></div>
                    <div class="stat-row"><span class="stat-label">Position Delta</span><span class="stat-val">{pos_delta}</span></div>
                    <div class="stat-row"><span class="stat-label">Gap to Leader</span><span class="stat-val">{format_gap(participant.gap_to_leader_seconds, is_leader=(participant.position == 1)) or 'Leader'}</span></div>
                    <div class="stat-row"><span class="stat-label">Interval Ahead</span><span class="stat-val">{format_gap(participant.interval_to_position_ahead_seconds, is_leader=(participant.position == 1)) or '-'}</span></div>
                    <div class="stat-row"><span class="stat-label">Gap Behind</span><span class="stat-val">{format_gap(participant.gap_behind_seconds)}</span></div>
                    <div class="stat-row"><span class="stat-label">Total Time</span><span class="stat-val">{total_time_str}</span></div>
                </div>

                <!-- Tyre & Pit Strategy -->
                <div class="card-block">
                    <div class="card-block-title">Tyre &amp; Pit Strategy</div>
                    <div class="stat-row"><span class="stat-label">Compound</span><span class="stat-val">{participant.compound or 'UNK'}</span></div>
                    <div class="stat-row"><span class="stat-label">Tyre Life</span><span class="stat-val">{int(participant.tyre_life) if participant.tyre_life is not None else 0} laps</span></div>
                    <div class="stat-row"><span class="stat-label">Fresh Tyre Set</span><span class="stat-val">{fresh_str}</span></div>
                    <div class="stat-row"><span class="stat-label">Current Stint</span><span class="stat-val">Stint {stint_num}</span></div>
                    <div class="stat-row"><span class="stat-label">Pits Completed</span><span class="stat-val">{participant.pit_count} (Last: L{participant.last_pit_lap or '-'})</span></div>
                    <div class="stat-row"><span class="stat-label">In/Out Lap</span><span class="stat-val">Pit-In: {participant.is_pit_in_lap} | Out: {participant.is_pit_out_lap}</span></div>
                </div>

                <!-- Lap Timing & Rolling Pace -->
                <div class="card-block">
                    <div class="card-block-title">Lap Timing &amp; Sectors</div>
                    <div class="stat-row"><span class="stat-label">Last Lap Time</span><span class="stat-val" style="color: #68d391;">{format_lap_time(participant.last_lap_time_seconds)}</span></div>
                    <div class="stat-row"><span class="stat-label">Sector 1</span><span class="stat-val">{f"{participant.sector_1_time_seconds:.2f}s" if participant.sector_1_time_seconds else "-"}</span></div>
                    <div class="stat-row"><span class="stat-label">Sector 2</span><span class="stat-val">{f"{participant.sector_2_time_seconds:.2f}s" if participant.sector_2_time_seconds else "-"}</span></div>
                    <div class="stat-row"><span class="stat-label">Sector 3</span><span class="stat-val">{f"{participant.sector_3_time_seconds:.2f}s" if participant.sector_3_time_seconds else "-"}</span></div>
                    <div class="stat-row"><span class="stat-label">Rolling 3-Lap</span><span class="stat-val">{format_lap_time(participant.rolling_3_lap_avg)}</span></div>
                    <div class="stat-row"><span class="stat-label">Rolling 5-Lap</span><span class="stat-val">{format_lap_time(participant.rolling_5_lap_avg)}</span></div>
                </div>

                {ai_block_html}

                <!-- Speed Traps & Recent Laps -->
                <div class="card-block">
                    <div class="card-block-title">Speeds &amp; Recent History</div>
                    <div class="stat-row"><span class="stat-label">Speed I1</span><span class="stat-val">{speed_i1}</span></div>
                    <div class="stat-row"><span class="stat-label">Speed I2</span><span class="stat-val">{speed_i2}</span></div>
                    <div class="stat-row"><span class="stat-label">Speed FL</span><span class="stat-val">{speed_fl}</span></div>
                    <div class="stat-row"><span class="stat-label">Speed ST</span><span class="stat-val">{speed_st}</span></div>
                    <div style="margin-top: 6px; font-size: 11px; color: #a0aec0;">
                        <b>Recent History:</b> [{recent_laps_str}]
                    </div>
                </div>
            </div>
        </div>
    </div>
    """
    return F1_THEME_CSS + card_html

