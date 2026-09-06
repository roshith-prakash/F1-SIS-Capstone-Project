from __future__ import annotations

from typing import TextIO

from .models import ParticipantState, RaceState


def format_lap_time(seconds: float | None) -> str:
    """Format a lap time in seconds into m:ss.sss or ss.sss string."""
    if seconds is None:
        return "-"
    if seconds < 0:
        return f"{seconds:.3f}"
    minutes = int(seconds // 60)
    rem_seconds = seconds % 60
    if minutes > 0:
        return f"{minutes}:{rem_seconds:06.3f}"
    return f"{rem_seconds:.3f}"


def format_total_time(seconds: float | None) -> str:
    """Format total race time in seconds into h:mm:ss.sss or m:ss.sss string."""
    if seconds is None:
        return "-"
    if seconds < 0:
        return f"{seconds:.3f}"
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    rem_seconds = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{rem_seconds:06.3f}"
    if minutes > 0:
        return f"{minutes}:{rem_seconds:06.3f}"
    return f"{rem_seconds:.3f}"


def format_gap(gap_seconds: float | None, is_leader: bool = False) -> str:
    """Format gap to leader or interval ahead/behind."""
    if is_leader:
        return ""
    if gap_seconds is None:
        return "-"
    return f"{gap_seconds:.3f}"


def format_pos_delta(change: int | None) -> str:
    """Format position delta (gain/loss)."""
    if change is None or change == 0:
        return "-"
    if change > 0:
        return f"+{change}"
    return f"{change}"


def format_tyre_info(compound: str | None, tyre_life: float | None, fresh: bool | None) -> str:
    """Format tyre compound, tyre life (laps), and freshness tag."""
    comp = (compound or "UNK").upper()
    life_str = f"L{int(tyre_life)}" if tyre_life is not None else "L0"
    fresh_str = "[F]" if fresh is True else "[U]" if fresh is False else ""
    return f"{comp} ({life_str}) {fresh_str}".strip()


def format_in_out(is_pit_in: bool, is_pit_out: bool) -> str:
    """Format pit in/out flag for this lap."""
    if is_pit_in and is_pit_out:
        return "IN/OUT"
    if is_pit_in:
        return "IN"
    if is_pit_out:
        return "OUT"
    return "-"


def format_pits(pit_count: int, last_pit_lap: int | None) -> str:
    """Format pit count and last pit lap."""
    if last_pit_lap is not None:
        return f"{pit_count} (L{last_pit_lap})"
    return str(pit_count)


def format_speeds(
    speed_i1: float | None,
    speed_i2: float | None,
    speed_fl: float | None,
    speed_st: float | None,
) -> str:
    """Format speed traps: I1 / I2 / FL / ST."""
    i1 = f"{speed_i1:.0f}" if speed_i1 is not None else "-"
    i2 = f"{speed_i2:.0f}" if speed_i2 is not None else "-"
    fl = f"{speed_fl:.0f}" if speed_fl is not None else "-"
    st = f"{speed_st:.0f}" if speed_st is not None else "-"
    return f"{i1}/{i2}/{fl}/{st}"


def format_race_state_compact(state: RaceState, participant_limit: int | None = 10) -> str:
    """Return a minimal compact view of the current race snapshot."""
    lines = [
        f"Race: {state.grand_prix or 'Unknown'} ({state.year or 'Unknown'})",
        f"Lap: {state.current_lap or 'N/A'}"
        + (
            f"/{state.total_laps_expected}"
            if state.total_laps_expected is not None
            else ""
        ),
        f"Drivers tracked: {len(state.participants)}",
        "",
        "Classification:",
        "Pos  Driver  Team                         Gap (s)",
        "---  ------  ---------------------------  -------",
    ]

    entries = state.classification
    if participant_limit is not None:
        entries = entries[:participant_limit]
    for entry in entries:
        team = (entry.team or "")[:27]
        gap = "" if entry.gap_to_leader_seconds is None or entry.position == 1 else f"{entry.gap_to_leader_seconds:.3f}"
        position = "" if entry.position is None else str(entry.position)
        lines.append(f"{position:>3}  {entry.driver:<6}  {team:<27}  {gap:>7}")

    if not entries:
        lines.append("(no classified drivers)")

    conditions = state.current_conditions
    lines.extend(
        [
            "",
            "Conditions:",
            f"Track status: {conditions.track_status or 'N/A'}",
            f"Air/track temperature: {conditions.air_temp or 'N/A'} / {conditions.track_temp or 'N/A'}",
            f"Rainfall: {conditions.rainfall if conditions.rainfall is not None else 'N/A'}",
            f"Safety car: {conditions.has_safety_car if conditions.has_safety_car is not None else 'N/A'}",
        ]
    )
    return "\n".join(lines)


def format_race_state(
    state: RaceState,
    participant_limit: int | None = 10,
    compact: bool = False,
) -> str:
    """Return a race snapshot formatted with two dedicated CLI-friendly tables (Timing + Telemetry/Strategy)."""
    if compact:
        return format_race_state_compact(state, participant_limit)

    total_expected = (
        f"/{state.total_laps_expected}"
        if state.total_laps_expected is not None
        else ""
    )

    entries = state.classification
    if participant_limit is not None:
        entries = entries[:participant_limit]

    lines = [
        f"Race: {state.grand_prix or 'Unknown'} ({state.year or 'Unknown'})",
        f"Lap: {state.current_lap or 'N/A'}{total_expected}",
        f"Drivers tracked: {len(state.participants)}",
        "",
        "Classification & Timing:",
        (
            f"{'Pos':>3} {'Chg':>4} {'Driver':<6} {'Team':<20} "
            f"{'Last Lap':>9} {'Total Time':>11} {'S1':>6} {'S2':>6} {'S3':>6} "
            f"{'Gap (s)':>8} {'Int (s)':>8} {'Behind (s)':>10}"
        ),
        (
            f"{'---':>3} {'---':>4} {'------':<6} {'--------------------':<20} "
            f"{'---------':>9} {'-----------':>11} {'------':>6} {'------':>6} {'------':>6} "
            f"{'--------':>8} {'--------':>8} {'----------':>10}"
        ),
    ]

    for entry in entries:
        driver = entry.driver
        p = state.participants.get(driver, ParticipantState(driver=driver))
        pos_str = str(entry.position) if entry.position is not None else "-"
        chg_str = format_pos_delta(p.position_change)
        driver_str = driver
        team_str = (p.team or entry.team or "")[:20]

        last_lap_str = format_lap_time(p.last_lap_time_seconds)
        total_time_str = format_total_time(p.total_race_time_seconds)
        s1_str = f"{p.sector_1_time_seconds:.2f}" if p.sector_1_time_seconds is not None else "-"
        s2_str = f"{p.sector_2_time_seconds:.2f}" if p.sector_2_time_seconds is not None else "-"
        s3_str = f"{p.sector_3_time_seconds:.2f}" if p.sector_3_time_seconds is not None else "-"

        gap_leader_str = format_gap(entry.gap_to_leader_seconds, is_leader=(entry.position == 1))
        int_ahead_str = format_gap(entry.interval_to_position_ahead_seconds, is_leader=(entry.position == 1))
        gap_behind_str = format_gap(entry.gap_behind_seconds or p.gap_behind_seconds)

        lines.append(
            f"{pos_str:>3} {chg_str:>4} {driver_str:<6} {team_str:<20} "
            f"{last_lap_str:>9} {total_time_str:>11} {s1_str:>6} {s2_str:>6} {s3_str:>6} "
            f"{gap_leader_str:>8} {int_ahead_str:>8} {gap_behind_str:>10}"
        )

    if not entries:
        lines.append("  (No classified drivers)")

    # Table 2: Tyre, Strategy & Telemetry
    lines.extend(
        [
            "",
            "Tyre, Pit Strategy & Telemetry:",
            (
                f"{'Pos':>3} {'Driver':<6} {'Tyre (Life)':<17} {'Stint':>5} "
                f"{'Pits (Last)':>11} {'In/Out':>6} {'Roll 3L':>9} "
                f"{'Speeds (I1/I2/FL/ST)':^21}  {'Recent Laps (Last 5)':<35}"
            ),
            (
                f"{'---':>3} {'------':<6} {'-----------------':<17} {'-----':>5} "
                f"{'-----------':>11} {'------':>6} {'---------':>9} "
                f"{'---------------------':^21}  {'-----------------------------------':<35}"
            ),
        ]
    )

    for entry in entries:
        driver = entry.driver
        p = state.participants.get(driver, ParticipantState(driver=driver))
        pos_str = str(entry.position) if entry.position is not None else "-"
        driver_str = driver

        tyre_str = format_tyre_info(p.compound, p.tyre_life, p.fresh_tyre)[:17]
        stint_str = f"S{int(p.stint)}" if p.stint is not None else "S1"
        pits_str = format_pits(p.pit_count, p.last_pit_lap)
        in_out_str = format_in_out(p.is_pit_in_lap, p.is_pit_out_lap)

        roll3_str = format_lap_time(p.rolling_3_lap_avg)
        speeds_str = format_speeds(p.speed_i1, p.speed_i2, p.speed_fl, p.speed_st)

        recent_laps_formatted = [format_lap_time(t) for t in p.recent_lap_times]
        recent_str = f"[{', '.join(recent_laps_formatted)}]" if recent_laps_formatted else "[]"

        lines.append(
            f"{pos_str:>3} {driver_str:<6} {tyre_str:<17} {stint_str:>5} "
            f"{pits_str:>11} {in_out_str:>6} {roll3_str:>9} "
            f"{speeds_str:^21}  {recent_str:<35}"
        )

    if not entries:
        lines.append("  (No classified drivers)")

    conditions = state.current_conditions
    lines.extend(
        [
            "",
            "Conditions:",
            f"Track status: {conditions.track_status or 'N/A'}",
            f"Air/track temperature: {conditions.air_temp or 'N/A'} / {conditions.track_temp or 'N/A'}",
            f"Rainfall: {conditions.rainfall if conditions.rainfall is not None else 'N/A'}",
            f"Safety car: {conditions.has_safety_car if conditions.has_safety_car is not None else 'N/A'}",
        ]
    )
    return "\n".join(lines)


def format_driver_detail(state: RaceState, driver_code: str) -> str:
    """Return an in-depth single-driver telemetry and strategy inspection card."""
    code = driver_code.strip().upper()
    participant = state.participants.get(code)
    if participant is None:
        return f"Driver '{code}' not found in current race state."

    sep_double = "=" * 80
    sep_single = "-" * 80

    stint_num = int(participant.stint) if participant.stint is not None else 1
    total_race_time_str = (
        format_total_time(participant.total_race_time_seconds)
        if participant.total_race_time_seconds is not None
        else "N/A"
    )

    lines = [
        sep_double,
        f" DRIVER TELEMETRY & STRATEGY CARD: {code} (#{participant.driver_number or 'N/A'})",
        sep_double,
        f" Full Name      : {participant.driver_full_name or code}",
        f" Team           : {participant.team or 'Unknown'}",
        f" Active Status  : {'ACTIVE' if participant.is_active else 'INACTIVE / RETIRED'}",
        sep_single,
        " TRACK POSITION & GAPS",
        sep_single,
        f" Position       : P{participant.position or 'N/A'} (Delta: {format_pos_delta(participant.position_change)})",
        f" Gap to Leader  : {format_gap(participant.gap_to_leader_seconds, is_leader=(participant.position == 1))}",
        f" Interval Ahead : {format_gap(participant.interval_to_position_ahead_seconds, is_leader=(participant.position == 1))}",
        f" Gap Behind     : {format_gap(participant.gap_behind_seconds)}",
        f" Total Time     : {total_race_time_str}",
        sep_single,
        " TYRE & PIT STRATEGY",
        sep_single,
        f" Compound       : {participant.compound or 'N/A'} (Tyre Life: {int(participant.tyre_life) if participant.tyre_life is not None else 0} laps)",
        f" Fresh Tyre     : {'Yes' if participant.fresh_tyre is True else 'No (Used)' if participant.fresh_tyre is False else 'Unknown'}",
        f" Current Stint  : Stint {stint_num}",
        f" Pit Stops Done : {participant.pit_count} (Last Pit on Lap: {participant.last_pit_lap or 'N/A'})",
        f" Laps Since Pit : {participant.laps_since_last_pit if participant.laps_since_last_pit is not None else 'N/A'}",
        f" In/Out Lap     : Pit-In: {participant.is_pit_in_lap} | Pit-Out: {participant.is_pit_out_lap}",
        sep_single,
        " LAP TIMING & SECTORS",
        sep_single,
        f" Last Lap Time  : {format_lap_time(participant.last_lap_time_seconds)} (Lap {participant.last_lap_number or 'N/A'})",
        f" Sector 1       : {format_lap_time(participant.sector_1_time_seconds)}",
        f" Sector 2       : {format_lap_time(participant.sector_2_time_seconds)}",
        f" Sector 3       : {format_lap_time(participant.sector_3_time_seconds)}",
        f" Rolling 3-Lap  : {format_lap_time(participant.rolling_3_lap_avg)}",
        f" Rolling 5-Lap  : {format_lap_time(participant.rolling_5_lap_avg)}",
        f" Recent History : {[format_lap_time(t) for t in participant.recent_lap_times]}",
        sep_single,
        " SPEED TRAP VELOCITIES (KM/H)",
        sep_single,
        f" Speed I1       : {format_speeds(participant.speed_i1, None, None, None).split('/')[0]} km/h",
        f" Speed I2       : {format_speeds(None, participant.speed_i2, None, None).split('/')[1]} km/h",
        f" Speed FL       : {format_speeds(None, None, participant.speed_fl, None).split('/')[2]} km/h",
        f" Speed ST       : {format_speeds(None, None, None, participant.speed_st).split('/')[3]} km/h",
        sep_double,
    ]
    return "\n".join(lines)


def print_race_state(
    state: RaceState,
    participant_limit: int | None = 10,
    compact: bool = False,
    driver: str | None = None,
    output: TextIO | None = None,
) -> None:
    """Print the formatted race snapshot or single-driver card to output stream."""
    import sys

    out = output or sys.stdout
    if driver:
        print(format_driver_detail(state, driver), file=out)
    else:
        print(format_race_state(state, participant_limit=participant_limit, compact=compact), file=out)
