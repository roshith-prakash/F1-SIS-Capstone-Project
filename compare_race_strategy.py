"""
compare_race_strategy.py
========================
Counterfactual Race Strategy Evaluation & Historical Backtesting Engine.

Compares actual historical race execution (from FastF1 / telemetry CSV) against
the F1-SIS Strategy & Decision Engine recommendations lap-by-lap.

Usage:
  python compare_race_strategy.py --driver VER --race italian
  python compare_race_strategy.py --driver NOR --race british --profile aggressive
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.models import CurrentConditions, ParticipantState, RaceState, normalize_driver
from strategy_engine.engine import StrategyEngine
from strategy_engine.types import StrategyEngineConfig
from decision_engine.baseline import BaselineDecisionPolicy
from decision_engine.rl.dqn import RLDecisionPolicy
from decision_engine.types import ImmediateAction


# Nominal degradation rates per lap (seconds/lap) and compound pace offsets relative to Medium
COMPOUND_PACE_OFFSETS = {
    "SOFT": -0.65,
    "MEDIUM": 0.0,
    "HARD": +0.55,
    "INTERMEDIATE": 3.0,
    "WET": 6.0,
}

COMPOUND_DEG_RATES = {
    "SOFT": 0.085,
    "MEDIUM": 0.045,
    "HARD": 0.025,
    "INTERMEDIATE": 0.040,
    "WET": 0.035,
}

NOMINAL_STINT_LIFESPANS = {
    "SOFT": 24,
    "MEDIUM": 34,
    "HARD": 46,
    "INTERMEDIATE": 30,
    "WET": 30,
}


def compute_tyre_pace_delta(
    ai_compound: str,
    ai_tyre_age: float,
    act_compound: str,
    act_tyre_age: float,
) -> float:
    """
    Computes lap pace delta (seconds) between AI tyre state and actual historical tyre state.
    Negative value means AI is faster; positive means AI is slower.
    """
    # 1. Base compound speed difference
    comp_delta = COMPOUND_PACE_OFFSETS.get(ai_compound, 0.0) - COMPOUND_PACE_OFFSETS.get(act_compound, 0.0)

    # 2. Linear wear component
    ai_deg = ai_tyre_age * COMPOUND_DEG_RATES.get(ai_compound, 0.045)
    act_deg = act_tyre_age * COMPOUND_DEG_RATES.get(act_compound, 0.045)

    # 3. Non-linear cliff penalty (calibrated to StrategyEngine: 0.008, NOT uncalibrated 0.25)
    ai_nom = NOMINAL_STINT_LIFESPANS.get(ai_compound, 34)
    if ai_tyre_age > ai_nom:
        ai_deg += 0.008 * ((ai_tyre_age - ai_nom) ** 1.3)

    act_nom = NOMINAL_STINT_LIFESPANS.get(act_compound, 34)
    if act_tyre_age > act_nom:
        act_deg += 0.008 * ((act_tyre_age - act_nom) ** 1.3)

    delta = comp_delta + (ai_deg - act_deg)
    # Physically clamp tyre pace delta to [-2.2s, +2.2s] (maximum possible F1 dry tyre pace delta)
    return float(np.clip(delta, -2.2, 2.2))


def load_race_snapshots_or_csv(race_keyword: str, year: int = 2025, csv_path: Path | None = None) -> tuple[list[dict[str, Any]], pd.DataFrame, str]:
    """Loads matching race CSV from the specified season year (defaults to 2025)."""
    data_dir = ROOT / "data_fastf1_v1"
    laps_dir = data_dir / "laps" / str(year)
    if not laps_dir.exists():
        laps_dir = data_dir / "laps" / "2024"
        year = 2024

    if csv_path is not None and Path(csv_path).exists():
        matched_csv = Path(csv_path)
    else:
        rk_clean = race_keyword.lower().replace(" ", "_")
        matched_csv = None
        for f in laps_dir.glob("*.csv"):
            f_clean = f.stem.lower()
            if rk_clean in f_clean or race_keyword.lower() in f_clean.replace("_", " "):
                matched_csv = f
                break

        if matched_csv is None:
            # Fallback to Italian GP (Monza)
            matched_csv = laps_dir / "Italian_Grand_Prix.csv"

    clean_name = matched_csv.stem.replace("_", " ")

    # Precomputed snapshots available for 2024 Italian and British GP
    snapshots = []
    if year == 2024:
        if "ital" in clean_name.lower() or "monza" in clean_name.lower():
            json_path = data_dir / "race_state_snapshots_italian_gp_2024_all_models.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    snapshots = json.load(f)
        elif "brit" in clean_name.lower() or "silverstone" in clean_name.lower():
            json_path = data_dir / "race_state_snapshots_british_gp_2024_all_models.json"
            if json_path.exists():
                with open(json_path, "r", encoding="utf-8") as f:
                    snapshots = json.load(f)

    df_laps = pd.read_csv(matched_csv)
    has_rain = bool(df_laps["Rainfall"].any()) if "Rainfall" in df_laps.columns else False
    weather_desc = "Mixed/Rain" if has_rain else "100% Dry"
    gp_name = f"{year} {clean_name} ({weather_desc})"

    return snapshots, df_laps, gp_name



def build_race_state_from_snapshot(s: dict[str, Any]) -> RaceState:
    """Deserializes snapshot dictionary into typed RaceState object."""
    cond = CurrentConditions(
        **{k: v for k, v in s.get("current_conditions", {}).items() if hasattr(CurrentConditions, k)}
    )
    parts = {}
    for d, p in s.get("participants", {}).items():
        clean_p = {k: v for k, v in p.items() if hasattr(ParticipantState, k)}
        parts[d] = ParticipantState(**clean_p)

    return RaceState(
        race_id=s.get("race_id"),
        year=s.get("year", 2024),
        grand_prix=s.get("grand_prix"),
        round=s.get("round"),
        country=s.get("country"),
        location=s.get("location"),
        current_lap=s.get("current_lap"),
        total_laps_expected=s.get("total_laps_expected", 53),
        current_conditions=cond,
        participants=parts,
    )


def build_race_state_from_csv(
    df_laps: pd.DataFrame, lap_num: int, total_laps: int, gp_name: str
) -> RaceState:
    """Constructs a typed RaceState directly from race CSV rows on lap_num."""
    lap_rows = df_laps[df_laps["LapNumber"] == lap_num]
    parts = {}
    for _, row in lap_rows.iterrows():
        d = str(row["Driver"])
        parts[d] = ParticipantState(
            driver=d,
            team=str(row.get("Team", "Racing Team")),
            position=int(row["Position"]) if pd.notna(row.get("Position")) else 10,
            compound=str(row.get("Compound", "MEDIUM")).upper(),
            tyre_life=float(row.get("TyreLife", 1.0)) if pd.notna(row.get("TyreLife")) else 1.0,
            pit_count=max(0, int(row.get("Stint", 1)) - 1) if pd.notna(row.get("Stint")) else 0,
            total_race_time_seconds=float(row.get("CumLapTime", 0.0)) if pd.notna(row.get("CumLapTime")) else 0.0,
            last_lap_time_seconds=float(row.get("CleanLapTime", 85.0)) if pd.notna(row.get("CleanLapTime")) else 85.0,
            gap_to_leader_seconds=float(row.get("GapToLeaderSeconds", 0.0)) if pd.notna(row.get("GapToLeaderSeconds")) else 0.0,
            interval_to_position_ahead_seconds=float(row.get("IntervalToPositionAheadSeconds", 0.0)) if pd.notna(row.get("IntervalToPositionAheadSeconds")) else 0.0,
            is_active=True,
        )

    row0 = lap_rows.iloc[0] if not lap_rows.empty else {}
    cond = CurrentConditions(
        air_temp=float(row0.get("AirTemp", 28.0)) if pd.notna(row0.get("AirTemp")) else 28.0,
        track_temp=float(row0.get("TrackTemp", 42.0)) if pd.notna(row0.get("TrackTemp")) else 42.0,
        has_safety_car=bool(row0.get("HasSafetyCar", False)),
        has_vsc=bool(row0.get("HasVSC", False)),
        rainfall=False,
    )

    return RaceState(
        race_id=f"2024_{gp_name.replace(' ', '_')}",
        current_lap=lap_num,
        total_laps_expected=total_laps,
        grand_prix=gp_name,
        current_conditions=cond,
        participants=parts,
    )



def simulate_and_compare(
    driver: str = "VER",
    race: str = "italian",
    year: int = 2025,
    csv_path: Path | None = None,
    profile: str = "balanced",
    horizon_laps: int = 8,
    n_rollouts: int = 15,
    policy: str = "rl",
    model_path: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    driver = normalize_driver(driver) or "VER"
    snapshots, df_laps, gp_name = load_race_snapshots_or_csv(race, year=year, csv_path=csv_path)

    # Configure Strategy Engine and Decision Policy
    cfg = StrategyEngineConfig(
        default_horizon_laps=horizon_laps,
        default_n_rollouts=n_rollouts,
        default_random_seed=42,
    )
    engine = StrategyEngine(config=cfg)

    ckpt_path = Path(model_path) if model_path else (ROOT / "models" / "Decision Engine" / "decision_engine_dqn.pt")
    if policy.lower() == "rl" and ckpt_path.exists():
        try:
            decision_policy = RLDecisionPolicy.from_checkpoint(ckpt_path)
            policy_desc = f"RL {decision_policy.architecture.upper()} DQN ({ckpt_path.name})"
        except Exception as e:
            decision_policy = BaselineDecisionPolicy(risk_profile=profile)
            policy_desc = f"Baseline Heuristic ({profile.upper()}, fallback: {e})"
    else:
        decision_policy = BaselineDecisionPolicy(risk_profile=profile)
        policy_desc = f"Baseline Heuristic ({profile.upper()})"

    if verbose:
        print(f"\n=========================================================================================")
        print(f"  F1-SIS COUNTERFACTUAL RACE STRATEGY EVALUATION")
        print(f"  Grand Prix : {gp_name}")
        print(f"  Target Ego : {driver} | Policy: {policy_desc}")
        print(f"=========================================================================================\n")

    # Filter actual laps for ego driver
    ego_laps = df_laps[df_laps["Driver"] == driver].sort_values("LapNumber").copy()
    if ego_laps.empty:
        raise ValueError(f"Driver '{driver}' not found in race data.")

    total_laps = int(ego_laps["LapNumber"].max())

    # Pre-calculate clean cumulative lap time from Lap 1 for all drivers
    df_laps["CleanLapTime"] = pd.to_numeric(df_laps["LapTimeSeconds"], errors="coerce").fillna(85.0)
    df_laps["CumLapTime"] = df_laps.groupby("Driver")["CleanLapTime"].cumsum()

    competitor_times_by_lap: dict[int, list[float]] = {}
    for lap_num in range(1, total_laps + 1):
        lap_rows = df_laps[(df_laps["LapNumber"] == lap_num) & (df_laps["Driver"] != driver)]
        c_times = lap_rows["CumLapTime"].dropna().tolist()
        competitor_times_by_lap[lap_num] = sorted(c_times)

    # Initialize simulation state
    first_row = ego_laps.iloc[0]
    ai_compound = str(first_row.get("Compound", "MEDIUM")).upper()
    ai_tyre_age = 0.0
    ai_pit_count = 0
    ai_cum_time = 0.0
    ai_pos = int(first_row["Position"]) if pd.notna(first_row.get("Position")) else 10

    act_cum_time = 0.0
    act_pit_count = 0

    lap_records = []
    strategic_divergences = []

    # Typical pit loss in seconds (Silverstone ~21s, Monza ~24s)
    pit_loss_sec = 24.0 if "monza" in gp_name.lower() or "ital" in gp_name.lower() else 21.0

    if verbose:
        print("Simulating race lap-by-lap with Strategy & Decision Engine...\n")

    if "CleanLapTime" in df_laps.columns and df_laps["CleanLapTime"].dropna().count() > 0:
        fastest_clean = float(df_laps["CleanLapTime"].dropna().min())
    else:
        fastest_clean = float(ego_laps["LapTimeSeconds"].dropna().min())
    last_clean_pace = fastest_clean

    for lap_idx, (_, row) in enumerate(ego_laps.iterrows(), start=1):
        lap_num = int(row["LapNumber"])

        # 1. Actual Driver Telemetry
        act_lap_time = float(row["LapTimeSeconds"]) if pd.notna(row["LapTimeSeconds"]) else 86.0
        act_compound = str(row["Compound"]).upper() if pd.notna(row["Compound"]) else "HARD"
        act_tyre_age = float(row["TyreLife"]) if pd.notna(row["TyreLife"]) else float(lap_idx)
        act_pos = int(row["Position"]) if pd.notna(row["Position"]) else 10

        # Out-lap identification
        raw_out_lap = pd.notna(row.get("PitOutTime"))
        raw_pitting = pd.notna(row.get("PitInTime"))

        # Caution identification
        track_status = str(row.get("TrackStatus", ""))
        is_caution = bool(row.get("HasSafetyCar", False) or row.get("HasVSC", False)) or ("4" in track_status) or ("6" in track_status) or ((act_lap_time > 1.35 * fastest_clean) and not raw_pitting and not raw_out_lap)
        curr_pit_loss = 14.0 if is_caution else pit_loss_sec

        # Distinguish genuine pit stop from Safety Car pit lane queue (where tyres were not changed)
        next_rows = ego_laps[ego_laps["LapNumber"] == lap_num + 1]
        next_life = float(next_rows["TyreLife"].values[0]) if not next_rows.empty and pd.notna(next_rows["TyreLife"].values[0]) else 1.0
        act_is_pitting = raw_pitting and not (is_caution and next_life > 5)
        is_act_out_lap = raw_out_lap and not (is_caution and act_tyre_age > 5)

        # Track clean flying pace when car is at racing speed under green flag
        if not is_caution and not raw_pitting and not is_act_out_lap and act_lap_time < 1.3 * fastest_clean:
            last_clean_pace = act_lap_time

        if act_is_pitting:
            act_pit_count += 1

        act_cum_time += act_lap_time

        # 2. AI Decision Engine Query
        if lap_num >= total_laps:
            ai_action = ImmediateAction.STAY_OUT
            target_compound = ai_compound
        else:
            # Build RaceState for current lap
            if lap_idx - 1 < len(snapshots) and snapshots:
                rs = build_race_state_from_snapshot(snapshots[lap_idx - 1])
            else:
                rs = build_race_state_from_csv(df_laps, lap_num, total_laps, gp_name)

            # Update ego participant in state to reflect AI's counterfactual state
            if driver in rs.participants:
                rs.participants[driver].position = ai_pos
                rs.participants[driver].compound = ai_compound
                rs.participants[driver].tyre_life = ai_tyre_age
                rs.participants[driver].pit_count = ai_pit_count

            try:
                se_result = engine.evaluate_race_state(rs, ego_driver=driver)
                decision = decision_policy.select_action(se_result, race_state=rs)
                ai_action = decision.immediate_action
                target_compound = decision.target_compound or "HARD"
            except Exception:
                ai_action = ImmediateAction.STAY_OUT
                target_compound = ai_compound

        # 3. Simulate Counterfactual AI Lap Time
        notes = []

        min_stint = 9.0 if is_caution else 12.0
        laps_remaining = total_laps - lap_num
        nominal_life = NOMINAL_STINT_LIFESPANS.get(ai_compound, 34)
        tyre_critical = ai_tyre_age >= nominal_life

        # Green flag discretionary pit stops not permitted in the final 8 laps
        can_pit_window = (laps_remaining >= 8) or is_caution

        # Maximum stops budget:
        # If already pitted, can only pit again if caution opens a cheap window, or tyre reaches structural cliff
        can_pit_budget = (ai_pit_count < 2) or is_caution or tyre_critical

        # Payback Filter: For discretionary 2nd+ stops under green flag,
        # verify that remaining laps can pay back the ~21-24s pit transit loss
        if ai_pit_count >= 1 and not is_caution and not tyre_critical:
            min_payback_laps = int(curr_pit_loss / 0.8)  # e.g. 21s / 0.8s = 26 laps
            can_payback = laps_remaining >= min_payback_laps
        else:
            can_payback = True

        is_ai_pitting = (
            (ai_action != ImmediateAction.STAY_OUT)
            and (ai_tyre_age >= min_stint)
            and can_pit_window
            and can_pit_budget
            and can_payback
        )

        if is_ai_pitting:
            ai_pit_count += 1
            old_comp = ai_compound
            ai_compound = target_compound
            ai_tyre_age = 0.0

            # Base clean flying pace + pit loss
            ai_lap_time = last_clean_pace + curr_pit_loss
            caution_tag = " (CAUTION PIT)" if is_caution else ""
            notes.append(f"AI BOX -> {ai_compound}{caution_tag}")

            if not act_is_pitting:
                notes.append("[AI UNDERCUT TRIGGERED]")
                strategic_divergences.append(
                    f"Lap {lap_num}: AI pitted for fresh {ai_compound} while actual stayed out on {act_compound} (age {act_tyre_age:.0f})."
                )
        else:
            ai_tyre_age += 1.0
            tyre_delta = compute_tyre_pace_delta(ai_compound, ai_tyre_age, act_compound, act_tyre_age)

            if act_is_pitting:
                # Actual driver entered pit lane (in-lap)
                # AI stays out on flying pace
                ai_lap_time = last_clean_pace + tyre_delta
                notes.append(f"AI STAYS OUT (Actual In-Lap)")
                strategic_divergences.append(
                    f"Lap {lap_num}: Actual pitted for {row.get('Compound', 'tyres')} while AI stayed out on {ai_compound} (age {ai_tyre_age:.0f})."
                )
            elif is_act_out_lap:
                # Actual driver exiting pit lane (out-lap)
                # AI stays out on flying pace
                ai_lap_time = last_clean_pace + tyre_delta
                notes.append("AI STAYS OUT (Actual Out-Lap)")
            else:
                # Both on track
                ai_lap_time = act_lap_time + tyre_delta

        # Physical sanity bounds:
        # 1. AI cannot drive faster than physical race lap record
        ai_lap_time = max(ai_lap_time, fastest_clean - 0.3)
        # 2. Under Safety Car / Caution, car cannot exceed Safety Car pacing delta
        if is_caution and not is_ai_pitting:
            ai_lap_time = max(ai_lap_time, act_lap_time - 1.5)

        ai_cum_time += ai_lap_time

        # 4. Compute Authentic Counterfactual Position
        # Compare ai_cum_time against all competitors on this lap
        rival_times = competitor_times_by_lap.get(lap_num, [])
        ai_pos = 1 + sum(1 for t in rival_times if t < ai_cum_time)

        # Gap calculation: (Actual cumulative time - AI cumulative time)
        # Positive gap = AI is ahead / faster than actual!
        cum_gap = act_cum_time - ai_cum_time
        pos_diff = act_pos - ai_pos  # Positive = AI has better position (e.g. P4 vs P6 = +2)

        lap_records.append({
            "lap": lap_num,
            "act_compound": act_compound,
            "act_tyre_age": int(act_tyre_age),
            "act_action": "PIT" if act_is_pitting else "STAY",
            "act_lap_time": act_lap_time,
            "act_pos": act_pos,
            "ai_compound": ai_compound,
            "ai_tyre_age": int(ai_tyre_age),
            "ai_action": "PIT" if is_ai_pitting else "STAY",
            "ai_lap_time": ai_lap_time,
            "ai_pos": ai_pos,
            "cum_gap_s": cum_gap,
            "pos_diff": pos_diff,
            "notes": ", ".join(notes),
        })

    # Final Grand Summary
    final_act_pos = lap_records[-1]["act_pos"]
    final_ai_pos = lap_records[-1]["ai_pos"]
    final_gap = lap_records[-1]["cum_gap_s"]

    if verbose:
        # Print comparative table
        print("=" * 115)
        print(
            f"{'Lap':<4} | {'Actual Tyre':<11} | {'Act Act':<7} | {'AI Tyre':<11} | {'AI Act':<7} | "
            f"{'Act Time':<8} | {'AI Time':<8} | {'Gap (s)':<8} | {'Act P':<5} | {'AI P':<5} | {'Key Event / Note'}"
        )
        print("-" * 115)

        # Show select sample laps and key divergence laps
        for rec in lap_records:
            gap_str = f"{rec['cum_gap_s']:+6.2f}s"
            note_str = rec["notes"]
            act_tyre = f"{rec['act_compound'][:4]} (L{rec['act_tyre_age']})"
            ai_tyre = f"{rec['ai_compound'][:4]} (L{rec['ai_tyre_age']})"

            # Highlight pit laps or strategic shifts
            is_highlight = rec["act_action"] == "PIT" or rec["ai_action"] == "PIT" or rec["lap"] in [1, 15, 20, 25, 30, 35, 40, 45, total_laps]

            if is_highlight:
                print(
                    f"L{rec['lap']:<3} | {act_tyre:<11} | {rec['act_action']:<7} | {ai_tyre:<11} | {rec['ai_action']:<7} | "
                    f"{rec['act_lap_time']:>6.2f}s  | {rec['ai_lap_time']:>6.2f}s  | {gap_str:<8} | "
                    f"P{rec['act_pos']:<4} | P{rec['ai_pos']:<4} | {note_str}"
                )

        print("=" * 115)

        print("\n-----------------------------------------------------------------------------------------")
        print("  STRATEGIC COMPARISON & FINAL VERDICT")
        print("-----------------------------------------------------------------------------------------")
        print(f"Total Race Laps          : {total_laps}")
        print(f"Actual Race Finish       : P{final_act_pos} (Total Time: {act_cum_time:.2f}s | Stops: {act_pit_count})")
        print(f"AI System Counterfactual : P{final_ai_pos} (Total Time: {ai_cum_time:.2f}s | Stops: {ai_pit_count})")

        time_verdict = f"{abs(final_gap):.2f}s FASTER" if final_gap > 0 else f"{abs(final_gap):.2f}s SLOWER"
        pos_verdict = (
            f"{final_act_pos - final_ai_pos} POSITIONS GAINED (P{final_act_pos} -> P{final_ai_pos})"
            if final_ai_pos < final_act_pos
            else "EQUIVALENT POSITION"
            if final_ai_pos == final_act_pos
            else f"{final_ai_pos - final_act_pos} POSITIONS CONCEDED"
        )

        print(f"Net Advantage            : {time_verdict} ({final_gap:+.2f}s total delta)")
        print(f"Position Outcome         : {pos_verdict}")
        print("\nKey Strategic Divergences:")
        for sd in strategic_divergences[:5]:
            print(f"  * {sd}")
        print("-----------------------------------------------------------------------------------------\n")

    return {
        "gp_name": gp_name,
        "driver": driver,
        "total_laps": total_laps,
        "final_act_pos": final_act_pos,
        "final_ai_pos": final_ai_pos,
        "pos_diff": final_act_pos - final_ai_pos,
        "final_gap": final_gap,
        "act_cum_time": act_cum_time,
        "ai_cum_time": ai_cum_time,
        "act_pit_count": act_pit_count,
        "ai_pit_count": ai_pit_count,
        "strategic_divergences": strategic_divergences,
        "records": lap_records,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Counterfactual F1 Race Strategy Comparison")
    parser.add_argument("--driver", default="VER", help="Driver code (e.g., VER, NOR, LEC)")
    parser.add_argument("--race", default="italian", help="Race name or keyword (e.g., italian, bahrain, austrian)")
    parser.add_argument("--year", type=int, default=2025, help="Championship season year (default: 2025)")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "aggressive", "conservative"])
    parser.add_argument("--policy", default="rl", choices=["rl", "baseline"], help="Decision Engine policy (default: rl)")
    parser.add_argument("--model-path", default=None, help="Path to RL checkpoint (.pt)")
    parser.add_argument("--horizon", type=int, default=8, help="Strategy Engine horizon laps")
    parser.add_argument("--rollouts", type=int, default=15, help="Rollouts per candidate")
    args = parser.parse_args()

    simulate_and_compare(
        driver=args.driver,
        race=args.race,
        year=args.year,
        profile=args.profile,
        horizon_laps=args.horizon,
        n_rollouts=args.rollouts,
        policy=args.policy,
        model_path=args.model_path,
    )

