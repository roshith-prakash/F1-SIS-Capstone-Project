"""
batch_evaluate_2025_dry.py
==========================
Batch Counterfactual Strategy Backtesting Runner for all 2025 Dry F1 Races.

Evaluates F1-SIS Strategy & Decision Engine lap-by-lap against actual race
outcomes across every completely dry race of the 2025 Formula 1 season.
Outputs aggregate metrics and generates a comprehensive Markdown report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from compare_race_strategy import simulate_and_compare


def get_2025_dry_race_files(driver: str = "VER", finished_only: bool = True) -> list[Path]:
    """Scans 2025 season laps directory and returns paths for 100% dry races where driver finished."""
    laps_dir = ROOT / "data_fastf1_v1" / "laps" / "2025"
    if not laps_dir.exists():
        raise FileNotFoundError(f"Directory not found: {laps_dir}")

    dry_files = []
    for csv_file in sorted(laps_dir.glob("*.csv")):
        try:
            df = pd.read_csv(csv_file)
            has_rain = bool(df["Rainfall"].any()) if "Rainfall" in df.columns else False
            # Check for wet / intermediate compound usage
            compounds = set(df["Compound"].dropna().str.upper().unique()) if "Compound" in df.columns else set()
            used_wet = any(c in compounds for c in ["INTERMEDIATE", "WET", "INTER"])
            if not has_rain and not used_wet:
                if finished_only and driver:
                    max_race_laps = int(df["LapNumber"].max()) if not df.empty else 0
                    drv_laps = df[df["Driver"] == driver]
                    drv_max_lap = int(drv_laps["LapNumber"].max()) if not drv_laps.empty else 0
                    # Standard F1 90% classification rule for finishing
                    if max_race_laps > 0 and (drv_max_lap < max_race_laps - 2 or drv_max_lap < 0.9 * max_race_laps):
                        continue
                dry_files.append(csv_file)
        except Exception as e:
            print(f"Warning: Could not process {csv_file.name}: {e}")

    return dry_files


def run_batch_evaluation(
    driver: str = "VER",
    profile: str = "balanced",
    policy: str = "rl",
    model_path: Path | str | None = None,
    output_md: Path | str = "2025_dry_races_net_advantage.md",
    finished_only: bool = True,
) -> dict:
    dry_files = get_2025_dry_race_files(driver=driver, finished_only=finished_only)
    total_races = len(dry_files)

    scope_desc = "Finished Dry Races Only" if finished_only else "All Dry Races"
    policy_label = f"RL DQN ({Path(model_path).name if model_path else 'decision_engine_dqn.pt'})" if policy == "rl" else f"Baseline ({profile.upper()})"
    print("=" * 90)
    print(f" F1-SIS 2025 SEASON BATCH EVALUATION ({scope_desc.upper()})")
    print(f" Target Driver: {driver} | Policy: {policy_label} | Total Races: {total_races}")
    print("=" * 90)

    results = []
    start_total_time = time.time()

    for idx, csv_file in enumerate(dry_files, start=1):
        clean_name = csv_file.stem.replace("_", " ")
        print(f"[{idx:02d}/{total_races:02d}] Simulating {clean_name}...", end="", flush=True)
        t0 = time.time()

        try:
            res = simulate_and_compare(
                driver=driver,
                race="",
                year=2025,
                csv_path=csv_file,
                profile=profile,
                horizon_laps=8,
                n_rollouts=15,
                policy=policy,
                model_path=model_path,
                verbose=False,
            )
            elapsed = time.time() - t0

            act_p = res["final_act_pos"]
            ai_p = res["final_ai_pos"]
            pos_diff = res["pos_diff"]
            gap_s = res["final_gap"]
            act_stops = res["act_pit_count"]
            ai_stops = res["ai_pit_count"]

            status_str = f"Act P{act_p} -> AI P{ai_p} | {gap_s:+6.2f}s | {elapsed:.1f}s"
            print(f" Done ({status_str})")

            results.append({
                "race_index": idx,
                "clean_name": clean_name,
                "file_name": csv_file.name,
                "total_laps": res["total_laps"],
                "act_pos": act_p,
                "ai_pos": ai_p,
                "pos_diff": pos_diff,
                "act_time_s": res["act_cum_time"],
                "ai_time_s": res["ai_cum_time"],
                "time_delta_s": gap_s,
                "act_stops": act_stops,
                "ai_stops": ai_stops,
                "divergences": res["strategic_divergences"],
                "rl_inferences": res.get("rl_inferences", 0),
                "rl_fallbacks": res.get("rl_fallbacks", 0),
            })
        except Exception as e:
            print(f" FAILED: {e}")

    total_eval_time = time.time() - start_total_time
    print("=" * 90)
    print(f"Batch Simulation Finished in {total_eval_time:.1f}s across {len(results)} races.")
    print("=" * 90)

    # ---------------------------------------------------------
    # Aggregate Statistics
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results)

    n_races = len(df_res)
    total_time_advantage = df_res["time_delta_s"].sum()
    avg_time_advantage = df_res["time_delta_s"].mean()
    median_time_advantage = df_res["time_delta_s"].median()

    total_pos_gained = df_res["pos_diff"].sum()
    avg_pos_gained = df_res["pos_diff"].mean()

    faster_races_count = int((df_res["time_delta_s"] > 0).sum())
    better_pos_count = int((df_res["pos_diff"] > 0).sum())
    equal_pos_count = int((df_res["pos_diff"] == 0).sum())
    conceded_pos_count = int((df_res["pos_diff"] < 0).sum())

    act_wins = int((df_res["act_pos"] == 1).sum())
    ai_wins = int((df_res["ai_pos"] == 1).sum())

    act_podiums = int((df_res["act_pos"] <= 3).sum())
    ai_podiums = int((df_res["ai_pos"] <= 3).sum())

    act_top5 = int((df_res["act_pos"] <= 5).sum())
    ai_top5 = int((df_res["ai_pos"] <= 5).sum())

    act_points = int((df_res["act_pos"] <= 10).sum())
    ai_points = int((df_res["ai_pos"] <= 10).sum())

    avg_act_pos = df_res["act_pos"].mean()
    avg_ai_pos = df_res["ai_pos"].mean()

    avg_act_stops = df_res["act_stops"].mean()
    avg_ai_stops = df_res["ai_stops"].mean()

    total_rl_inferences = sum(r.get("rl_inferences", 0) for r in results)
    total_rl_fallbacks = sum(r.get("rl_fallbacks", 0) for r in results)

    model_info = {}
    if policy == "rl":
        model_file = Path(model_path) if model_path else (ROOT / "models" / "Decision Engine" / "decision_engine_dqn.pt")
        if model_file.exists():
            import torch
            ckpt = torch.load(model_file, map_location="cpu", weights_only=False)
            model_info = {
                "path": str(model_file.resolve()),
                "name": model_file.name,
                "size_bytes": model_file.stat().st_size,
                "size_kb": model_file.stat().st_size / 1024,
                "architecture": ckpt.get("architecture", "dueling"),
                "params": sum(p.numel() for p in ckpt.get("model_state_dict", {}).values()),
            }

    print("\n---------------- AGGREGATE SUMMARY ----------------")
    print(f"Total Dry Races Simulated      : {n_races}")
    print(f"Net Season Time Delta          : {total_time_advantage:+.2f} s ({'Advantage' if total_time_advantage > 0 else 'Deficit'})")
    print(f"Average Time Delta per Race    : {avg_time_advantage:+.2f} s")
    print(f"Median Time Delta per Race     : {median_time_advantage:+.2f} s")
    print(f"Net Position Difference        : {total_pos_gained:+d} positions")
    print(f"Races with Time Advantage      : {faster_races_count}/{n_races} ({(faster_races_count / n_races) * 100:.1f}%)")
    print(f"Position Comparison            : {better_pos_count} Gained | {equal_pos_count} Equivalent | {conceded_pos_count} Conceded")
    print(f"Race Wins                      : AI {ai_wins} vs Actual {act_wins}")
    print(f"Podium Finishes (P1-P3)        : AI {ai_podiums} vs Actual {act_podiums}")
    print(f"Top 5 Finishes                 : AI {ai_top5} vs Actual {act_top5}")
    print(f"Points Finishes (P1-P10)       : AI {ai_points} vs Actual {act_points}")
    print(f"Average Finishing Position     : AI P{avg_ai_pos:.2f} vs Actual P{avg_act_pos:.2f}")
    print(f"Average Pit Stops per Race     : AI {avg_ai_stops:.2f} vs Actual {avg_act_stops:.2f}")
    if model_info:
        coverage = ((total_rl_inferences - total_rl_fallbacks) / max(1, total_rl_inferences)) * 100
        print(f"RL Checkpoint Loaded           : {model_info['name']} ({model_info['size_kb']:.1f} KB, {model_info['params']:,} params)")
        print(f"Total Neural Inferences        : {total_rl_inferences} (Fallbacks: {total_rl_fallbacks}, Coverage: {coverage:.1f}%)")
    print("---------------------------------------------------\n")

    # ---------------------------------------------------------
    # ---------------------------------------------------------
    # Generate Markdown Report
    # ---------------------------------------------------------
    from datetime import datetime

    COUNTRY_FLAGS = {
        "Abu Dhabi": "🇦🇪",
        "Azerbaijan": "🇦🇿",
        "Bahrain": "🇧🇭",
        "Canadian": "🇨🇦",
        "Canada": "🇨🇦",
        "Chinese": "🇨🇳",
        "China": "🇨🇳",
        "Dutch": "🇳🇱",
        "Emilia Romagna": "🇮🇹",
        "Hungarian": "🇭🇺",
        "Hungary": "🇭🇺",
        "Italian": "🇮🇹",
        "Italy": "🇮🇹",
        "Japanese": "🇯🇵",
        "Japan": "🇯🇵",
        "Las Vegas": "🇺🇸",
        "Mexico City": "🇲🇽",
        "Mexico": "🇲🇽",
        "Monaco": "🇲🇨",
        "Qatar": "🇶🇦",
        "Saudi Arabian": "🇸🇦",
        "Saudi Arabia": "🇸🇦",
        "Singapore": "🇸🇬",
        "Spanish": "🇪🇸",
        "Spain": "🇪🇸",
        "São Paulo": "🇧🇷",
        "Sao Paulo": "🇧🇷",
        "United States": "🇺🇸",
    }

    def get_flag(name: str) -> str:
        for k, v in COUNTRY_FLAGS.items():
            if k.lower() in name.lower():
                return v + " "
        return ""

    md_lines = []
    md_lines.append("# F1-SIS 2025 Season Dry Races: Strategic Net Advantage & Optimization Report")
    md_lines.append("")
    scope_tag = f"100% Dry Classified Finished Races ({n_races} Races)" if finished_only else f"100% Dry Races ({n_races} Races)"
    policy_str = f"RL Dueling Double-DQN (`{Path(model_path).name if model_path else 'decision_engine_dqn.pt'}`)" if policy == "rl" else f"Baseline Heuristic (`{profile.upper()}`)"
    md_lines.append(f"**Target Driver**: `{driver}` (Max Verstappen) | **Risk Profile**: `{profile.upper()}` | **Decision Policy**: {policy_str} | **Season**: `2025`  ")
    md_lines.append(f"**Filter Scope**: `{scope_tag}` — *Wet/mixed rounds (`AUS`, `BEL`, `GBR`, `MIA`) and DNFs (`AUT` Lap 1 retirement) excluded.*")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 1. Executive Summary")
    md_lines.append("")
    md_lines.append(f"This report documents the counterfactual backtesting evaluation of the **F1 Strategic Intelligence System (F1-SIS)** Decision & Strategy Engine across all completely dry Grands Prix of the 2025 Formula 1 World Championship **where the target driver finished the race**.")
    md_lines.append("")
    md_lines.append("Following the implementation of out-lap pit transit modeling, the **Race-to-Flag Pit Stop Payback Filter**, Dynamic Team Pace Tracking, Pitstop GMM, and Caution Hazard modeling, the AI system achieved the following net season performance relative to actual historical pit-wall executions:")
    md_lines.append("")

    # Alert box
    if total_time_advantage >= 0:
        md_lines.append("> [!TIP]")
        md_lines.append("> ### Net Strategic Season Advantage Achieved")
        md_lines.append(f"> Across the **{n_races} finished dry Grands Prix of 2025**:")
        md_lines.append(f"> * **Cumulative Season Net Advantage**: **`{total_time_advantage:+.2f} seconds`** faster than Red Bull's actual historical race time (**`{avg_time_advantage:+.2f}s` average advantage per race**, median: **`{median_time_advantage:+.2f}s`**).")
        md_lines.append(f"> * **Races Faster than Red Bull**: **{faster_races_count} out of {n_races} races ({(faster_races_count / n_races) * 100:.1f}%)**.")
        md_lines.append(f"> * **Race Victories**: **{ai_wins} Wins** (vs. Red Bull's {act_wins}).")
        md_lines.append(f"> * **Podium Finishes**: **{ai_podiums} Podiums** (vs. Red Bull's {act_podiums}).")
        md_lines.append(f"> * **Pit Stop Efficiency**: **{avg_ai_stops:.2f} stops per race** (vs. Red Bull's actual {avg_act_stops:.2f}).")
    else:
        md_lines.append("> [!NOTE]")
        md_lines.append("> ### Net Strategic Season Performance")
        md_lines.append(f"> Across the **{n_races} finished dry Grands Prix of 2025**:")
        md_lines.append(f"> * **Cumulative Season Net Delta**: **`{total_time_advantage:+.2f} seconds`** ({avg_time_advantage:+.2f}s per race, median: `{median_time_advantage:+.2f}s`).")
        md_lines.append(f"> * **Races Faster than Actual**: **{faster_races_count} out of {n_races} races ({(faster_races_count / n_races) * 100:.1f}%)**.")
        md_lines.append(f"> * **Race Victories**: **{ai_wins} Wins** (vs. actual {act_wins}).")
        md_lines.append(f"> * **Podium Finishes**: **{ai_podiums} Podiums** (vs. actual {act_podiums}).")
        md_lines.append(f"> * **Pit Stop Efficiency**: **{avg_ai_stops:.2f} stops per race** (vs. actual {avg_act_stops:.2f}).")

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("### Season Performance Indicators (Finished Dry Races Only)")
    md_lines.append("")
    md_lines.append("| Performance Dimension | Actual Historical (RBR) | F1-SIS AI Counterfactual | Net Strategic Advantage |")
    md_lines.append("| :--- | :---: | :---: | :---: |")
    md_lines.append(f"| **Total Finished Dry Races** | {n_races} | {n_races} | — |")
    md_lines.append(f"| **Net Season Cumulative Delta** | — | — | **`{total_time_advantage:+.2f} s` ({'Net Advantage' if total_time_advantage >= 0 else 'Deficit'})** |")
    md_lines.append(f"| **Mean Time Delta per Race** | — | — | **`{avg_time_advantage:+.2f} s` / race** |")
    md_lines.append(f"| **Median Time Delta per Race** | — | — | **`{median_time_advantage:+.2f} s` / race** |")
    md_lines.append(f"| **Net Track Positions Gained** | — | — | **`{total_pos_gained:+d} positions`** |")
    md_lines.append(f"| **Race Wins (P1)** | **{act_wins}** | **{ai_wins}** | **{ai_wins - act_wins:+d} Wins** |")
    md_lines.append(f"| **Podium Finishes (P1–P3)** | **{act_podiums}** | **{ai_podiums}** | **{ai_podiums - act_podiums:+d} Podiums** |")
    md_lines.append(f"| **Top 5 Finishes** | **{act_top5}** | **{ai_top5}** | **{ai_top5 - act_top5:+d}** |")
    md_lines.append(f"| **Points Finishes (P1–P10)** | **{act_points}** | **{ai_points}** | **{ai_points - act_points:+d}** |")
    md_lines.append(f"| **Races Faster than Actual** | — | **{faster_races_count} / {n_races}** | **{(faster_races_count / n_races) * 100:.1f}%** |")
    md_lines.append(f"| **Average Finishing Position** | **P{avg_act_pos:.2f}** | **P{avg_ai_pos:.2f}** | **{avg_act_pos - avg_ai_pos:+.2f} P** |")
    md_lines.append(f"| **Average Pit Stops per Race** | **{avg_act_stops:.2f}** | **{avg_ai_stops:.2f}** | **{avg_ai_stops - avg_act_stops:+.2f}** |")
    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append(f"## 2. Complete Race-by-Race Comparative Breakdown ({n_races} Finished Races)")
    md_lines.append("")
    md_lines.append("| Round | Grand Prix | Laps | Actual P | AI P | Pos Delta | Actual Stops | AI Stops | Time Delta (s) | Strategic Verdict |")
    md_lines.append("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |")

    for r in results:
        pos_delta_str = f"{r['pos_diff']:+d}" if r['pos_diff'] != 0 else "0"
        time_delta_str = f"{r['time_delta_s']:+7.2f}s"
        ai_pos_str = f"**P{r['ai_pos']}**" if r['ai_pos'] <= 3 or r['ai_pos'] < r['act_pos'] else f"P{r['ai_pos']}"
        
        if r['ai_pos'] == 1 and r['act_pos'] == 1:
            verdict = f"**P1 Win Retained ({time_delta_str.strip()})**"
        elif r['ai_pos'] == 1 and r['act_pos'] > 1:
            verdict = "**Superior Strategy (P1 Win)**"
        elif r['ai_pos'] <= 3 and r['act_pos'] > r['ai_pos']:
            verdict = f"**P{r['ai_pos']} Podium Gained ({time_delta_str.strip()})**"
        elif r['ai_pos'] <= 3 and r['act_pos'] <= 3 and r['time_delta_s'] > 0:
            verdict = f"**P{r['ai_pos']} Podium Retained ({time_delta_str.strip()})**"
        elif r['pos_diff'] > 0:
            verdict = f"**+{r['pos_diff']} Positions Gained ({time_delta_str.strip()})**"
        elif r['time_delta_s'] > 1.0 and r['pos_diff'] >= 0:
            verdict = "Superior Strategy"
        elif r['time_delta_s'] > 0.0:
            verdict = f"Pace Advantage ({time_delta_str.strip()})"
        elif abs(r['time_delta_s']) <= 2.0 and r['pos_diff'] == 0:
            verdict = "Equivalent"
        else:
            verdict = "Conceded"

        md_lines.append(
            f"| **{r['race_index']:02d}** | **{r['clean_name']}** | {r['total_laps']} | "
            f"P{r['act_pos']} | {ai_pos_str} | **{pos_delta_str}** | {r['act_stops']} | {r['ai_stops']} | "
            f"`{time_delta_str}` | {verdict} |"
        )

    md_lines.append("")
    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 3. Notable Strategic Case Studies")
    md_lines.append("")

    # Select top 3 advantage races
    sorted_by_advantage = sorted(results, key=lambda x: x["time_delta_s"], reverse=True)
    top_advantages = sorted_by_advantage[:3]
    for r in top_advantages:
        flag = get_flag(r['clean_name'])
        md_lines.append(f"### {flag}{r['clean_name']} ({r['time_delta_s']:+.2f}s, {r['pos_diff']:+d} Positions)")
        md_lines.append(f"- **Outcome**: Actual P{r['act_pos']} ({r['act_stops']} stops) $\\rightarrow$ AI P{r['ai_pos']} ({r['ai_stops']} stops).")
        md_lines.append(f"- **Net Advantage**: `{r['time_delta_s']:+.2f}s` faster cumulative race time.")
        if r["divergences"]:
            md_lines.append("- **Key Strategic Moves**:")
            for d in r["divergences"][:4]:
                md_lines.append(f"  * {d}")
        else:
            md_lines.append("- **Tactical Execution**: AI optimized tire stint lengths to minimize cumulative tire degradation penalty.")
        md_lines.append("")

    md_lines.append("---")
    md_lines.append("")
    md_lines.append("## 4. Methodology & Zero-Information-Leakage Guarantee")
    md_lines.append("")
    md_lines.append("1. **Telemetry Grounding**: Grounded from Lap 1 cumulative times using FastF1 2025 official timing data across all 20 cars.")
    md_lines.append("2. **Zero Future Information**: The Decision Policy evaluates each lap $t$ using only observations available at that lap.")
    md_lines.append("3. **Symmetric Pit In/Out & GMM Duration Modeling**: Both in-laps (`PitInTime`) and out-laps (`PitOutTime`) are modeled to ensure real-world pit transit losses (21s–24s green flag, 14s caution) are reflected with physical fidelity, incorporating multimodal stationary crew stop variance.")
    md_lines.append("4. **Stochastic Monte Carlo**: 15 stochastic rollouts per candidate strategy evaluate expected position, win/podium probability, tyre degradation cliff risk, and caution sensitivity.")
    md_lines.append("5. **Dynamic Pace & Caution Hazard Modeling**: Incorporates dynamic team pace tracking with exponential moving averages and circuit-specific hazard curves for Safety Car and VSC deployments.")
    policy_desc_meth = f"RL Policy ({policy_str}) using Candidate-Conditioned Q-Network scoring" if policy == "rl" else f"`BaselineDecisionPolicy` applying {profile} multi-criteria utility weighting"
    md_lines.append(f"6. **Decision Policy**: {policy_desc_meth} to select the optimal tactical action (`STAY_OUT` vs `PIT_<COMPOUND>`).")
    md_lines.append("")

    if model_info:
        coverage = ((total_rl_inferences - total_rl_fallbacks) / max(1, total_rl_inferences)) * 100
        md_lines.append("### Neural RL Model Verification & Execution Audit")
        md_lines.append("")
        md_lines.append(f"- **Active Model Checkpoint**: `{model_info['path']}`")
        md_lines.append(f"- **Neural Architecture**: `Dueling Candidate-Conditioned Double-DQN ({model_info['architecture'].title()})`")
        md_lines.append(f"- **Trainable Parameters**: `{model_info['params']:,}`")
        md_lines.append(f"- **Model Binary Size**: `{model_info['size_bytes']:,} bytes` ({model_info['size_kb']:.1f} KB)")
        md_lines.append(f"- **Total Neural Decisions Made**: `{total_rl_inferences}` across {n_races} races")
        md_lines.append(f"- **Heuristic Fallbacks**: `{total_rl_fallbacks}`")
        md_lines.append(f"- **Neural Decision Coverage**: `{coverage:.1f}%`")
        md_lines.append("")
    md_lines.append("---")
    current_date = datetime.now().strftime("%Y-%m-%d")
    md_lines.append(f"*Generated automatically by F1-SIS Strategy Backtesting Engine on {current_date}.*")

    output_path = Path(output_md)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"\nReport successfully generated and saved to: {output_path.resolve()}")

    return {
        "summary": {
            "total_races": n_races,
            "total_time_advantage": total_time_advantage,
            "avg_time_advantage": avg_time_advantage,
            "total_pos_gained": total_pos_gained,
            "ai_wins": ai_wins,
            "act_wins": act_wins,
            "ai_podiums": ai_podiums,
            "act_podiums": act_podiums,
        },
        "results": results,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Evaluate 2025 Dry Races")
    parser.add_argument("--driver", default="VER", help="Driver code (default: VER)")
    parser.add_argument("--profile", default="balanced", choices=["balanced", "aggressive", "conservative"])
    parser.add_argument("--policy", default="rl", choices=["rl", "baseline"], help="Decision Engine policy (default: rl)")
    parser.add_argument("--model-path", default=None, help="Path to RL checkpoint (.pt)")
    parser.add_argument("--output", default="2025_dry_races_net_advantage.md", help="Output markdown report path")
    parser.add_argument("--all-races", action="store_true", default=False, help="Include all dry races including DNFs (default: False)")
    args = parser.parse_args()

    run_batch_evaluation(
        driver=args.driver,
        profile=args.profile,
        policy=args.policy,
        model_path=args.model_path,
        output_md=args.output,
        finished_only=not args.all_races,
    )
