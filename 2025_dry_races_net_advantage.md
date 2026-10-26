# F1-SIS 2025 Season Dry Races: Strategic Net Advantage & Optimization Report

**Target Driver**: `VER` (Max Verstappen) | **Risk Profile**: `BALANCED` | **Decision Policy**: RL Dueling Double-DQN (`decision_engine_dqn.pt`) | **Season**: `2025`  
**Filter Scope**: `100% Dry Classified Finished Races (19 Races)` — *Wet/mixed rounds (`AUS`, `BEL`, `GBR`, `MIA`) and DNFs (`AUT` Lap 1 retirement) excluded.*

---

## 1. Executive Summary

This report documents the counterfactual backtesting evaluation of the **F1 Strategic Intelligence System (F1-SIS)** Decision & Strategy Engine across all completely dry Grands Prix of the 2025 Formula 1 World Championship **where the target driver finished the race**.

Following the implementation of out-lap pit transit modeling, the **Race-to-Flag Pit Stop Payback Filter**, Dynamic Team Pace Tracking, Pitstop GMM, and Caution Hazard modeling, the AI system achieved the following net season performance relative to actual historical pit-wall executions:

> [!TIP]
> ### Net Strategic Season Advantage Achieved
> Across the **19 finished dry Grands Prix of 2025**:
> * **Cumulative Season Net Advantage**: **`+108.79 seconds`** faster than Red Bull's actual historical race time (**`+5.73s` average advantage per race**, median: **`+6.00s`**).
> * **Races Faster than Red Bull**: **17 out of 19 races (89.5%)**.
> * **Race Victories**: **12 Wins** (vs. Red Bull's 8).
> * **Podium Finishes**: **15 Podiums** (vs. Red Bull's 14).
> * **Pit Stop Efficiency**: **1.21 stops per race** (vs. Red Bull's actual 1.63).

---

### Season Performance Indicators (Finished Dry Races Only)

| Performance Dimension | Actual Historical (RBR) | F1-SIS AI Counterfactual | Net Strategic Advantage |
| :--- | :---: | :---: | :---: |
| **Total Finished Dry Races** | 19 | 19 | — |
| **Net Season Cumulative Delta** | — | — | **`+108.79 s` (Net Advantage)** |
| **Mean Time Delta per Race** | — | — | **`+5.73 s` / race** |
| **Median Time Delta per Race** | — | — | **`+6.00 s` / race** |
| **Net Track Positions Gained** | — | — | **`+4 positions`** |
| **Race Wins (P1)** | **8** | **12** | **+4 Wins** |
| **Podium Finishes (P1–P3)** | **14** | **15** | **+1 Podiums** |
| **Top 5 Finishes** | **17** | **17** | **+0** |
| **Points Finishes (P1–P10)** | **19** | **18** | **-1** |
| **Races Faster than Actual** | — | **17 / 19** | **89.5%** |
| **Average Finishing Position** | **P2.63** | **P2.42** | **+0.21 P** |
| **Average Pit Stops per Race** | **1.63** | **1.21** | **-0.42** |

---

## 2. Complete Race-by-Race Comparative Breakdown (19 Finished Races)

| Round | Grand Prix | Laps | Actual P | AI P | Pos Delta | Actual Stops | AI Stops | Time Delta (s) | Strategic Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **01** | **Abu Dhabi Grand Prix** | 58 | P1 | **P1** | **0** | 1 | 1 | ` +11.91s` | **P1 Win Retained (+11.91s)** |
| **02** | **Azerbaijan Grand Prix** | 51 | P1 | **P1** | **0** | 1 | 1 | `  +6.00s` | **P1 Win Retained (+6.00s)** |
| **03** | **Bahrain Grand Prix** | 57 | P6 | P13 | **-7** | 2 | 1 | `  +1.65s` | Pace Advantage (+1.65s) |
| **04** | **Canadian Grand Prix** | 70 | P2 | **P1** | **+1** | 2 | 2 | `  +7.80s` | **Superior Strategy (P1 Win)** |
| **05** | **Chinese Grand Prix** | 56 | P4 | **P1** | **+3** | 1 | 1 | ` +17.12s` | **Superior Strategy (P1 Win)** |
| **06** | **Dutch Grand Prix** | 72 | P2 | **P1** | **+1** | 2 | 1 | `  +2.33s` | **Superior Strategy (P1 Win)** |
| **07** | **Emilia Romagna Grand Prix** | 63 | P1 | P4 | **-3** | 2 | 1 | ` -16.53s` | Conceded |
| **08** | **Hungarian Grand Prix** | 70 | P9 | **P6** | **+3** | 2 | 2 | `  +8.90s` | **+3 Positions Gained (+8.90s)** |
| **09** | **Italian Grand Prix** | 53 | P1 | **P1** | **0** | 1 | 1 | `  +5.96s` | **P1 Win Retained (+5.96s)** |
| **10** | **Japanese Grand Prix** | 53 | P1 | **P1** | **0** | 1 | 1 | `  +2.39s` | **P1 Win Retained (+2.39s)** |
| **11** | **Las Vegas Grand Prix** | 50 | P1 | **P1** | **0** | 1 | 1 | ` +12.43s` | **P1 Win Retained (+12.43s)** |
| **12** | **Mexico City Grand Prix** | 71 | P3 | **P2** | **+1** | 1 | 1 | ` +12.04s` | **P2 Podium Gained (+12.04s)** |
| **13** | **Monaco Grand Prix** | 78 | P4 | P4 | **0** | 2 | 1 | ` +11.77s` | Superior Strategy |
| **14** | **Qatar Grand Prix** | 57 | P1 | **P1** | **0** | 2 | 2 | `  -7.75s` | **P1 Win Retained (-7.75s)** |
| **15** | **Saudi Arabian Grand Prix** | 50 | P2 | **P1** | **+1** | 1 | 1 | ` +10.39s` | **Superior Strategy (P1 Win)** |
| **16** | **Singapore Grand Prix** | 62 | P2 | **P1** | **+1** | 1 | 1 | `  +7.85s` | **Superior Strategy (P1 Win)** |
| **17** | **Spanish Grand Prix** | 66 | P5 | **P3** | **+2** | 4 | 2 | `  +5.60s` | **P3 Podium Gained (+5.60s)** |
| **18** | **São Paulo Grand Prix** | 71 | P3 | **P2** | **+1** | 3 | 1 | `  +5.62s` | **P2 Podium Gained (+5.62s)** |
| **19** | **United States Grand Prix** | 56 | P1 | **P1** | **0** | 1 | 1 | `  +3.31s` | **P1 Win Retained (+3.31s)** |

---

## 3. Notable Strategic Case Studies

### 🇨🇳 Chinese Grand Prix (+17.12s, +3 Positions)
- **Outcome**: Actual P4 (1 stops) $\rightarrow$ AI P1 (1 stops).
- **Net Advantage**: `+17.12s` faster cumulative race time.
- **Key Strategic Moves**:
  * Lap 13: Actual pitted for MEDIUM while AI stayed out on MEDIUM (age 13).
  * Lap 35: AI pitted for fresh SOFT while actual stayed out on HARD (age 22).

### 🇺🇸 Las Vegas Grand Prix (+12.43s, +0 Positions)
- **Outcome**: Actual P1 (1 stops) $\rightarrow$ AI P1 (1 stops).
- **Net Advantage**: `+12.43s` faster cumulative race time.
- **Key Strategic Moves**:
  * Lap 25: Actual pitted for MEDIUM while AI stayed out on MEDIUM (age 25).
  * Lap 33: AI pitted for fresh SOFT while actual stayed out on HARD (age 8).

### 🇲🇽 Mexico City Grand Prix (+12.04s, +1 Positions)
- **Outcome**: Actual P3 (1 stops) $\rightarrow$ AI P2 (1 stops).
- **Net Advantage**: `+12.04s` faster cumulative race time.
- **Tactical Execution**: AI optimized tire stint lengths to minimize cumulative tire degradation penalty.

---

## 4. Methodology & Zero-Information-Leakage Guarantee

1. **Telemetry Grounding**: Grounded from Lap 1 cumulative times using FastF1 2025 official timing data across all 20 cars.
2. **Zero Future Information**: The Decision Policy evaluates each lap $t$ using only observations available at that lap.
3. **Symmetric Pit In/Out & GMM Duration Modeling**: Both in-laps (`PitInTime`) and out-laps (`PitOutTime`) are modeled to ensure real-world pit transit losses (21s–24s green flag, 14s caution) are reflected with physical fidelity, incorporating multimodal stationary crew stop variance.
4. **Stochastic Monte Carlo**: 15 stochastic rollouts per candidate strategy evaluate expected position, win/podium probability, tyre degradation cliff risk, and caution sensitivity.
5. **Dynamic Pace & Caution Hazard Modeling**: Incorporates dynamic team pace tracking with exponential moving averages and circuit-specific hazard curves for Safety Car and VSC deployments.
6. **Decision Policy**: RL Policy (RL Dueling Double-DQN (`decision_engine_dqn.pt`)) using Candidate-Conditioned Q-Network scoring to select the optimal tactical action (`STAY_OUT` vs `PIT_<COMPOUND>`).

### Neural RL Model Verification & Execution Audit

- **Active Model Checkpoint**: `C:\Roshith\Projects\F1-SIS\models\Decision Engine\decision_engine_dqn.pt`
- **Neural Architecture**: `Dueling Candidate-Conditioned Double-DQN (Dueling)`
- **Trainable Parameters**: `22,594`
- **Model Binary Size**: `98,801 bytes` (96.5 KB)
- **Total Neural Decisions Made**: `1145` across 19 races
- **Heuristic Fallbacks**: `0`
- **Neural Decision Coverage**: `100.0%`

---
*Generated automatically by F1-SIS Strategy Backtesting Engine on 2026-10-26.*