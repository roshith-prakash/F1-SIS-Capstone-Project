# F1-SIS 2025 Season Dry Races: Strategic Net Advantage Report

**Driver**: `VER` | **Decision Policy**: RL Dueling Double-DQN (`decision_engine_dqn.pt`) | **Season**: `2025` | **Conditions**: `100% Dry Finished Races (19 Races)`

## 1. Executive Summary

This report presents the complete counterfactual backtesting evaluation of the **F1 Strategic Intelligence System (F1-SIS)** Decision & Strategy Engine across all completely dry Grands Prix of the 2025 Formula 1 season where the driver finished the race. The AI system makes autonomous, lap-by-lap tactical decisions (box vs. stay out, undercut, tire compound selection) using Monte Carlo rollouts and multi-criteria utility ranking, without any future information leakage.

> [!TIP]
> **Net Strategic Advantage**: Over the 19 dry finished races of the 2025 season, the F1-SIS Strategy Engine achieved a cumulative net advantage of **+89.60 seconds** (+4.72s average per race) and **+2 net positions gained** relative to actual historical pit-wall executions.

### Key Season Performance Indicators

| Metric | Actual Historical | F1-SIS AI Counterfactual | Net Advantage |
| :--- | :---: | :---: | :---: |
| **Total Dry Races** | 19 | 19 | — |
| **Net Season Cumulative Time Delta** | — | — | **+89.60 s** |
| **Mean Time Delta per Race** | — | — | **+4.72 s** |
| **Median Time Delta per Race** | — | — | **+5.96 s** |
| **Net Track Positions Gained** | — | — | **+2 positions** |
| **Average Finish Position** | P2.63 | P2.53 | **+0.11 P** |
| **Race Wins (P1)** | 8 | 10 | **+2** |
| **Podium Finishes (P1–P3)** | 14 | 15 | **+1** |
| **Top 5 Finishes** | 17 | 17 | **+0** |
| **Points Finishes (P1–P10)** | 19 | 18 | **-1** |
| **Races Faster than Actual** | — | **15 / 19** | **78.9%** |
| **Average Pit Stops per Race** | 1.63 | 1.26 | -0.37 |

## 2. Race-by-Race Comparative Breakdown

The table below details the performance comparison for every completely dry 2025 Grand Prix.

| Round | Grand Prix | Laps | Actual P | AI P | Pos Delta | Actual Stops | AI Stops | Time Delta (s) | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | **Abu Dhabi Grand Prix** | 58 | P1 | P1 | **0** | 1 | 1 | ` +11.91s` | Superior Strategy |
| 02 | **Azerbaijan Grand Prix** | 51 | P1 | P1 | **0** | 1 | 1 | `  +0.96s` | Advantage |
| 03 | **Bahrain Grand Prix** | 57 | P6 | P13 | **-7** | 2 | 1 | ` +18.37s` | Advantage |
| 04 | **Canadian Grand Prix** | 70 | P2 | P1 | **+1** | 2 | 2 | `  +7.80s` | Superior Strategy |
| 05 | **Chinese Grand Prix** | 56 | P4 | P1 | **+3** | 1 | 1 | ` +17.12s` | Superior Strategy |
| 06 | **Dutch Grand Prix** | 72 | P2 | P2 | **0** | 2 | 2 | `  -0.07s` | Equivalent |
| 07 | **Emilia Romagna Grand Prix** | 63 | P1 | P4 | **-3** | 2 | 1 | ` -16.53s` | Conceded |
| 08 | **Hungarian Grand Prix** | 70 | P9 | P6 | **+3** | 2 | 2 | `  +8.90s` | Superior Strategy |
| 09 | **Italian Grand Prix** | 53 | P1 | P1 | **0** | 1 | 1 | `  +5.96s` | Superior Strategy |
| 10 | **Japanese Grand Prix** | 53 | P1 | P1 | **0** | 1 | 1 | `  +2.39s` | Superior Strategy |
| 11 | **Las Vegas Grand Prix** | 50 | P1 | P1 | **0** | 1 | 1 | `  +6.78s` | Superior Strategy |
| 12 | **Mexico City Grand Prix** | 71 | P3 | P2 | **+1** | 1 | 1 | `  +7.22s` | Superior Strategy |
| 13 | **Monaco Grand Prix** | 78 | P4 | P4 | **0** | 2 | 1 | `  +4.39s` | Superior Strategy |
| 14 | **Qatar Grand Prix** | 57 | P1 | P2 | **-1** | 2 | 2 | ` -11.29s` | Conceded |
| 15 | **Saudi Arabian Grand Prix** | 50 | P2 | P1 | **+1** | 1 | 1 | ` +10.39s` | Superior Strategy |
| 16 | **Singapore Grand Prix** | 62 | P2 | P1 | **+1** | 1 | 1 | `  +7.85s` | Superior Strategy |
| 17 | **Spanish Grand Prix** | 66 | P5 | P3 | **+2** | 4 | 2 | `  +5.60s` | Superior Strategy |
| 18 | **São Paulo Grand Prix** | 71 | P3 | P2 | **+1** | 3 | 1 | `  +2.02s` | Superior Strategy |
| 19 | **United States Grand Prix** | 56 | P1 | P1 | **0** | 1 | 1 | `  -0.16s` | Equivalent |

## 3. In-Depth Strategic Divergence Case Studies

### Bahrain Grand Prix (+18.37s, -7 Positions)
- **Outcome**: Actual P6 (2 stops) $\rightarrow$ AI P13 (1 stops).
- **Net Advantage**: `+18.37s` faster cumulative race time.
- **Key Strategic Moves**:
  * Lap 10: Actual pitted for SOFT while AI stayed out on SOFT (age 10).
  * Lap 18: AI pitted for fresh HARD while actual stayed out on HARD (age 8).
  * Lap 26: Actual pitted for HARD while AI stayed out on HARD (age 8).

### Chinese Grand Prix (+17.12s, +3 Positions)
- **Outcome**: Actual P4 (1 stops) $\rightarrow$ AI P1 (1 stops).
- **Net Advantage**: `+17.12s` faster cumulative race time.
- **Key Strategic Moves**:
  * Lap 13: Actual pitted for MEDIUM while AI stayed out on MEDIUM (age 13).
  * Lap 35: AI pitted for fresh SOFT while actual stayed out on HARD (age 22).

### Abu Dhabi Grand Prix (+11.91s, +0 Positions)
- **Outcome**: Actual P1 (1 stops) $\rightarrow$ AI P1 (1 stops).
- **Net Advantage**: `+11.91s` faster cumulative race time.
- **Key Strategic Moves**:
  * Lap 23: Actual pitted for MEDIUM while AI stayed out on MEDIUM (age 23).
  * Lap 35: AI pitted for fresh SOFT while actual stayed out on HARD (age 12).

## 4. Methodology & Evaluation Architecture

1. **Data Ingestion**: Session telemetry is loaded from FastF1 2025 season race data. Laps are grounded from Lap 1 cumulative race times against all 19 competitors.
2. **Zero Future-Leakage**: The Strategy Engine only observes committed lap state $t$, never accessing laps $> t$.
3. **Stochastic Rollouts**: 15 Monte Carlo rollouts per candidate strategy evaluate expected position, win/podium probability, tyre degradation cliff risk, and caution sensitivity.
4. **Decision Policy**: RL Policy (RL Dueling Double-DQN (`decision_engine_dqn.pt`)) using Candidate-Conditioned Q-Network scoring to select the optimal tactical action (`STAY_OUT` vs `PIT_<COMPOUND>`).
5. **Caution Dynamic Pit Loss**: Safety Car and VSC pit losses are calibrated to 14.0s (vs 21.0s–24.0s green flag), reflecting realistic delta pacings.

### Neural RL Model Verification & Execution Audit

- **Active Model Checkpoint**: `C:\Roshith\Projects\F1-SIS\models\Decision Engine\decision_engine_dqn.pt`
- **Neural Architecture**: `Dueling Candidate-Conditioned Double-DQN (Dueling)`
- **Trainable Parameters**: `22,594`
- **Model Binary Size**: `98,801 bytes` (96.5 KB)
- **Total Neural Decisions Made**: `1145` across 19 races
- **Heuristic Fallbacks**: `0`
- **Neural Decision Coverage**: `100.0%`

---
*Generated automatically by F1-SIS Strategy Backtesting Engine on 2026-10-04.*