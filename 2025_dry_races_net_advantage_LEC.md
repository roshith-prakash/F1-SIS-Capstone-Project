# F1-SIS 2025 Season Dry Races: Strategic Net Advantage Report

**Driver**: `LEC` | **Risk Profile**: `BALANCED` | **Season**: `2025` | **Conditions**: `100% Dry Finished Races (18 Races)`

## 1. Executive Summary

This report presents the complete counterfactual backtesting evaluation of the **F1 Strategic Intelligence System (F1-SIS)** Decision & Strategy Engine across all completely dry Grands Prix of the 2025 Formula 1 season where the driver finished the race. The AI system makes autonomous, lap-by-lap tactical decisions (box vs. stay out, undercut, tire compound selection) using Monte Carlo rollouts and multi-criteria utility ranking, without any future information leakage.

> [!TIP]
> **Net Strategic Advantage**: Over the 18 dry finished races of the 2025 season, the F1-SIS Strategy Engine achieved a cumulative net advantage of **+136.91 seconds** (+7.61s average per race) and **+13 net positions gained** relative to actual historical pit-wall executions.

### Key Season Performance Indicators

| Metric | Actual Historical | F1-SIS AI Counterfactual | Net Advantage |
| :--- | :---: | :---: | :---: |
| **Total Dry Races** | 18 | 18 | — |
| **Net Season Cumulative Time Delta** | — | — | **+136.91 s** |
| **Mean Time Delta per Race** | — | — | **+7.61 s** |
| **Median Time Delta per Race** | — | — | **+6.40 s** |
| **Net Track Positions Gained** | — | — | **+13 positions** |
| **Average Finish Position** | P4.50 | P3.78 | **+0.72 P** |
| **Race Wins (P1)** | 0 | 3 | **+3** |
| **Podium Finishes (P1–P3)** | 6 | 10 | **+4** |
| **Top 5 Finishes** | 13 | 14 | **+1** |
| **Points Finishes (P1–P10)** | 18 | 18 | **+0** |
| **Races Faster than Actual** | — | **13 / 18** | **72.2%** |
| **Average Pit Stops per Race** | 1.56 | 1.56 | +0.00 |

## 2. Race-by-Race Comparative Breakdown

The table below details the performance comparison for every completely dry 2025 Grand Prix.

| Round | Grand Prix | Laps | Actual P | AI P | Pos Delta | Actual Stops | AI Stops | Time Delta (s) | Verdict |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 01 | **Abu Dhabi Grand Prix** | 58 | P4 | P2 | **+2** | 2 | 1 | ` +20.16s` | Superior Strategy |
| 02 | **Austrian Grand Prix** | 70 | P3 | P3 | **0** | 2 | 2 | `  +8.70s` | Superior Strategy |
| 03 | **Azerbaijan Grand Prix** | 51 | P9 | P8 | **+1** | 1 | 1 | `  +3.80s` | Superior Strategy |
| 04 | **Bahrain Grand Prix** | 57 | P4 | P5 | **-1** | 2 | 1 | ` -15.28s` | Conceded |
| 05 | **Canadian Grand Prix** | 70 | P5 | P1 | **+4** | 2 | 2 | `  +4.45s` | Superior Strategy |
| 06 | **Chinese Grand Prix** | 56 | P5 | P5 | **0** | 1 | 2 | `  +6.22s` | Superior Strategy |
| 07 | **Emilia Romagna Grand Prix** | 63 | P6 | P1 | **+5** | 2 | 2 | ` +36.21s` | Superior Strategy |
| 08 | **Hungarian Grand Prix** | 70 | P4 | P4 | **0** | 2 | 2 | ` +11.31s` | Superior Strategy |
| 09 | **Italian Grand Prix** | 53 | P4 | P2 | **+2** | 1 | 1 | `  +6.59s` | Superior Strategy |
| 10 | **Japanese Grand Prix** | 53 | P4 | P4 | **0** | 1 | 1 | `  +9.35s` | Superior Strategy |
| 11 | **Las Vegas Grand Prix** | 50 | P6 | P6 | **0** | 1 | 1 | `  -2.20s` | Conceded |
| 12 | **Mexico City Grand Prix** | 71 | P2 | P3 | **-1** | 1 | 2 | `  -7.19s` | Conceded |
| 13 | **Monaco Grand Prix** | 78 | P2 | P1 | **+1** | 2 | 2 | `  +9.36s` | Superior Strategy |
| 14 | **Qatar Grand Prix** | 57 | P8 | P3 | **+5** | 2 | 2 | ` +45.15s` | Superior Strategy |
| 15 | **Saudi Arabian Grand Prix** | 50 | P3 | P3 | **0** | 1 | 1 | `  +1.96s` | Superior Strategy |
| 16 | **Singapore Grand Prix** | 62 | P6 | P6 | **0** | 1 | 2 | `  +7.07s` | Superior Strategy |
| 17 | **Spanish Grand Prix** | 66 | P3 | P8 | **-5** | 3 | 2 | `  -6.41s` | Conceded |
| 18 | **United States Grand Prix** | 56 | P3 | P3 | **0** | 1 | 1 | `  -2.35s` | Conceded |

## 3. In-Depth Strategic Divergence Case Studies

### Qatar Grand Prix (+45.15s, +5 Positions)
- **Outcome**: Actual P8 (2 stops) $\rightarrow$ AI P3 (2 stops).
- **Net Advantage**: `+45.15s` faster cumulative race time.
- **Key Strategic Moves**:
  * Lap 7: Actual pitted for MEDIUM while AI stayed out on MEDIUM (age 7).
  * Lap 10: AI pitted for fresh MEDIUM while actual stayed out on MEDIUM (age 3).
  * Lap 32: Actual pitted for MEDIUM while AI stayed out on MEDIUM (age 22).
  * Lap 45: AI pitted for fresh MEDIUM while actual stayed out on HARD (age 13).

### Emilia Romagna Grand Prix (+36.21s, +5 Positions)
- **Outcome**: Actual P6 (2 stops) $\rightarrow$ AI P1 (2 stops).
- **Net Advantage**: `+36.21s` faster cumulative race time.
- **Key Strategic Moves**:
  * Lap 10: Actual pitted for MEDIUM while AI stayed out on MEDIUM (age 10).
  * Lap 14: AI pitted for fresh SOFT while actual stayed out on HARD (age 4).

### Abu Dhabi Grand Prix (+20.16s, +2 Positions)
- **Outcome**: Actual P4 (2 stops) $\rightarrow$ AI P2 (1 stops).
- **Net Advantage**: `+20.16s` faster cumulative race time.
- **Key Strategic Moves**:
  * Lap 16: Actual pitted for MEDIUM while AI stayed out on MEDIUM (age 16).
  * Lap 24: AI pitted for fresh MEDIUM while actual stayed out on HARD (age 8).
  * Lap 39: Actual pitted for HARD while AI stayed out on MEDIUM (age 15).

## 4. Methodology & Evaluation Architecture

1. **Data Ingestion**: Session telemetry is loaded from FastF1 2025 season race data. Laps are grounded from Lap 1 cumulative race times against all 19 competitors.
2. **Zero Future-Leakage**: The Strategy Engine only observes committed lap state $t$, never accessing laps $> t$.
3. **Stochastic Rollouts**: 15 Monte Carlo rollouts per candidate strategy evaluate expected position, win/podium probability, tyre degradation cliff risk, and caution sensitivity.
4. **Decision Policy**: `BaselineDecisionPolicy` applies multi-criteria utility weighting to select the optimal tactical action (`STAY_OUT` vs `PIT_<COMPOUND>`).
5. **Caution Dynamic Pit Loss**: Safety Car and VSC pit losses are calibrated to 14.0s (vs 21.0s–24.0s green flag), reflecting realistic delta pacings.

---
*Generated automatically by F1-SIS Strategy Backtesting Engine on 2026-10-03.*