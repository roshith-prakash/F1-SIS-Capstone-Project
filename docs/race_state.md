# Race State Management

## Overview

RaceState is the full, lap-by-lap snapshot of an entire Grand Prix as it evolves through time. It tracks all participating drivers, lap timing, tyre life, pit history, sector splits, speed trap telemetry, and track/weather conditions.

## Terminal Output Format (Split Tables)

The race state snapshot is organized into two dedicated tables that fit standard CLI widths without horizontal line wrapping:

```text
Race: British Grand Prix (2024)
Lap: 52/52
Drivers tracked: 19

Classification & Timing:
Pos  Chg Driver Team                  Last Lap  Total Time     S1     S2     S3  Gap (s)  Int (s) Behind (s)
---  --- ------ -------------------- --------- ----------- ------ ------ ------ -------- -------- ----------
  1    - HAM    Mercedes              1:30.146 2:20:42.803  29.30  36.13  24.71                        1.534
  2    - VER    Red Bull Racing       1:29.089 2:20:44.268  29.15  35.71  24.23    1.534    1.534      6.035
  3    - NOR    McLaren               1:30.913 2:20:50.350  29.63  36.65  24.64    7.569    6.035      4.918

Tyre, Pit Strategy & Telemetry:
Pos Driver Tyre (Life)       Stint Pits (Last) In/Out   Roll 3L Speeds (I1/I2/FL/ST)   Recent Laps (Last 5)               
--- ------ ----------------- ----- ----------- ------ --------- ---------------------  -----------------------------------
  1 HAM    SOFT (L15) [U]       S3     4 (L39)      -  1:29.857    294/264/246/301     [1:29.579, 1:29.667, 1:29.682, 1:29.743, 1:30.146]
  2 VER    HARD (L14) [F]       S3     4 (L39)      -  1:29.273    298/266/251/306     [1:28.952, 1:29.560, 1:29.286, 1:29.443, 1:29.089]
  3 NOR    SOFT (L15) [U]       S3     4 (L40)      -  1:30.742    298/262/250/304     [1:30.556, 1:30.515, 1:30.544, 1:30.769, 1:30.913]

Conditions:
Track status: 1
Air/track temperature: 15.8 / 25.2
Rainfall: False
Safety car: False
```

## Tracked Features

### Table 1: Classification & Timing
1. `position_change` (`Chg`): Position gain/loss relative to previous lap (`+1`, `-2`, `-`)
2. `driver` (`Driver`): 3-letter driver code
3. `team` (`Team`): Constructor/team name
4. `last_lap_time` (`Last Lap`): Formatted lap time string (`m:ss.sss`)
5. `total_race_time` (`Total Time`): Cumulative race duration (`h:mm:ss.sss`)
6. `sector_times` (`S1`, `S2`, `S3`): Individual sector split times in seconds
7. `gap_to_leader` (`Gap (s)`): Time gap to the race leader
8. `interval_to_car_ahead` (`Int (s)`): Time delta to the preceding car
9. `gap_to_car_behind` (`Behind (s)`): Time delta to the trailing car

### Table 2: Tyre, Strategy & Telemetry
10. `compound` & `tyre_life` (`Tyre (Life)`): Compound, tyre age in laps, and freshness tag (`SOFT (L15) [U]`)
11. `stint` (`Stint`): Stint number (`S1`, `S2`, `S3`)
12. `pit_count` & `last_pit_lap` (`Pits (Last)`): Total stops and lap of last pit (`4 (L39)`)
13. `pit_in` / `pit_out` (`In/Out`): Pit lane status on this lap (`IN`, `OUT`, `-`)
14. `rolling_3_lap_avg` (`Roll 3L`): 3-lap rolling pace average
15. `speed_trap_values` (`Speeds (I1/I2/FL/ST)`): Velocities at Intermediates, Finish Line & Speed Trap (km/h)
16. `recent_lap_times` (`Recent Laps (Last 5)`): Rolling list of the last 5 completed lap times

## CLI Execution

```bash
# Default view (top 10 drivers)
python -m src.race_state.view_race_state data_fastf1_v1/laps/2024/British_Grand_Prix.csv

# Show all 20 drivers
python -m src.race_state.view_race_state data_fastf1_v1/laps/2024/British_Grand_Price.csv --limit 0

# Inspect single driver card
python -m src.race_state.view_race_state data_fastf1_v1/laps/2024/British_Grand_Price.csv --driver VER

# Run compact mode
python -m src.race_state.view_race_state data_fastf1_v1/laps/2024/British_Grand_Price.csv --compact
```
