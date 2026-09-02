# Safety Car Probability Model — Learnings and Decisions

## Purpose

The Safety Car (SC) module is intended to provide calibrated uncertainty to the Monte Carlo strategy simulator after every completed lap:

```text
Race State Manager -> SC probability model -> P(SC in 1/3/5/10 laps) -> Monte Carlo strategy simulator
```

It must output probabilities, never a Boolean prediction, and it must not use information from future laps.

The model's useful question is not “will an accident happen?” Instead it is: **given the currently known race state, how should the simulator weight SC scenarios in the near future?**

## What the current notebook does

`SC_Hazard_Probability_Model.ipynb` converts FastF1 driver-lap data into one causal race-state row per race lap. It then:

1. Decodes `TrackStatus` at race level.
2. Defines `SC_START` when a lap contains status code `4` and the previous lap did not.
3. Defines `VSC_START` similarly using status code `6`.
4. Creates future SC targets for 1, 3, 5, and 10 laps.
5. Splits races chronologically into train, validation, and test sets.
6. Compares historical baselines, an offset-specific hazard model, and the saved XGBoost trigger model.
7. Replays held-out races lap by lap and produces `race_id`, `lap`, `P_SC_1`, `P_SC_3`, `P_SC_5`, and `P_SC_10`.

The notebook has been run end-to-end on the available data.

## Data and event-definition findings

- Available data: 172 races, 188,482 driver-lap rows, and 10,317 aggregated race-lap rows.
- Detected events: 128 SC starts and 92 VSC starts.
- The target is correctly an **event start**, not every lap on which SC is active. This avoids counting a single SC once for every driver and once for every active lap.
- The notebook's known-case validation was plausible:
  - Italian GP 2021: VSC starts on laps 1 and 42; SC starts on lap 25.
  - Abu Dhabi GP 2021: VSC starts on lap 34; SC starts on lap 51.
- Consecutive SC status laps are treated as one SC event.

## Evaluation design that should be retained

- Do not randomly split laps: laps from one race are highly correlated.
- Split chronologically by race. The present split is 103 training races, 34 validation races, and 35 test races.
- The test period begins at 2024 round 14 and includes the available 2025 races.
- The next-lap SC-start prevalence in the test set is about 0.78%. This is a rare-event problem, so accuracy is misleading.
- Use Brier score, log loss, PR-AUC, ROC-AUC, and calibration curves. Prefer race-level bootstrap confidence intervals when results are presented formally.
- Evaluate each probability window against its own future target. A good one-lap probability does not automatically imply a good ten-lap probability.

## Current evidence

The current test-set results for next-lap SC starts are:

| Model | Brier (lower is better) | Log loss (lower is better) | PR-AUC | ROC-AUC |
| --- | ---: | ---: | ---: | ---: |
| Historical race-progress | 0.00773 | 0.04540 | 0.01109 | 0.62879 |
| Circuit + race-progress | 0.00806 | 0.05390 | 0.00816 | 0.47435 |
| Offset-specific hazard | 0.00774 | 0.04557 | 0.00978 | 0.56591 |
| Existing XGBoost | 0.00774 | 0.04578 | 0.00803 | 0.48434 |

The historical race-progress baseline is currently the best model on the held-out period. The more elaborate hazard model is close, but does not beat it. Its PR-AUC is only slightly above the 0.78% event prevalence.

The same pattern persists at the longer windows. For example, at 10 laps, the historical baseline has a Brier score of 0.06966 and ROC-AUC of 0.57683, while the offset-specific model has 0.06991 and 0.52699 respectively.

### Interpretation

The available lap-summary data does not contain enough advance evidence to reliably predict sudden collisions, mechanical failures, debris, or race-control decisions. This is an important result, not a failure of the project. The data supports a historical risk estimate better than it supports a strong imminent-event detector.

## Problems discovered and corrected

### Final-race-length leakage

The early version calculated race progress using the final observed lap number. That value is only known after the race and therefore leaks future information.

The notebook now uses `planned_race_laps`: a causal historical proxy drawn only from completed earlier races at the same circuit, with a global/default fallback. This is an interim solution. The Race State Manager should eventually supply the official, pre-race scheduled lap count.

### Deleted laps are not retirements

The early version treated `Deleted` driver laps as retirements/incidents. There are 1,691 deleted driver-lap records, and many are track-limit deletions. They must not be labelled as retirements.

The notebook now calls these `deleted_laps` and excludes fabricated retirement counts. A true retirement feature should be added only when an authoritative, causal retirement/status feed is available.

### Repeating one hazard across every future lap

The early version transformed one next-lap hazard using `1 - (1 - h)^n`. This gives monotonic probabilities, but assumes the same risk on every future lap.

The notebook now trains a pooled landmark discrete-time hazard model. Every state at lap `t` produces at-risk training rows for offsets `k = 1...10`, allowing it to estimate:

```text
h(t + k | RaceState at t, k)
```

The cumulative result remains:

```text
P(SC within n laps) = 1 - product from k=1 to n of (1 - h(t + k))
```

This preserves `P_SC_1 <= P_SC_3 <= P_SC_5 <= P_SC_10` while using separate hazards at each offset.

### Calibration

Isotonic calibration produced very few unique probability levels on this rare target, making the result too coarse. The current notebook uses smooth Platt scaling trained only on the later validation races. Calibration should continue to be assessed on held-out races, not assumed correct because a calibration step exists.

## Current architectural decision

Use the historical race-progress baseline as the production fallback for the simulator now. It is simple, interpretable, calibrated/evaluable, and currently has the best held-out performance.

The probability module still helps the overall system, but in a narrower and honest role:

- It weights SC branches in Monte Carlo simulations.
- It exposes uncertainty to strategy recommendations.
- It allows strategies to be compared under plausible SC risk by race phase and circuit.
- It does **not** claim to foretell accidents from the present feature set.

The simulator can therefore use:

```text
P(SC soon) -> sample/weight SC scenarios -> compare pit and tyre strategies under those scenarios
```

## What not to claim

- Do not claim that the current model predicts imminent crashes or SC deployments reliably.
- Do not claim that an ML model is better than the historical baseline; the current evidence says otherwise.
- Do not use accuracy as the headline metric for this rare-event problem.
- Do not treat Abu Dhabi 2021 and Italian 2021 as performance validation; they validate event decoding and target construction only.
- Do not predict SC duration in this module.

## Limitations still present

- `planned_race_laps` is an historical proxy, not the official race schedule. Replace it when the Race State Manager has pre-race event metadata.
- The model is history-aware through causal rolling/cumulative features, but it is not yet a recurrent or transformer sequence model.
- The raw data has no reliable causal retirement-status field.
- `SpeedST` coverage is not DRS use; it must not be presented as a real DRS-density feature.
- Current future hazards use the present state plus offset/progress. They do not simulate future tyre changes, weather changes, retirements, or field compression. A state-transition simulator could later supply those counterfactual states.
- The circuit-aware baseline may have weak estimates for circuits with sparse historical samples.
- The saved XGBoost comparison is directional because it was trained around a different feature contract; it needs the same authoritative state inputs for a fully fair comparison.

## Recommended next steps

1. **Integrate the historical baseline with the Monte Carlo simulator.** Treat it as the default SC-risk source and verify that strategy rankings sensibly change as SC risk changes.
2. **Finish the Race State Manager contract.** It should provide official scheduled laps, true retirements, pit activity, tyre state, race-control flags, weather, and reliable gaps at the end of each lap.
3. **Add real predictive signals only when causal.** Useful candidates include local yellows, incident/stoppage reports, mechanical anomalies, detailed weather changes, close battles, and position compression.
4. **Keep the hazard-model gate.** Promote any ML model only if it materially beats the baseline on future, race-disjoint test seasons in Brier score and log loss, with uncertainty intervals.
5. **Use race-level resampling for confidence intervals.** Do not present a tiny metric difference as an improvement without checking uncertainty.
6. **Consider state-transition forecasting later.** For longer windows, feed plausible future race states from the simulator into the hazard model instead of relying on state-at-`t` features alone.

## Bottom line

The project should continue, but the valuable deliverable at this stage is a calibrated SC-risk baseline for scenario simulation—not an overclaimed accident-prediction model. The baseline already supports the system's core decision problem: selecting strategies that remain robust across plausible Safety Car outcomes.
