# Adaptive AI Decision Intelligence Platform
### F1 Strategic AI — Complete Project Specification

> **Document Version:** 1.0  
> **Status:** Architecture Finalised — Pre-Implementation  
> **Domain:** Formula 1 (Proof of Concept)  
> **Purpose:** Final-Year Capstone Project

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Core Philosophy](#2-core-philosophy)
3. [System Architecture](#3-system-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Component Specifications](#5-component-specifications)
   - 5.1 Race State Generator
   - 5.2 Event Prediction Engine
   - 5.3 Candidate Action Generator
   - 5.4 Monte Carlo Scenario Simulator
   - 5.5 Safety Car Simulation Module
   - 5.6 Opponent Model
   - 5.7 Strategy Evaluation Engine
   - 5.8 Risk Profile Engine
   - 5.9 Recommendation Engine
   - 5.10 Explainability Engine
6. [Machine Learning Components](#6-machine-learning-components)
7. [Dataset Specification](#7-dataset-specification)
8. [All Architectural Decisions & Assumptions](#8-all-architectural-decisions--assumptions)
9. [Known Limitations & Trade-offs](#9-known-limitations--trade-offs)
10. [Build Roadmap](#10-build-roadmap)

---

## 1. Project Overview

This project is an **Adaptive AI Decision Intelligence Platform** — an AI strategist that operates in real time (simulated), making lap-by-lap strategic recommendations for a Formula 1 race driver.

Unlike conventional race analytics projects that load an entire race and produce predictions after the fact, this system processes **one lap at a time**, exactly as a real race engineer receives telemetry. After every lap, the system:

1. Updates the complete race state
2. Predicts future race events probabilistically
3. Generates all currently valid strategic actions
4. Simulates the next N laps for every candidate action using Monte Carlo methods
5. Evaluates each simulated strategy against multiple criteria
6. Scores strategies through the lens of a chosen risk profile
7. Recommends the optimal action
8. Explains why that action was chosen (with a counterfactual for the runner-up)

This creates an **AI strategist**, not a prediction model.

---

## 2. Core Philosophy

### What Makes This Different from Typical Capstone Projects

| Typical Student Project | This Project |
|---|---|
| Predicts final race result | Makes decisions lap-by-lap |
| Single model, single output | Pipeline of coordinated models |
| No uncertainty quantification | Monte Carlo distributions over outcomes |
| No explanation | SHAP + natural language reasoning |
| No strategy alternatives | All candidate strategies scored and ranked |
| One risk tolerance | 5 risk profiles with different scoring weights |

### The Core Loop

```
Lap Data Arrives
      │
      ▼
Race State Generator
      │
      ▼
Event Prediction Engine
      │
      ▼
Candidate Action Generator
      │
      ▼
Scenario Simulation Engine  ◄─── (Monte Carlo, 200 rollouts per action)
      │
      ▼
Strategy Evaluation Engine
      │
      ▼
Risk Profile Engine
      │
      ▼
Recommendation Engine
      │
      ▼
Explanation Engine
      │
      ▼
Wait for Next Lap
```

This loop repeats every lap for the entire race duration.

---

## 3. System Architecture

### Layered Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Orchestrator                       │
│              process_lap(lap_data) → Recommendation          │
└──────────┬──────────────┬───────────────┬────────────────────┘
           │              │               │
    ┌──────▼──────┐  ┌────▼────┐  ┌──────▼──────────────┐
    │  Race State │  │  Event  │  │   Action Generator   │
    │  Generator  │  │Predictor│  │  + Constraint Engine │
    └──────┬──────┘  └────┬────┘  └──────┬──────────────┘
           │              │               │
           └──────────────▼───────────────┘
                          │
              ┌───────────▼──────────────┐
              │  Monte Carlo Simulator   │
              │  ┌────────────────────┐  │
              │  │ Tiered Opponent    │  │
              │  │ Model (T1/T2/T3)  │  │
              │  ├────────────────────┤  │
              │  │ SC Simulation      │  │
              │  │ (Trigger/Duration/ │  │
              │  │  Window/Restart)   │  │
              │  ├────────────────────┤  │
              │  │ Vectorized Batch   │  │
              │  │ Inference (200 MC) │  │
              │  └────────────────────┘  │
              └───────────┬──────────────┘
                          │
              ┌───────────▼──────────────┐
              │  Strategy Evaluator      │
              │  Score = Σ wᵢ(profile)   │
              │         × fᵢ(outcome)    │
              └───────────┬──────────────┘
                          │
              ┌───────────▼──────────────┐
              │  Recommendation +        │
              │  SHAP Explainer +        │
              │  Counterfactual          │
              └──────────────────────────┘
```

### Train / Serve Separation

```
OFFLINE (done once, before simulation):
  FastF1 data → Feature Engineering → Train Models → Serialize (joblib/ONNX)

ONLINE (every lap, during simulation):
  Load serialized models → Inference only → No retraining during simulation
```

No model is retrained during a live simulation. All models are pre-trained and loaded at startup.

---

## 4. Technology Stack

### Backend

| Layer | Technology | Purpose |
|---|---|---|
| API Framework | FastAPI (Python) | REST endpoints + WebSocket for lap-by-lap streaming |
| Orchestration | Python | Core simulation loop, component coordination |
| Serialization | joblib / ONNX | Serialize trained models for offline → online handoff |

### Machine Learning

| Library | Role |
|---|---|
| XGBoost | SC trigger, pit decision, rain, overtake models |
| LightGBM | Lap time (quantile regression), undercut success |
| Scikit-Learn | Calibration (Platt scaling / isotonic), preprocessing, SMOTE via imbalanced-learn |
| SHAP | TreeExplainer for feature importance and explanation generation |
| CatBoost | Available but not selected as primary — XGBoost/LightGBM preferred for this dataset size |
| TensorFlow / PyTorch | Stretch goal only (sequence models if dataset proves sufficient) |

### Data

| Library | Role |
|---|---|
| FastF1 | Primary data source — pulls from Ergast API + official F1 timing API |
| Pandas | Data manipulation, feature engineering, stint-level aggregations |
| NumPy | Numerical operations, vectorized Monte Carlo rollout arithmetic |

### Simulation

| Component | Implementation |
|---|---|
| Monte Carlo Engine | Custom Python — vectorized batch inference using NumPy arrays |
| SC Simulation Module | Custom Python — integrated into Monte Carlo rollout loop |
| Opponent Model | Custom Python — tiered rule-based system |

### Database

| Environment | Database | Purpose |
|---|---|---|
| Development | SQLite | Lightweight, no server required, easy to inspect |
| Production | PostgreSQL | Persistent race session storage, replay data |

### Deployment

| Tool | Purpose |
|---|---|
| Docker | Containerise backend + model artifacts for reproducible deployment |
| GitHub Actions | CI/CD pipeline — linting, testing, automated deployment |
| AWS / Render | Backend hosting (FastAPI server) |

> **Note:** Frontend is explicitly out of scope for the current phase. The backend API is designed as a standalone service that can be consumed by any frontend later.

---

## 5. Component Specifications

### 5.1 Race State Generator

Converts incoming lap data into a complete, structured race state object.

**Primary Entity: Constructor, not Driver**

The system models strategy at the **constructor level**, not the driver level. Drivers frequently transfer between teams across seasons (e.g. Hamilton to Ferrari), making driver identity an unstable long-term feature. The constructor's car performance, pit crew speed, tyre strategy philosophy, and historical pace are all tied to the constructor — not the individual behind the wheel.

- The **ego entity** in every simulation is a constructor (team) competing in a specific race.
- `constructor_season_id` (e.g. `"RedBull_2023"`) is the primary model identifier.
- `driver_id` is retained as a **secondary feature** only — driver skill and style can influence tyre degradation and sector pace, but it is not the primary grouping key.
- All historical aggregations (median pit windows, pace benchmarks, degradation curves) are computed at the **constructor × season × circuit** level.

**State Schema (features per constructor per lap):**

| Feature | Type | Source | Notes |
|---|---|---|---|
| `lap_number` | int | FastF1 | Absolute race lap |
| `race_completion_pct` | float | Derived | lap / total_laps |
| `position` | int | FastF1 | End-of-lap position |
| `constructor_season_id` | categorical | Derived | Primary entity key e.g. "RedBull_2023" |
| `driver_id` | categorical | FastF1 | Secondary — driving style proxy only |
| `tyre_compound` | categorical | FastF1 | Soft / Medium / Hard / Inter / Wet |
| `tyre_age` | int | FastF1 | Laps completed on current set |
| `tyre_age_squared` | float | Derived | Non-linear degradation proxy |
| `lap_time` | float | FastF1 | Seconds |
| `sector_1_time` | float | FastF1 | Seconds |
| `sector_2_time` | float | FastF1 | Seconds |
| `sector_3_time` | float | FastF1 | Seconds |
| `lap_time_delta_from_stint_start` | float | Derived | Degradation trend |
| `lap_time_rolling_3lap_avg` | float | Derived | Smoothed pace |
| `gap_ahead` | float | Derived | Reconstructed from positions + lap times |
| `gap_behind` | float | Derived | Reconstructed from positions + lap times |
| `laps_since_last_pit` | int | Derived | Resets at each stop |
| `track_status` | categorical | FastF1 | Green / SC / VSC / Red |
| `under_sc` | bool | Derived | True if SC deployed this lap |
| `air_temp` | float | FastF1 | Session-level, sparse |
| `track_temp` | float | FastF1 | Session-level, sparse |
| `rainfall` | bool | FastF1 | Session-level boolean |
| `circuit_id` | categorical | FastF1 | Per-circuit identifier |

**Fuel load is intentionally excluded.** The model learns fuel effect implicitly through `lap_number`, `race_completion_pct`, and `laps_since_last_pit`.

---

### 5.2 Event Prediction Engine

Predicts the probability or expected value of future race events. Each prediction is an input to the simulator.

| Event | Output Type | Target Horizon |
|---|---|---|
| Rain probability | Probability (0–1) | Next 5 laps |
| Safety Car probability | Probability (0–1) | Next 5 laps |
| Tyre degradation | Expected rate (s/lap) | Current stint |
| Undercut success probability | Probability (0–1) | If pit taken now |
| Pit traffic risk | Probability (0–1) | If pit taken now |
| Lap time degradation | Expected delta (s) | Next 3 laps |
| Overtake probability | Probability (0–1) | Current lap |
| Pit window quality | Score (0–1) | Current lap |

All outputs become stochastic inputs to the Monte Carlo rollouts.

---

### 5.3 Candidate Action Generator

Generates all strategically valid actions at the current lap. Invalid actions are pruned by the constraint engine.

**Possible Actions:**

- Stay out
- Pit now — Soft
- Pit now — Medium
- Pit now — Hard
- Pit next lap — (each compound)
- Push mode (extract maximum lap time)
- Standard pace
- Tyre conservation mode
- Fuel saving mode

**Constraint Engine Rules (derived from race state, not hard-coded):**

- Cannot pit if the compound is already exhausted (all sets used)
- Soft tyres unavailable if `rainfall == True` (wet conditions)
- Minimum tyre usage rule: cannot pit before completing minimum stint laps
- Cannot pit if already in pit lane
- Fuel saving only valid if within defined margin threshold
- Tyre conservation disabled if gap_behind < threshold (under threat)

---

### 5.4 Monte Carlo Scenario Simulator

The core engine. For every candidate action, runs 200 stochastic rollouts across 10 laps (or remaining laps, whichever is less), producing a distribution of outcomes.

**Simulation Horizon:**
```
horizon = min(10, remaining_laps)
```

**Per-Rollout Process:**
```
State(t)
    │
    ▼  (sample stochastic inputs)
Apply Action
    │
    ▼
Predict next state using ML models
    │
    ▼
State(t+1)
    │
    ▼
Repeat until horizon reached
```

**Stochastic Sampling per Rollout:**
- Rain deployment: sampled from `rain_probability`
- SC deployment: sampled from `sc_trigger_probability`
- Tyre degradation: sampled from predicted rate ± noise
- Opponent decisions: sampled from pit probability functions

**Vectorized Batch Inference:**
All 200 rollouts are executed in a single batched model call per lap per action. This reduces 200 sequential model calls to 1 batched call, enabling the system to stay within the 10–15 second latency budget.

**Latency estimate:**
```
6 actions × 200 rollouts × 10 laps × 5 model calls = 60,000 model calls
With batch inference → ~6 batched calls per lap
Inference time: ~0.5 seconds
Overhead budget: remaining 9.5–14.5 seconds for orchestration, WebSocket, rendering
```

---

### 5.5 Safety Car Simulation Module

SC is not treated as a simple probability modifier. When SC deploys mid-rollout, it triggers a full scenario branch.

**Four Sub-Models:**

| Sub-Model | Function |
|---|---|
| SC Trigger Model | P(SC deploys on lap t) — calibrated classifier |
| SC Duration Model | How many laps SC lasts — historical distribution |
| SC Lap Time Model | Fixed delta (~10–15s slower) + field compression logic |
| SC Pit Window Evaluator | Is pitting under SC beneficial? Calculates effective pit cost vs. green flag |

**SC Effects on Simulation:**

- **Gap compression:** All cars within ~5s bunch up behind SC. `gap_ahead` and `gap_behind` reset toward 0.
- **Free pit window:** Pit stop under SC costs ~0 track position (vs. ~22s under green). This makes pitting under SC frequently the dominant strategy.
- **SC restart:** First 2 laps after restart have elevated overtake probability. Cold tyre risk modelled.
- **Opponent behaviour override:** Tier 1 and Tier 2 opponents will pit under SC if `tyre_age > 10` (unconditional rule).

**Mid-Rollout SC Branching:**
```
For each lap in rollout:
  Sample SC trigger from sc_trigger_model
  If SC deploys:
    Sample SC duration from duration distribution
    Apply gap compression
    Re-evaluate all candidate actions under SC rules
    SC pit becomes dominant option for most profiles
  Continue rollout under SC-modified state
```

---

### 5.6 Opponent Model

Opponents are modelled at the **constructor level**, consistent with the ego entity. Each opponent slot in the simulation represents a constructor competing in the same race, not an individual driver.

**Tiered architecture to balance accuracy against computational cost:**

```
TIER 1: Full Simulation
  Who: Constructors at positions P(n-2) to P(n+2) relative to ego constructor
  What: Full lap-by-lap state tracking, including pit decisions
  Why: Undercut risk, overcut, traffic — these require accurate modelling

TIER 2: Statistical Approximation
  Who: All other constructors outside Tier 1
  What: Predicted lap time distribution ± noise. No per-lap decision making.
  Why: Position tracking, SC field compression. No strategy interaction.

TIER 3: Leader Track
  Who: P1 constructor always, regardless of ego position
  What: Always tracked at full fidelity
  Why: Gap-to-leader calculation
```

**Tier 1 Opponent Pit Decision Logic (Option C — Data-Driven Rules):**

Pit windows are derived from historical FastF1 data per compound, aggregated at the **constructor × circuit** level where sufficient data exists, otherwise falling back to circuit-level averages:

| Compound | Median Stint | Std Dev |
|---|---|---|
| Soft | ~18 laps | ±4 laps |
| Medium | ~28 laps | ±6 laps |
| Hard | ~38 laps | ±8 laps |

Pit probability is modelled as a sigmoid function of tyre age relative to the compound median. The closer the constructor is to (or past) the median stint length, the higher the pit probability per lap.

**SC Override:** Any opponent with `tyre_age > 10` pits unconditionally under Safety Car.

---

### 5.7 Strategy Evaluation Engine

Every simulated strategy is evaluated using a weighted multi-criteria utility function:

```
Score(strategy, profile) = Σ wᵢ(profile) × fᵢ(strategy_outcome)
```

Where:
- `wᵢ(profile)` is the weight for criterion i under the chosen risk profile
- `fᵢ(strategy_outcome)` is the normalised score [0, 1] for criterion i

**Evaluation Criteria:**

| Criterion | Description |
|---|---|
| Expected finishing position | Lower = better |
| Expected race time | Total estimated race time |
| Tyre degradation risk | How close to cliff edge by race end |
| Pit stop count | Fewer stops generally better |
| Rain exposure | Laps on wrong compound in rain |
| Traffic risk post-pit | Whether pit exit puts driver in traffic |
| Overtake opportunity | Position delta possible on fresh tyres |
| Strategy robustness | How sensitive the outcome is to SC/rain events |
| Confidence | Variance of outcomes across 200 rollouts |

---

### 5.8 Risk Profile Engine

Five strategic profiles. Each profile is a **weight vector** applied to the same simulation outputs. The simulation itself does not change — only the scoring weights change.

**Weight Vectors:**

| Criterion | Aggressive | Conservative | Defensive | Opportunistic | Balanced |
|---|---|---|---|---|---|
| Expected Position | 0.40 | 0.20 | 0.15 | 0.25 | 0.30 |
| Race Time | 0.15 | 0.30 | 0.20 | 0.10 | 0.25 |
| Tyre Risk | 0.05 | 0.25 | 0.30 | 0.05 | 0.15 |
| Rain Exposure | 0.05 | 0.15 | 0.10 | 0.25 | 0.10 |
| Traffic Risk | 0.05 | 0.05 | 0.05 | 0.05 | 0.05 |
| Overtake Opportunity | 0.20 | 0.00 | 0.00 | 0.10 | 0.05 |
| Strategy Robustness | 0.05 | 0.05 | 0.15 | 0.15 | 0.05 |
| Confidence | 0.05 | 0.00 | 0.05 | 0.05 | 0.05 |

**Profile Descriptions:**

- **Aggressive:** Chase position, accept risk. Higher expected reward, higher variance.
- **Conservative:** Protect current position, minimise tyre and rain risk.
- **Defensive:** Priority on preventing undercuts from rivals immediately behind.
- **Opportunistic:** Wait for SC or rain events to extract maximum value from uncertainty.
- **Balanced:** Balanced trade-off between expected gain and risk.

---

### 5.9 Recommendation Engine

Selects the highest-scoring action under the active risk profile and formats the output.

**Output:**

```
Recommended Action: Pit This Lap — Medium
Expected Finish: P3
Confidence: 91%
Expected Race Time: 1h 28m 12s
Strategy Score: 0.847 (Aggressive Profile)
Runner-up: Stay Out (Score: 0.761)
Score Delta: +0.086
```

---

### 5.10 Explainability Engine

Every recommendation includes three explanation layers:

**Layer 1 — SHAP Feature Importance**
SHAP TreeExplainer (optimised for tree-based models) identifies which features most influenced the recommendation. Not shown directly to user — used to generate Layer 2.

**Layer 2 — Natural Language Reasoning**
Human-readable reasons generated from SHAP values + prediction outputs:

```
Recommended because:
  → Tyre degradation is expected to exceed threshold in 2 laps
  → Current pit window has low traffic risk (22%)
  → Rain probability rises to 42% after Lap 30
  → Pitting now maximises expected finishing position under Aggressive profile
```

**Layer 3 — Counterfactual Explanation**
Always answers: *"Why not the runner-up strategy?"*

```
Why not Stay Out?
  → Staying out risks tyre cliff in 2 laps (+4.2s lap time expected)
  → Rain after Lap 30 on aged Mediums would likely cost P3 position
  → Score delta: Pit Now is 10.3% better under current profile
```

---

## 6. Machine Learning Components

### Model Registry

| Model | Task | Algorithm | Notes |
|---|---|---|---|
| Lap Time Predictor | Regression | LightGBM (quantile) | Quantile regression gives uncertainty bounds, not just point estimates |
| Tyre Degradation Model | Regression | XGBoost with monotonic constraints | Degradation cannot decrease — monotonic constraint enforced |
| Pit Decision Predictor | Classification | XGBoost (calibrated) | Outputs true probabilities, not just labels |
| SC Trigger Model | Classification | XGBoost + SMOTE | SC is rare — imbalanced class handling required |
| SC Duration Model | Regression | Historical distribution (empirical) | Simple histogram is sufficient |
| Rain Prediction Model | Classification | XGBoost (calibrated) | Must output calibrated probabilities |
| Undercut Success Model | Classification | LightGBM | Custom label engineering required |
| Overtake Probability Model | Classification | XGBoost | Derived from gap, tyre delta, circuit features |

### Why Not LSTM / Transformer

Sequence models are listed as a stretch goal only. Reasons for not using them as primary models:
- FastF1 data yields ~140,000 clean training rows — insufficient for sequence models to outperform tree-based models
- LSTMs require sequential data per driver per race — the sequences are short (~60 timesteps)
- XGBoost/LightGBM with lag features captures the same temporal patterns with far less data and better inference speed
- Inference latency for LSTMs is higher — problematic within the 10–15 second per-lap budget

### Calibration

All classification models producing probabilities (SC trigger, rain, undercut, overtake) must be **calibrated** using Platt scaling or isotonic regression. An uncalibrated classifier saying "70% probability" may be systematically over- or under-confident. The Monte Carlo simulation depends on these probabilities being accurate.

---

## 7. Dataset Specification

### Data Source

**FastF1** Python library — pulls from the Ergast API and the official F1 timing API.

### Seasons Used

```
Train:      2018 – 2022  (5 seasons)
Validation: 2023         (1 season)
Test:       2024         (1 season)
```

Temporal split is mandatory. Random splits are invalid for this data because consecutive laps within a race are strongly correlated.

### Raw Data Volume

```
7 seasons × 23 races × 20 drivers × ~58 laps = ~187,000 raw lap records
After filtering: ~140,000 clean training laps
```

### Filtering Rules

| Filter | Reason |
|---|---|
| Remove formation laps | Not representative of race pace |
| Remove in-laps and out-laps | Artificially slow; distort tyre age signals |
| Remove SC / VSC laps | Lap times are not representative of true pace |
| Remove DNF laps | Driver may have been coasting / damage |
| Remove laps with missing data | API gaps in FastF1 are not uncommon |

### Engineered Features (Must Be Computed)

| Feature | Engineering Method |
|---|---|
| `gap_ahead / gap_behind` | Reconstruct from cumulative lap times + positions |
| `tyre_degradation_rate` | `(lap_time_N - lap_time_stint_start) / tyre_age` |
| `lap_time_delta_from_stint_start` | Absolute delta per lap within stint |
| `lap_time_rolling_3lap_avg` | 3-lap rolling mean per driver |
| `tyre_age_squared` | Non-linear degradation proxy |
| `laps_since_last_pit` | Reset at each pit stop |
| `race_completion_pct` | `lap_number / total_laps` |
| `constructor_season_id` | `f"{constructor}_{year}"` (e.g. "RedBull_2023") — primary entity key |
| `driver_id` | Raw driver identifier — secondary feature only |
| `undercut_success` | Custom label — see definition below |
| `sc_deployed` | Derived from track status codes |

### Undercut Success Label Definition

An undercut attempt is identified when:
- Driver A pits while within 3 seconds of Driver B
- Driver A is on the same compound or older tyres

An undercut succeeds if:
- Driver A exits the pits ahead of Driver B, OR
- Driver A closes the gap by more than 2 seconds within 3 laps post-pit

### Known Data Gaps

| Missing Data | Workaround |
|---|---|
| Fuel load | Implicit via `lap_number` and `race_completion_pct` |
| Wind speed / humidity | Not available — excluded |
| DRS per-lap telemetry | Zone-based boolean per circuit added as static feature |
| Opponent strategy intent | Historical replay only |
| Granular weather per lap | Session-level values used; interpolated where possible |
| Brake / tyre temperatures | No workaround — excluded |

### Circuit Heterogeneity

A single model is used across all circuits, with `circuit_id` as a categorical feature. This allows the model to learn circuit-specific tyre degradation and pace patterns without maintaining 24 separate models.

Each circuit contributes approximately:
```
7 seasons × 20 constructors × 58 laps ≈ 8,000 laps per circuit
```
Sufficient for XGBoost to learn meaningful circuit-specific effects via categorical splits.

### Constructor vs. Driver — Data Aggregation

All historical aggregations (pit window medians, degradation benchmarks, pace curves) are computed at the **constructor × season × circuit** level. Driver-level data is not aggregated across constructors, even if the same driver appears for multiple teams across seasons. This ensures that:

- A driver's performance at Team A does not pollute Team B's constructor model
- Constructor identity (car, crew, strategy culture) is preserved as the stable unit
- The `constructor_season_id` feature captures within-constructor variation without being elevated to a primary grouping key

---

## 8. All Architectural Decisions & Assumptions

This section is a complete record of every design decision and assumption made during architectural review.

### Simulation

| Decision | Choice | Rationale |
|---|---|---|
| Simulation horizon | 10 laps (min with remaining) | Enough to capture full pit windows and SC windows |
| Monte Carlo rollouts | 200 per action | Balances statistical accuracy with 15s latency budget |
| Batch inference | Vectorized (all 200 rollouts in one model call) | Reduces inference time from ~15s to ~0.5s |
| State transition contract | Formal `RaceState` dataclass | Enforces modularity; all models must conform to this interface |

### Opponent Modelling

| Decision | Choice | Rationale |
|---|---|---|
| Opponent model type | Tiered (T1/T2/T3) | Balances accuracy for close rivals against computational cost for distant ones |
| Tier 1 radius | P(n-2) to P(n+2) | Within undercut / overcut range |
| Tier 2 method | Predicted lap time ± noise | Position tracking only — no strategic decisions |
| Tier 3 | Always track P1 | Gap-to-leader required for race strategy |
| Opponent pit decision | Option C: Data-driven rule-based | Avoids circular dependency, more realistic than using ego driver's model |
| SC override for opponents | Always pit if tyre_age > 10 | Mirrors real-world team behaviour under SC |

### Safety Car

| Decision | Choice | Rationale |
|---|---|---|
| SC modelling depth | Full deployment simulation | SC changes strategy fundamentally — cannot be a simple weight modifier |
| SC trigger | Calibrated classifier per lap | Rare event — must handle class imbalance (SMOTE) |
| SC duration | Empirical histogram | Simple, accurate, no model needed |
| SC effects | Gap compression + free pit window + restart overtake boost | Covers all strategic implications |
| Mid-rollout SC | Full scenario branch | When SC triggers mid-simulation, remaining laps are re-simulated under SC rules |

### Machine Learning

| Decision | Choice | Rationale |
|---|---|---|
| Primary ML framework | XGBoost + LightGBM | Best performance on tabular data at this scale |
| LSTM / Transformer | Stretch goal only | Insufficient data; tree models with lag features are superior at this scale |
| Calibration | Platt scaling / isotonic regression on all classifiers | Monte Carlo depends on accurate probabilities, not just rank ordering |
| Tyre model constraint | Monotonic constraint on degradation | Degradation cannot decrease — constraints improve model realism |
| Lap time model type | Quantile regression | Gives uncertainty bounds, not point estimates — feeds MC uncertainty |
| Explainability | SHAP TreeExplainer | Optimised for tree-based models; fast at inference time |
| Counterfactual | Always explain runner-up strategy | Stronger explanations than just "why this?" |

### Dataset

| Decision | Choice | Rationale |
|---|---|---|
| Data source | FastF1 (2018–2024) | Only reliable public source of per-lap F1 timing data |
| Train/test split | Temporal by season (2022 / 2023 / 2024) | Prevents data leakage from correlated laps within a race |
| Fuel load | Excluded — implicit via lap number | No telemetry available; lap number captures fuel burn trend |
| Circuit model | Single model with circuit_id categorical feature | 8,000 laps per circuit — sufficient for XGBoost to learn per-circuit patterns |
| Primary modeling entity | Constructor, not driver | Drivers transfer between teams; constructor identity is stable across seasons |
| Constructor identifier | `constructor_season_id` (e.g. "RedBull_2023") | Captures car + crew + strategy culture as a unit; season suffix accounts for regulation changes |
| Driver as feature | Secondary feature (`driver_id`) only | Captures within-constructor style differences without being the primary grouping key |
| Historical aggregation level | Constructor × season × circuit | Prevents driver data from one constructor contaminating another constructor's model |
| Undercut label | Custom engineered | Not available in raw data — must be defined and applied |

### Risk Profiles

| Decision | Choice | Rationale |
|---|---|---|
| Number of profiles | 5 (Aggressive, Conservative, Defensive, Opportunistic, Balanced) | Covers the realistic spectrum of real-world F1 team philosophies |
| Profile implementation | Weight vectors over evaluation criteria | Simulation stays identical; only scoring changes. Clean, auditable, explainable. |

### Domain Scope

| Decision | Choice | Rationale |
|---|---|---|
| Domain | F1-specific (no domain generalisation) | Reduces architectural complexity; generalisation is future scope |

---

## 9. Known Limitations & Trade-offs

### Data Limitations

- **Gap reconstruction is approximate.** Gaps derived from cumulative lap times are not identical to real timing gaps, especially under SC or when lapped traffic is involved. Careful filtering required.
- **Tyre degradation is a noisy signal.** True degradation is contaminated by fuel burn, track evolution, traffic, and driver input. Feature engineering can reduce but not eliminate this noise.
- **Weather data is coarse.** Only session-level weather is available. Lap-level weather changes (e.g. light rain beginning mid-race) cannot be captured precisely.
- **~25% of raw laps are filtered out.** Formation laps, in/out laps, SC laps, DNF laps reduce the effective dataset size.

### Simulation Limitations

- **Opponents do not adapt.** Tier 1 opponents use rule-based pit decisions. They do not respond to your driver's strategy. Real teams react to each other's moves.
- **Physics-free simulation.** All state transitions are driven by ML models, not physics. Model errors compound across the 10-lap horizon. Confidence degrades with distance.
- **200 rollouts may be insufficient for rare events.** A 5% SC probability requires many rollouts before SC scenarios are adequately represented. This is a known limitation of the sample size.

### ML Limitations

- **No online learning.** Models are not updated during a race. Real F1 teams adjust their models as new data arrives.
- **Historical training data may not reflect current regulations.** Tyre compounds, stint lengths, and pit strategies vary by season and regulation change.

---

## 10. Build Roadmap

### Phase 1 — Data & Dataset (Weeks 1–2)
- FastF1 data collection pipeline
- Feature engineering (all derived features)
- Undercut success label engineering
- SC trigger label extraction
- Train / validation / test split construction
- Data dictionary documentation

### Phase 2 — Model Training (Weeks 3–4)
- Lap time predictor (LightGBM quantile)
- Tyre degradation model (XGBoost monotonic)
- Pit decision predictor (calibrated XGBoost)
- SC trigger model (XGBoost + SMOTE)
- SC duration model (empirical distribution)
- Rain prediction model (calibrated XGBoost)
- Undercut success model (LightGBM)
- Overtake probability model (XGBoost)
- Model evaluation, calibration, and serialization

### Phase 3 — Core Engine (Weeks 5–6)
- Race State Generator (formal RaceState dataclass)
- Event Prediction Engine (all sub-models integrated)
- Candidate Action Generator + Constraint Engine
- Tiered Opponent Model (T1/T2/T3)

### Phase 4 — Simulation (Weeks 7–8)
- Monte Carlo Simulator (vectorized batch inference)
- SC Simulation Module (all four sub-models)
- SC mid-rollout branching logic

### Phase 5 — Strategy Intelligence (Weeks 9–10)
- Strategy Evaluation Engine (utility function)
- Risk Profile Engine (weight vectors, 5 profiles)
- Recommendation Engine

### Phase 6 — Explainability (Week 11)
- SHAP TreeExplainer integration
- Natural language reasoning generator
- Counterfactual explanation engine

### Phase 7 — Backend API (Week 12)
- FastAPI orchestrator
- WebSocket for lap-by-lap streaming
- Post-race replay module

---

*Document generated from architectural review session. All decisions are final pending implementation.*
