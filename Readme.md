# 🏎️ F1 Strategy Simulator

![F1 Strategy Dashboard Dashboard](file:///C:/Users/roshi/.gemini/antigravity/brain/5823e70c-606c-4c84-bb17-21b1bd5e6d9a/f1_strategy_simulator_hero_1776087542823.png)

> [!IMPORTANT]
> **PROJECT STATUS: CORE ENGINE & ML MODELS COMPLETED ✅**
> Both the **Tyre Degradation Model** and the **Race Simulation Engine** have been successfully developed, integrated, and validated. This repository serves as a complete strategy recommendation platform for Formula 1.

---

## 👥 Contributors

- [**Roshith Prakash**](https://github.com/roshith-prakash)
- [**Rushil Patel**](https://github.com/rushilpatel11)
- [**Achal Chhag**](https://github.com/achalchhag)
- [**Soumyadeep Das**](https://github.com/s-h-u-v)

---

## 🏁 Final Achievements

### 1. Multi-Circuit Laptime Predictor
Our **XGBoost Regressor** now delivers millisecond-level accuracy for lap-time forecasting across a diverse set of circuits (Monza, Bahrain, Saudi Arabia, Hungary, etc.).
- **Overall MAE**: `0.4223 s`
- **Overall R²**: `0.9966`
- **Driver Coverage**: ALB, LEC, NOR, RUS, VER consistently modeled across 2022–2025 seasons.

### 2. Tyre Degradation Engine
We have successfully modeled the non-linear "grip cliff" for all dry compounds (SOFT, MEDIUM, HARD).
- **Core Metric**: Predicts wear % with a test MAE of `1.91%`.
- **Strategy Trigger**: Engine accurately identifies the optimal pit window by monitoring a `95%` degradation threshold.

### 3. Interactive Web Dashboard
The project has evolved into a full-stack Flask application where users can:
- **Configure**: Select Driver, Track, and Year for simulation.
- **Simulate**: Run exhaustive strategy searches in seconds.
- **Explore**: Compare the top 3 simulated strategies against real-world historical results.

---

## 🔬 Technical Deep Dive

### Machine Learning Methodology
The simulator relies on two primary gradient-boosted models specialized for high-dimensional telemetry data.

#### **Laptime Model ($M_{LT}$)**
- **Algorithm**: XGBoost Regressor.
- **Key Features**: 
    - `Driver_encoded`, `Track_encoded` (Label Encoded)
    - `LapNumber` (Race progression)
    - `TyreLife` (Stint age)
    - `Year` (Extrapolation capability)
    - `Compound_HARD`, `Compound_MEDIUM`, `Compound_SOFT` (One-Hot Encoded)
- **Validation Strategy**: 80/20 train-test split, stratified by driver to ensure fairness across field performance.

#### **Degradation Model ($M_{TD}$)**
- **Algorithm**: Gradient Boosting Regressor.
- **Key Features**: `Initial_Life`, `Stint_Usage`, `Expected_Max_Life`.
- **Accuracy**: Hungarian GP (1.16% MAE), Saudi Arabia (1.68% MAE).

---

## 🏎️ Simulation Engine Mechanics

The engine performs a **Monte Carlo-style exhaustive search** over all valid compound sequences.

1.  **Permutation Generation**: Generates 18+ valid 2-stint and 3-stint sequences (e.g., `SOFT -> HARD`, `MEDIUM -> HARD -> SOFT`).
2.  **Per-Lap Simulation**: For each strategy, the engine predicts lap times and wear in a loop.
3.  **Pit Stop Logic**: If `TyreDeg >= 95.0%`, the engine simulates a transition to the next compound in the sequence, adding a track-specific `PITSTOP_TIME` penalty.
4.  **Ranking**: Strategies are ranked by total race duration. The Top 3 are cached with full telemetry for visual exploration.

---

## 🌐 Interactive Dashboard Guide

The web dashboard (powered by Flask) provides a "Mission Control" experience for strategy analysts.

1.  **Configuration Sidebar**: Select your driver (e.g., `VER`), circuit (e.g., `Italy`), and simulation year.
2.  **Strategy Leaderboard**: View the ranked results. The **Recommended Strategy** is highlighted with a gold badge.
3.  **Telemetry Exploration**: Click any strategy card to populate the **Best Strategy Telemetry** table. This shows per-lap time, compound usage, and a color-coded degradation bar.
4.  **Historical Benchmarking**: Automatically compares simulated outcomes with the **Actual Race Strategy** used in historical sessions.

---

## 📂 Project Structure

```text
Mini Project/
├── app.py                      # Flask web server & API
├── simulation_engine.py        # Core simulation & ML logic (ASCII-Safe)
├── models/                     # Serialized ML models (.pkl)
│   ├── laptime_model.pkl       # Lap-by-lap time predictor
│   └── tyre_deg_model.pkl      # Tyre wear forecast model
├── static/                     # CSS, JS, and Asset files
├── templates/                  # Flask HTML layouts (Mission Control)
├── datasets/                   # Cleaned historical race data
└── Readme.md                   # Comprehensive documentation
```

---

## 🚀 Getting Started

1. **Install Dependencies**: 
   ```bash
   pip install flask pandas numpy xgboost scikit-learn joblib fastf1
   ```
2. **Run the Dashboard**:
   ```bash
   python app.py
   ```
3. **Run the CLI Simulator**:
   ```bash
   python simulation_engine.py
   ```

---

> [!NOTE]
> This completes our NMIMS Mini Project requirements. We have developed a state-of-the-art F1 strategy tool that provides actionable insights from complex telemetry data.
