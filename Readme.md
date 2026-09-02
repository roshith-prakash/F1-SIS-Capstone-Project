# 🏎️ F1 Strategy Simulator

> [!IMPORTANT]
> **PROJECT STATUS: CORE ENGINE & ML MODELS COMPLETED ✅**
> Both the **Tyre Degradation Model** and the **Race Simulation Engine** have been successfully developed, integrated, and validated. This repository serves as a complete strategy recommendation platform for Formula 1.

---

## 👥 Contributors

- [**Pratham Parma**](https://github.com/pratham-parmar-37)
- [**Roshith Prakash**](https://github.com/roshith-prakash)
- [**Rushil Patel**](https://github.com/RushilPatel11)
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
f1-strategy-simulator/
├── app.py                             # Flask web server & API
├── f1_strategy_simulation_engine.py   # Core simulation & ML logic
├── lap_time_predictor.py              # Single stint / lap predictor API
├── feature_engineering.py             # Feature engineering pipeline
├── model_training.py                  # Stacked Ensemble trainer
├── model_evaluation.py                # Model evaluation & SHAP metrics
├── data_collector_v2.py               # FastF1 telemetry dataset builder
├── f1_strategy_simulation_demo.ipynb  # Interactive demonstration notebook
├── models/                            # Serialized ML models (.pkl)
│   ├── laptime_model.pkl              # Lap-by-lap time predictor
│   ├── laptime_metadata.pkl           # Laptime model metadata & encoders
│   ├── tyre_deg_model.pkl             # Tyre degradation model
│   └── tyre_deg_metadata.pkl          # Tyre degradation metadata
├── papers/                            # Academic reference papers
│   └── ieee/                          # 6 IEEE research paper PDFs
├── IEEE_RESEARCH_REFERENCES.md        # Comprehensive paper citation guide
├── static/                            # CSS & JS UI web assets
├── templates/                         # HTML layouts (index.html)
├── datasets/                          # Cleaned historical race datasets
│   ├── f1_consolidated_data.csv       # Multi-season consolidated dataset
│   ├── download_race_data.py          # FastF1 API race data downloader
│   └── download_all_races.py          # Batch download all GP races
└── Readme.md                          # Project documentation
```

---

## 🚀 Getting Started

1. **Install Dependencies**: 
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the Interactive Web Dashboard**:
   ```bash
   python app.py
   ```
3. **Run the Simulation Engine**:
   ```bash
   python f1_strategy_simulation_engine.py
   ```
4. **Explore Demonstration Notebook**:
   Open `f1_strategy_simulation_demo.ipynb` in VS Code or Jupyter Notebook.

---

> [!NOTE]
> This completes our NMIMS Mini Project requirements. We have developed a state-of-the-art F1 strategy tool that provides actionable insights from complex telemetry data.
