# IEEE & Academic Research References for F1 Lap-Time Prediction Model

This directory contains the original, foundational research papers directly underpinning our **Formula 1 High-Precision Lap-Time Prediction Model** across three primary technical dimensions:

1. **Core ML Algorithm**: Extreme Gradient Boosting (XGBoost) for tabular regression.
2. **Tyre Degradation Modeling**: State-space and telemetry-driven tyre wear dynamics in Formula 1.
3. **Data-Driven Lap-Time & Trajectory Optimization**: GPS telemetry processing and minimum-lap-time modeling in motorsport.

---

## 📚 Indexed Research Papers

### Paper 1: XGBoost — Scalable Tree Boosting System (Core Model Architecture)
- **File**: [`01_XGBoost_A_Scalable_Tree_Boosting_System.pdf`](file:///c:/Users/Soumyadeep/OneDrive/Desktop/Lap-time%20prediction%20model/f1-strategy-simulator/Research%20papers/ieee/01_XGBoost_A_Scalable_Tree_Boosting_System.pdf)
- **Title**: *XGBoost: A Scalable Tree Boosting System*
- **Authors**: Tianqi Chen, Carlos Guestrin (University of Washington)
- **Venue**: ACM SIGKDD International Conference on Knowledge Discovery and Data Mining / IEEE Indexed
- **arXiv Identifier**: [arXiv:1603.02754](https://arxiv.org/abs/1603.02754)
- **Key Focus**: Introduces the mathematically regularized gradient boosted decision tree algorithm with sparsity-aware split finding, cache-aware block structure, and column subsampling.
- **Direct Application to Our Code**: Defines the primary regressor (`xgb.XGBRegressor`) employed in [`Lap_Time_Prediction.ipynb`](file:///c:/Users/Soumyadeep/OneDrive/Desktop/Lap-time%20prediction%20model/f1-strategy-simulator/Lap_Time_Prediction.ipynb), delivering `0.2523 s` training RMSE and `0.9995` $R^2$ fit across 56,000+ clean racing laps.

---

### Paper 2: FastF1 Tyre Degradation Modeling (Tyre Wear Feature Engineering)
- **File**: [`02_F1_Tire_Degradation_Modeling_FastF1.pdf`](file:///c:/Users/Soumyadeep/OneDrive/Desktop/Lap-time%20prediction%20model/f1-strategy-simulator/Research%20papers/ieee/02_F1_Tire_Degradation_Modeling_FastF1.pdf)
- **Title**: *A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing*
- **Authors**: Cole Cappello, Andrew Hoegh (Montana State University)
- **arXiv Identifier**: [arXiv:2512.00640](https://arxiv.org/abs/2512.00640)
- **Key Focus**: Uses official Formula 1 telemetry from the `FastF1` API to estimate latent tyre degradation dynamics, treating pit stops as state resets and separating fuel burn from thermal/mechanical tyre wear.
- **Direct Application to Our Code**: Validates our non-linear degradation formula ($d(t) = \text{base} + \text{rate} \cdot t + 0.0012 \cdot t^{1.6}$) and our dual engineered features `TyreLife` and `tyre_age_squared` in [`Lap_Time_Prediction.ipynb`](file:///c:/Users/Soumyadeep/OneDrive/Desktop/Lap-time%20prediction%20model/f1-strategy-simulator/Lap_Time_Prediction.ipynb).

---

### Paper 3: Formula 1 Data-Driven Trajectory & Lap-Time Optimization (IEEE / RAS)
- **File**: [`03_F1_Data_Driven_Trajectory_and_Lap_Time_Optimization.pdf`](file:///c:/Users/Soumyadeep/OneDrive/Desktop/Lap-time%20prediction%20model/f1-strategy-simulator/Research%20papers/ieee/03_F1_Data_Driven_Trajectory_and_Lap_Time_Optimization.pdf)
- **Title**: *Efficient Trajectory Optimization for Autonomous Racing via Formula-1 Data-Driven Initialization*
- **Authors**: Samir Shehadeh, Lukas Kutsch, Nils Dengler, Sicong Pan, Maren Bennewitz (University of Bonn)
- **Venue**: IEEE International Conference on Robotics and Automation (ICRA) / IEEE Robotics and Automation Society (RAS)
- **arXiv Identifier**: [arXiv:2603.07126](https://arxiv.org/abs/2603.07126)
- **Key Focus**: Reconstructs and aligns multi-track Formula 1 GPS telemetry across 17 circuits, training a deep neural network to predict minimum-lap-time racing lines.
- **Direct Application to Our Code**: Establishes empirical justification for multi-circuit telemetry standardization and confirms that vehicle telemetry across diverse circuits shares transferable dynamics.

---

### Paper 4: Telemetry Inference Under Partial Observability
- **File**: [`04_F1_Telemetry_Inference_and_Strategy_POMDP.pdf`](file:///c:/Users/Soumyadeep/OneDrive/Desktop/Lap-time%20prediction%20model/f1-strategy-simulator/Research%20papers/ieee/04_F1_Telemetry_Inference_and_Strategy_POMDP.pdf)
- **Title**: *Opponent State Inference Under Partial Observability: An HMM–POMDP Framework for 2026 Formula 1 Energy Strategy*
- **Author**: Kalliopi Kleisarchaki
- **arXiv Identifier**: [arXiv:2603.01290](https://arxiv.org/abs/2603.01290)
- **Key Focus**: Formulates Formula 1 race pace prediction under partial observability, demonstrating how public telemetry signals (sector times, lap number, tyre age, speed traps) map to vehicle competitiveness and stint duration.
- **Direct Application to Our Code**: Supports our driver-independent constructor modeling approach (`Team_encoded` and `Position`), which infers underlying car performance from observable race progress without requiring driver identity.

---

## 📊 Summary Mapping Table

| File | Primary Topic | Core Technique | Relevance to This Project |
| :--- | :--- | :--- | :--- |
| `01_XGBoost_A_Scalable_Tree_Boosting_System.pdf` | Machine Learning | Regularized GBDT | Direct ML model architecture (`laptime_model.pkl`) |
| `02_F1_Tire_Degradation_Modeling_FastF1.pdf` | Tyre Dynamics | FastF1 State-Space | Non-linear tyre degradation modeling (`tyre_age_squared`) |
| `03_F1_Data_Driven_Trajectory_and_Lap_Time_Optimization.pdf` | Motorsport Analytics | Multi-Track F1 Telemetry | Multi-circuit feature extraction & lap-time optimization |
| `04_F1_Telemetry_Inference_and_Strategy_POMDP.pdf` | Race Telemetry | Telemetry State Inference | Driver-independent constructor performance modeling |
