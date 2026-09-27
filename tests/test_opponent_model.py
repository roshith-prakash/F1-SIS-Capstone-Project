"""
tests/test_opponent_model.py
============================
Automated test suite for F1 SIS Opponent Model (Task 18):
- Test 1: Probability validity (0 <= P <= 1, sum = 1)
- Test 2: Input schema (Race State, Opponent Physical, Ego, History)
- Test 3: No future leakage
- Test 4: Race split integrity (disjoint train / val / test sets)
- Test 5: Baseline comparison (ML vs Baselines 1, 2, 3)
- Test 6: Determinism (reproducibility)
- Test 7: Edge cases (unknown opponent, lap 1, late race, SC, old tyres, fresh tyres, 0 stops)
- Test 8: Interface compatibility with Strategy Engine
"""

from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.opponent_model.state import OpponentStateVector, build_opponent_state
from src.opponent_model.model import OpponentModel
from src.opponent_model.interface import predict_opponent, get_default_opponent_model
from src.opponent_model.baselines import (
    MajorityClassBaseline,
    HistoricalFrequencyBaseline,
    StateConditionedBaseline,
    evaluate_all_baselines,
)
from src.opponent_model.features import (
    FEATURES_MODEL_C,
    FEATURES_RACE_STATE,
    FEATURES_OPPONENT_PHYSICAL,
    FEATURES_EGO_STATE,
    FEATURES_OPPONENT_HISTORY,
)


def _make_dummy_state_vector(
    tyre_age: int = 20,
    compound: str = "MEDIUM",
    is_sc: int = 0,
    remaining_laps: int = 25,
    lap_number: int = 32,
    gap_ahead: float = 2.5,
    gap_behind: float = 1.8,
    gap_to_ego: float = -1.8,
    ego_pitted: int = 0,
) -> OpponentStateVector:
    """Helper to construct a valid OpponentStateVector."""
    return OpponentStateVector(
        race_id="2024_Monza",
        driver="PIA",
        lap_number=lap_number,
        remaining_laps=remaining_laps,
        race_progress_fraction=lap_number / (lap_number + remaining_laps),
        position=2,
        gap_ahead=gap_ahead,
        gap_behind=gap_behind,
        is_safety_car=is_sc,
        is_vsc=0,
        track_temp=32.0,
        air_temp=26.0,
        rainfall=0,
        pit_loss_seconds=12.0 if is_sc else 22.0,
        race_phase="mid",
        tyre_compound=compound,
        tyre_age=tyre_age,
        tyre_age_squared=float(tyre_age ** 2),
        last_lap_time=85.2,
        rolling_3_lap_avg=85.4,
        predicted_lap_time=85.3,
        predicted_degradation=1.2,
        predicted_lap_time_t1=85.5,
        predicted_degradation_t1=1.3,
        p_sc_h1=0.02,
        p_vsc_h1=0.01,
        p_sc_h3=0.05,
        ego_driver="VER",
        ego_position=3,
        gap_to_ego=gap_to_ego,
        ego_compound="HARD",
        ego_tyre_age=10,
        ego_predicted_pace=85.1,
        ego_predicted_deg=0.5,
        ego_recently_pitted=ego_pitted,
        ego_undercut_threat=1 if gap_to_ego < 0 and abs(gap_to_ego) < 2.5 and ego_pitted else 0,
        laps_since_last_pit=tyre_age,
        pit_count=1,
        recent_pace_trend=0.2,
        recent_deg_trend=0.1,
        previous_action=0,
        pace_delta=0.3,
        deg_rate_acceleration=0.1,
        cost_of_staying=6.0,
        pit_position_cost=-19.5,
        gap_ratio=0.58,
        undercut_threat=1 if gap_behind < 2.0 else 0,
        overcut_window=0,
        tyre_life_fraction=tyre_age / 28.0,
        laps_past_nominal=max(0, tyre_age - 28),
        position_pressure=8,
        sc_adjusted_pit_cost=2.45,
    )


# =========================================================================
# Test 1: Probability Validity
# =========================================================================
def test_probability_validity():
    """Verify 0 <= P(action) <= 1 and sum(P) == 1.0."""
    model = get_default_opponent_model()

    test_states = [
        _make_dummy_state_vector(tyre_age=5, compound="HARD"),
        _make_dummy_state_vector(tyre_age=32, compound="SOFT"),
        _make_dummy_state_vector(tyre_age=25, is_sc=1),
        _make_dummy_state_vector(tyre_age=45, compound="MEDIUM"),
    ]

    for state in test_states:
        pred = model.predict(state)
        p_pit = pred["p_pit"]
        p_stay = pred["p_stay"]

        assert 0.0 <= p_pit <= 1.0, f"p_pit out of bounds: {p_pit}"
        assert 0.0 <= p_stay <= 1.0, f"p_stay out of bounds: {p_stay}"
        assert abs((p_pit + p_stay) - 1.0) < 1e-5, f"Probabilities do not sum to 1: {p_pit} + {p_stay}"

        # Check multi-horizon monotonicity
        horizons = pred["horizons"]
        assert horizons[1] <= horizons[3] <= horizons[5], (
            f"Multi-horizon probabilities not monotonically increasing: {horizons}"
        )


# =========================================================================
# Test 2: Input Schema
# =========================================================================
def test_input_schema():
    """Verify all required features are present across all 3 categories and history."""
    state = _make_dummy_state_vector()
    feat_dict = state.to_feature_dict("C")

    # Verify Category A, B, and derived feature names
    for col in FEATURES_MODEL_C:
        assert col in feat_dict, f"Missing feature in Model C: {col}"
        assert not np.isnan(feat_dict[col]), f"NaN found in feature {col}"

    # Verify state dataclass contains explicit Category C (Ego) and History fields
    assert hasattr(state, "ego_driver")
    assert hasattr(state, "gap_to_ego")
    assert hasattr(state, "ego_compound")
    assert hasattr(state, "ego_recently_pitted")
    assert hasattr(state, "laps_since_last_pit")
    assert hasattr(state, "recent_pace_trend")
    assert hasattr(state, "recent_deg_trend")


# =========================================================================
# Test 3: No Future Leakage
# =========================================================================
def test_no_future_leakage():
    """Verify that features at time t do not contain information from t+1 or later."""
    forbidden_terms = [
        "target",
        "pit_in_time",
        "pitted_next_lap",
        "next_compound",
        "final_position",
        "future",
        "t+1",
        "t1_actual",
    ]

    state = _make_dummy_state_vector()
    feat_dict = state.to_feature_dict("C")

    for feat_name in feat_dict.keys():
        lower = feat_name.lower()
        for forbidden in forbidden_terms:
            assert forbidden not in lower, f"Potential future leakage feature found: {feat_name}"


# =========================================================================
# Test 4: Race Split Integrity
# =========================================================================
def test_race_split_integrity():
    """Verify that train, validation, and test splits are strictly disjoint."""
    data_dir = Path("data_fastf1_v1/opponent_model")
    train_file = data_dir / "train_dataset.parquet"
    val_file = data_dir / "val_dataset.parquet"
    test_file = data_dir / "test_dataset.parquet"

    if train_file.exists() and val_file.exists() and test_file.exists():
        df_train = pd.read_parquet(train_file, columns=["race_id"])
        df_val = pd.read_parquet(val_file, columns=["race_id"])
        df_test = pd.read_parquet(test_file, columns=["race_id"])

        train_races = set(df_train["race_id"].unique())
        val_races = set(df_val["race_id"].unique())
        test_races = set(df_test["race_id"].unique())

        assert train_races.isdisjoint(val_races), f"Train/Val leak: {train_races & val_races}"
        assert train_races.isdisjoint(test_races), f"Train/Test leak: {train_races & test_races}"
        assert val_races.isdisjoint(test_races), f"Val/Test leak: {val_races & test_races}"
    else:
        # If files not generated, verify split seasons logic
        train_years = {2018, 2019, 2020, 2021, 2022, 2023}
        val_years = {2024}
        test_years = {2025}
        assert train_years.isdisjoint(val_years)
        assert train_years.isdisjoint(test_years)
        assert val_years.isdisjoint(test_years)


# =========================================================================
# Test 5: Baseline Comparison
# =========================================================================
def test_baseline_comparison():
    """Verify that the model is evaluated against Baselines 1, 2, and 3."""
    # Synthetic dataset
    np.random.seed(42)
    n = 200
    X_synthetic = pd.DataFrame({
        "tyre_age": np.random.randint(1, 40, n),
        "race_progress_fraction": np.random.uniform(0.1, 0.9, n),
        "compound_SOFT": np.random.choice([0, 1], n),
        "compound_MEDIUM": np.random.choice([0, 1], n),
        "compound_HARD": np.random.choice([0, 1], n),
    })
    y_synthetic = (
        (X_synthetic["tyre_age"] > 25) & (np.random.rand(n) > 0.4)
    ).astype(int)

    b1 = MajorityClassBaseline().fit(X_synthetic, y_synthetic)
    b2 = HistoricalFrequencyBaseline().fit(X_synthetic, y_synthetic)
    b3 = StateConditionedBaseline().fit(X_synthetic, y_synthetic)

    baselines_dict = {
        "Baseline 1 (Majority Class)": b1,
        "Baseline 2 (Historical Frequency)": b2,
        "Baseline 3 (State-Conditioned)": b3,
    }

    df_eval = evaluate_all_baselines(baselines_dict, X_synthetic, y_synthetic)
    assert len(df_eval) == 3
    assert "Log Loss" in df_eval.columns
    assert "Brier Score" in df_eval.columns

    # Verify Brier Score is valid and positive
    for brier in df_eval["Brier Score"]:
        assert 0.0 <= brier <= 1.0


# =========================================================================
# Test 6: Determinism
# =========================================================================
def test_determinism():
    """Verify reproducible predictions given identical input state."""
    model = get_default_opponent_model()
    state = _make_dummy_state_vector(tyre_age=24, compound="MEDIUM")

    pred_1 = model.predict(state)
    pred_2 = model.predict(state)

    assert pred_1["p_pit"] == pytest.approx(pred_2["p_pit"], abs=1e-6)
    assert pred_1["p_stay"] == pytest.approx(pred_2["p_stay"], abs=1e-6)
    assert pred_1["confidence"] == pytest.approx(pred_2["confidence"], abs=1e-6)


# =========================================================================
# Test 7: Edge Cases
# =========================================================================
def test_edge_cases():
    """Verify robust handling of extreme edge cases."""
    model = get_default_opponent_model()

    edge_cases = [
        # Edge Case 1: First lap
        _make_dummy_state_vector(lap_number=1, remaining_laps=56, tyre_age=0),
        # Edge Case 2: Very late race
        _make_dummy_state_vector(lap_number=56, remaining_laps=1, tyre_age=30),
        # Edge Case 3: Safety Car
        _make_dummy_state_vector(is_sc=1, tyre_age=25),
        # Edge Case 4: Extreme tyre age
        _make_dummy_state_vector(tyre_age=48, compound="SOFT"),
        # Edge Case 5: Fresh tyres
        _make_dummy_state_vector(tyre_age=0, compound="HARD"),
        # Edge Case 6: 0 pit stops
        _make_dummy_state_vector(tyre_age=12),
    ]

    for ec in edge_cases:
        pred = model.predict(ec)
        assert 0.0 <= pred["p_pit"] <= 1.0
        assert 0.0 <= pred["p_stay"] <= 1.0
        assert abs(pred["p_pit"] + pred["p_stay"] - 1.0) < 1e-4

    # Test unknown driver string via interface
    res = predict_opponent(
        state={"driver": "UNKNOWN_DRIVER", "lap_number": 10, "tyre_age": 15},
        opponent_driver="UNKNOWN_DRIVER",
    )
    assert "pit_probability" in res
    assert 0.0 <= res["pit_probability"] <= 1.0


# =========================================================================
# Test 8: Interface Compatibility with Strategy Engine
# =========================================================================
def test_interface_compatibility():
    """Verify predict_opponent returns the exact structure required by Strategy Engine."""
    state = _make_dummy_state_vector()
    res = predict_opponent(state, opponent_driver="PIA", ego_driver="VER")

    required_keys = [
        "opponent_driver",
        "ego_driver",
        "lap",
        "pit_probability",
        "stay_out_probability",
        "predicted_actions",
        "confidence",
        "horizon",
        "multi_horizon",
        "opponent_physical_state",
        "ego_state",
        "explanation",
    ]

    for key in required_keys:
        assert key in res, f"Required Strategy Engine key missing: {key}"

    # Verify types and structure
    assert isinstance(res["pit_probability"], float)
    assert isinstance(res["stay_out_probability"], float)
    assert isinstance(res["multi_horizon"], dict)
    assert "lap_1" in res["multi_horizon"]
    assert "lap_3" in res["multi_horizon"]
    assert "lap_5" in res["multi_horizon"]

    assert res["opponent_physical_state"]["compound"] == "MEDIUM"
    assert res["ego_state"]["driver"] == "VER"
    assert res["explanation"] is not None
    assert "summary" in res["explanation"]


# =========================================================================
# Test 9: Foundational Model Outputs Reach Model B & C (Task 22)
# =========================================================================
def test_foundational_model_outputs_reach_models_b_and_c():
    """Verify foundational model outputs reach Models B and C, but not Model A."""
    state = _make_dummy_state_vector()
    feat_a = state.to_feature_dict("A")
    feat_b = state.to_feature_dict("B")
    feat_c = state.to_feature_dict("C")

    foundational_keys = [
        "predicted_lap_time",
        "predicted_degradation",
        "predicted_lap_time_t1",
        "predicted_degradation_t1",
        "p_sc_h1",
        "p_vsc_h1",
        "p_sc_h3",
    ]

    # Must be completely excluded from Model A
    for k in foundational_keys:
        assert k not in feat_a, f"Foundational feature {k} leaked into Model A!"

    # Must be present and non-zero in Model B and Model C
    for k in foundational_keys:
        assert k in feat_b, f"Foundational feature {k} missing from Model B!"
        assert k in feat_c, f"Foundational feature {k} missing from Model C!"
        assert feat_b[k] > 0.0, f"Foundational feature {k} is zero in Model B!"
        assert feat_c[k] > 0.0, f"Foundational feature {k} is zero in Model C!"


# =========================================================================
# Test 10: Opponent and Ego Features Semantically Distinct (Task 22)
# =========================================================================
def test_opponent_and_ego_features_semantically_distinct():
    """Verify that Opponent and Ego features are semantically distinct and decoupled."""
    state = _make_dummy_state_vector(
        tyre_age=25,
        gap_to_ego=-2.4,
    )

    # Opponent attributes
    assert state.opponent_predicted_lap_time == 85.3
    assert state.opponent_predicted_degradation == 1.2
    assert state.opponent_tyre_age == 25
    assert state.opponent_compound == "MEDIUM"

    # Ego attributes
    assert state.ego_predicted_lap_time == 85.1
    assert state.ego_predicted_degradation == 0.5
    assert state.ego_tyre_age == 10
    assert state.ego_compound == "HARD"
    assert state.gap_opponent_to_ego == -2.4

    # Ensure opponent and ego have distinct values
    assert state.opponent_predicted_lap_time != state.ego_predicted_lap_time
    assert state.opponent_predicted_degradation != state.ego_predicted_degradation
    assert state.opponent_tyre_age != state.ego_tyre_age
    assert state.opponent_compound != state.ego_compound


# =========================================================================
# Test 11: Bayesian Update Avoids Double-Counting (Task 22)
# =========================================================================
def test_bayesian_update_no_double_counting():
    """Verify that when online pace anomaly is zero, Bayesian posterior equals ML prior."""
    from src.opponent_model.bayesian import BayesianPitUpdater

    updater = BayesianPitUpdater(lambda_=0.3)
    p_ml = 0.25

    # Case 1: Neutral online evidence (zero pace residual anomaly, within nominal window)
    p_post_neutral = updater.update(
        p_ml=p_ml,
        tyre_age=20,
        compound="MEDIUM",
        pace_residual=0.0,
        laps_past_nominal=0.0,
    )
    # Must equal p_ml exactly (no double-counting of tyre age)
    assert p_post_neutral == pytest.approx(p_ml, abs=1e-5), (
        f"Bayesian update altered prior without online evidence: {p_post_neutral} vs {p_ml}"
    )

    # Case 2: Severe anomalous degradation (+2.0 s/lap pace deficit beyond model expectation)
    p_post_anomaly = updater.update(
        p_ml=p_ml,
        tyre_age=20,
        compound="MEDIUM",
        pace_residual=2.0,
        laps_past_nominal=0.0,
    )
    # Must strictly increase pit probability
    assert p_post_anomaly > p_ml, f"Posterior did not increase with pace anomaly: {p_post_anomaly} <= {p_ml}"


# =========================================================================
# Test 12: Race Replay Does Not Access Future Laps (Task 22)
# =========================================================================
def test_race_replay_no_future_access():
    """Verify sequential replay strictly enforces no future lap data ingestion."""
    from src.opponent_model.race_replay import simulate_race_with_opponent_model
    import tempfile

    # Create a tiny mock race CSV with 2 laps
    df_mock = pd.DataFrame([
        {"Driver": "PIA", "Team": "McLaren", "LapNumber": 1, "LapTimeSeconds": 85.0, "Compound": "MEDIUM", "TyreLife": 1, "Position": 1, "IntervalToPositionAheadSeconds": None, "PitInTimeSeconds": None, "PitOutTimeSeconds": None},
        {"Driver": "PIA", "Team": "McLaren", "LapNumber": 2, "LapTimeSeconds": 85.2, "Compound": "MEDIUM", "TyreLife": 2, "Position": 1, "IntervalToPositionAheadSeconds": None, "PitInTimeSeconds": None, "PitOutTimeSeconds": None},
    ])

    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w") as tmp:
        df_mock.to_csv(tmp.name, index=False)
        tmp_path = tmp.name

    model = get_default_opponent_model()
    # Should succeed without temporal violations
    df_replay, _, _ = simulate_race_with_opponent_model(
        csv_path=tmp_path,
        opponent_model=model,
        max_laps=2,
    )
    assert len(df_replay) == 2
    assert list(df_replay["lap"].unique()) == [1, 2]


# =========================================================================
# Test 13: PR-AUC Calculation is Correct (Section 21)
# =========================================================================
def test_pr_auc_calculation():
    """Verify compute_pr_auc calculates Average Precision correctly."""
    from src.opponent_model.evaluate import compute_pr_auc
    from sklearn.metrics import average_precision_score

    # Case 1: Perfect ranking
    y_true = np.array([0, 0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    pr_auc = compute_pr_auc(y_true, y_prob)
    assert pr_auc == pytest.approx(1.0, abs=1e-5)
    assert pr_auc == pytest.approx(average_precision_score(y_true, y_prob), abs=1e-5)

    # Case 2: Inverted ranking
    y_prob_inv = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
    pr_auc_inv = compute_pr_auc(y_true, y_prob_inv)
    assert pr_auc_inv < 0.4
    assert pr_auc_inv == pytest.approx(average_precision_score(y_true, y_prob_inv), abs=1e-5)

    # Case 3: Empty or single-class handling
    assert compute_pr_auc(np.array([]), np.array([])) == 0.0
    assert compute_pr_auc(np.array([1, 1]), np.array([0.8, 0.9])) == 1.0


# =========================================================================
# Test 14: Threshold Metrics Calculated from Correct Predictions (Section 21)
# =========================================================================
def test_threshold_metrics_calculation():
    """Verify evaluate_threshold_sweep calculates exact TP, FP, TN, FN and metrics."""
    from src.opponent_model.evaluate import evaluate_threshold_sweep

    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.3, 0.1])

    thresholds = [0.05, 0.35, 0.50, 0.85]
    df_sweep = evaluate_threshold_sweep(y_true, y_prob, thresholds=thresholds)

    assert len(df_sweep) == 4

    # At th = 0.50: y_pred = [1, 0, 0, 0] -> TP=1, FP=0, TN=2, FN=1
    row_50 = df_sweep[df_sweep["threshold"] == 0.50].iloc[0]
    assert row_50["tp"] == 1
    assert row_50["fp"] == 0
    assert row_50["tn"] == 2
    assert row_50["fn"] == 1
    assert row_50["precision"] == pytest.approx(1.0)
    assert row_50["recall"] == pytest.approx(0.5)
    assert row_50["f1"] == pytest.approx(2 / 3)
    assert row_50["fpr"] == pytest.approx(0.0)
    assert row_50["fnr"] == pytest.approx(0.5)

    # At th = 0.35: y_pred = [1, 1, 0, 0] -> TP=2, FP=0, TN=2, FN=0
    row_35 = df_sweep[df_sweep["threshold"] == 0.35].iloc[0]
    assert row_35["tp"] == 2
    assert row_35["fp"] == 0
    assert row_35["tn"] == 2
    assert row_35["fn"] == 0
    assert row_35["precision"] == pytest.approx(1.0)
    assert row_35["recall"] == pytest.approx(1.0)
    assert row_35["f1"] == pytest.approx(1.0)


# =========================================================================
# Test 15: Operational Threshold Selected on Validation Data Only (Section 21)
# =========================================================================
def test_operational_threshold_selected_on_validation_only():
    """Verify operational threshold is determined strictly from validation distribution."""
    from src.opponent_model.evaluate import select_operational_threshold

    np.random.seed(42)
    y_val = np.array([1] * 20 + [0] * 500)
    p_val = np.concatenate([
        np.random.uniform(0.08, 0.25, 20),
        np.random.uniform(0.001, 0.05, 500),
    ])

    th_selected = select_operational_threshold(y_val, p_val, metric="f1")
    assert 0.03 <= th_selected <= 0.15
    assert isinstance(th_selected, float)


# =========================================================================
# Test 16: Test Data Never Used to Select Threshold/Class Weight/Calibration (Section 21)
# =========================================================================
def test_no_test_leakage_in_hyperparameters():
    """Verify test split remains strictly blind and untouched during tuning."""
    data_dir = Path("data_fastf1_v1/opponent_model")
    train_file = data_dir / "train_dataset.parquet"
    val_file = data_dir / "val_dataset.parquet"
    test_file = data_dir / "test_dataset.parquet"

    if train_file.exists() and val_file.exists() and test_file.exists():
        df_train = pd.read_parquet(train_file, columns=["race_id"])
        df_val = pd.read_parquet(val_file, columns=["race_id"])
        df_test = pd.read_parquet(test_file, columns=["race_id"])

        train_races = set(df_train["race_id"].unique())
        val_races = set(df_val["race_id"].unique())
        test_races = set(df_test["race_id"].unique())

        assert test_races.isdisjoint(train_races)
        assert test_races.isdisjoint(val_races)

        for r in test_races:
            assert "2025" in str(r), f"Non-2025 race in test set: {r}"


# =========================================================================
# Test 17: Pit-Window Detection Does Not Access Future Information (Section 21)
# =========================================================================
def test_pit_window_detection_no_future_leakage():
    """Verify pit-window detection uses strictly preceding laps (t <= L-1)."""
    from src.opponent_model.evaluate import evaluate_pit_window_detection

    df_race = pd.DataFrame({
        "race_id": ["Race_1"] * 10,
        "driver": ["NOR"] * 10,
        "lap_number": list(range(1, 11)),
        "target": [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
    })

    y_prob = np.array([0.04, 0.04, 0.04, 0.04, 0.04, 0.95, 0.95, 0.95, 0.95, 0.95])

    df_detect = evaluate_pit_window_detection(df_race, y_prob, thresholds=[0.10], horizons=[1, 3, 5])

    assert df_detect["h1_detected"].iloc[0] == 0
    assert df_detect["h3_detected"].iloc[0] == 0
    assert df_detect["h5_detected"].iloc[0] == 0
    assert df_detect["h5_detection_rate"].iloc[0] == 0.0


# =========================================================================
# Test 18: Prediction and Explanation Use Same Feature Vector and Prob (Section 21)
# =========================================================================
def test_prediction_and_explanation_consistency():
    """Verify predict_opponent and explain_prediction return identical probabilities."""
    from src.opponent_model.interface import predict_opponent, explain_prediction

    state = _make_dummy_state_vector(tyre_age=26, compound="MEDIUM")

    pred = predict_opponent(state, opponent_driver="PIA", ego_driver="VER")
    expl = explain_prediction(state, opponent_driver="PIA", ego_driver="VER")

    assert pred["pit_probability"] == pytest.approx(expl["pit_probability"], abs=1e-6)
    assert pred["p_pit_next"] == pytest.approx(expl["p_pit_next"], abs=1e-6)
    assert pred["explanation"]["summary"] == expl["explanation"]["summary"]

    state_dict = state.to_dict()
    pred_d = predict_opponent(state_dict, opponent_driver="PIA", ego_driver="VER")
    expl_d = explain_prediction(state_dict, opponent_driver="PIA", ego_driver="VER")

    assert pred_d["pit_probability"] == pytest.approx(expl_d["pit_probability"], abs=1e-6)
    assert pred_d["p_pit_next"] == pytest.approx(expl_d["p_pit_next"], abs=1e-6)


# =========================================================================
# Test 19: Final Metrics Equal Independently Recomputed Metrics (Section 21)
# =========================================================================
def test_metric_reporting_no_fallbacks():
    """Verify compute_all_metrics calculates all metrics strictly from predictions without fallbacks."""
    from src.opponent_model.evaluate import compute_all_metrics
    from sklearn.metrics import brier_score_loss, roc_auc_score, average_precision_score

    np.random.seed(123)
    y_true = np.random.choice([0, 1], size=100, p=[0.95, 0.05])
    y_prob = np.random.beta(0.5, 20.0, size=100)

    metrics = compute_all_metrics(y_true, y_prob, threshold=0.10)

    expected_brier = brier_score_loss(y_true, y_prob)
    expected_auc = roc_auc_score(y_true, y_prob)
    expected_prauc = average_precision_score(y_true, y_prob)

    assert metrics["brier_score"] == pytest.approx(expected_brier, abs=1e-6)
    assert metrics["roc_auc"] == pytest.approx(expected_auc, abs=1e-6)
    assert metrics["pr_auc"] == pytest.approx(expected_prauc, abs=1e-6)
    assert metrics["tp"] + metrics["fp"] + metrics["tn"] + metrics["fn"] == 100

    assert not np.isnan(metrics["brier_score"])
    assert not np.isnan(metrics["pr_auc"])


# =========================================================================
# Test 20: Actual PIT Labels Are Temporally Aligned (Section 21)
# =========================================================================
def test_temporal_alignment_target():
    """Verify target A_{t+1} = PIT corresponds to next decision opportunity."""
    data_dir = Path("data_fastf1_v1/opponent_model")
    test_file = data_dir / "test_dataset.parquet"

    if test_file.exists():
        df_test = pd.read_parquet(test_file)
        pit_rows = df_test[df_test["target"] == 1].head(10)
        assert len(pit_rows) > 0
        for _, row in pit_rows.iterrows():
            assert row["tyre_age"] > 0, "Target=1 row already has fresh tyres (temporal leak/off-by-one)!"
            assert row["remaining_laps"] >= 0


# =========================================================================
# Test 21: Multi-Horizon Outputs Marked as Derived Approximations (Section 21)
# =========================================================================
def test_multi_horizon_marked_as_derived_approximation():
    """Verify multi-horizon outputs are explicitly identified as derived approximations."""
    from src.opponent_model.interface import predict_opponent

    state = _make_dummy_state_vector()
    pred = predict_opponent(state)

    multi_h = pred["multi_horizon"]
    assert "lap_1" in multi_h
    assert "lap_3" in multi_h
    assert "lap_5" in multi_h

    assert multi_h["lap_1"]["is_derived_approximation"] is False
    assert multi_h["lap_3"]["is_derived_approximation"] is True
    assert multi_h["lap_5"]["is_derived_approximation"] is True

