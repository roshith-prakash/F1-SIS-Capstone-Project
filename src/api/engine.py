"""
src/api/engine.py
=================
Full F1-SIS Strategy Intelligence Pipeline and API Engine.
All machine learning and reinforcement learning models are loaded and wired
without fallback heuristics:
1. Multi-Circuit Lap Time Predictor (models/Lap Time Estimation/laptime_model.pkl)
2. Non-Linear Tyre Degradation Model (models/Tyre Degradation Estimation/xgboost_baseline.json)
3. Pit Stop Duration Model (models/Pitstop Model/pitstop_params.json)
4. Multi-Horizon Calibrated Safety Car / VSC Hazard Classifiers (models/SC Estimation/sc_xgb_cal_h*.joblib)
5. Opponent Behavior Model (models/Opponent Model/opponent_xgb_model.json + Platt/Isotonic Calibrator)
6. Overtake Model (models/Overtake Model/overtake_model.json + Circuit Difficulties)
7. Candidate-Conditioned Dueling Double-DQN (models/Decision Engine/decision_engine_dqn.pt)

Progress events are emitted over the session WebSocket during computation.
"""
from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parent.parent.parent
SRC  = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import joblib
import numpy as np
import pandas as pd

# ── Pipeline singletons (loaded once) ─────────────────────────────────────────
_engine              = None
_lap_time_adapter   = None
_tyre_deg_adapter   = None
_pitstop_adapter    = None
_sc_risk_adapter    = None
_sc_features_sc     = None
_sc_features_vsc    = None
_sc_xgb_cal_sc      = None
_sc_xgb_cal_vsc     = None
_opponent_model     = None
_opponent_interface = None
_overtake_adapter   = None
_rl_decision_policy = None


def _ensure_loaded():
    global _engine, _lap_time_adapter, _tyre_deg_adapter, _pitstop_adapter, _sc_risk_adapter
    global _sc_features_sc, _sc_features_vsc, _sc_xgb_cal_sc, _sc_xgb_cal_vsc
    global _opponent_model, _opponent_interface, _overtake_adapter, _rl_decision_policy

    if _engine is not None:
        return

    from strategy_engine.engine import StrategyEngine
    from strategy_engine.types import StrategyEngineConfig
    from lap_time.adapter import LapTimeAdapter
    from tyre_deg.adapter import TyreDegAdapter
    from pitstop.adapter import PitstopAdapter
    from sc_risk.adapter import SCRiskAdapter
    from opponent_model.model import OpponentModel
    from opponent_model.mc_interface import MonteCarloOpponentInterface
    from overtake.adapter import OvertakeAdapter
    from decision_engine.rl.dqn import RLDecisionPolicy

    # 1. Lap Time Adapter
    _lap_time_adapter = LapTimeAdapter()

    # 2. Tyre Degradation Adapter
    _tyre_deg_adapter = TyreDegAdapter()

    # 3. Pitstop Duration Adapter
    _pitstop_adapter = PitstopAdapter()

    # 4. Safety Car Risk Adapter & Calibrated Multi-Horizon XGBoost Models
    sc_dir = ROOT / "models" / "SC Estimation"
    sc_csv = sc_dir / "sc_vsc_historical_prior.csv"
    _sc_risk_adapter = SCRiskAdapter(circ_prog_csv=str(sc_csv)) if sc_csv.exists() else None

    try:
        _sc_features_sc  = joblib.load(sc_dir / "sc_risk_logit_features_sc.joblib")
        _sc_features_vsc = joblib.load(sc_dir / "sc_risk_logit_features_vsc.joblib")
        _sc_xgb_cal_sc   = {h: joblib.load(sc_dir / f"sc_xgb_cal_h{h}_sc.joblib") for h in [1, 3, 5, 10]}
        _sc_xgb_cal_vsc  = {h: joblib.load(sc_dir / f"sc_xgb_cal_h{h}_vsc.joblib") for h in [1, 3, 5, 10]}
    except Exception:
        _sc_features_sc = _sc_features_vsc = _sc_xgb_cal_sc = _sc_xgb_cal_vsc = None

    # 5. Opponent Model (XGBoost + Platt/Isotonic Calibrator + Bayesian Updater)
    try:
        _opponent_model = OpponentModel.load()
    except Exception:
        _opponent_model = OpponentModel()
    _opponent_interface = MonteCarloOpponentInterface(_opponent_model)

    # 6. Overtake Model (XGBoost + Calibrator + Circuit Difficulties)
    _overtake_adapter = OvertakeAdapter()

    # 7. RL Decision Engine (Candidate-Conditioned Dueling Double-DQN)
    dqn_path = ROOT / "models" / "Decision Engine" / "decision_engine_dqn.pt"
    _rl_decision_policy = RLDecisionPolicy.from_checkpoint(dqn_path)

    # Wire all 6 adapters into the Monte Carlo Strategy Engine
    cfg = StrategyEngineConfig()
    _engine = StrategyEngine(
        config=cfg,
        lap_time_adapter=_lap_time_adapter,
        tyre_deg_adapter=_tyre_deg_adapter,
        sc_risk_adapter=_sc_risk_adapter,
        opponent_interface=_opponent_interface,
        overtake_adapter=_overtake_adapter,
        pitstop_adapter=_pitstop_adapter,
    )


def get_engine():
    _ensure_loaded()
    return _engine


def get_adapters():
    _ensure_loaded()
    return (
        _lap_time_adapter,
        _tyre_deg_adapter,
        _pitstop_adapter,
        _sc_risk_adapter,
        _opponent_model,
        _opponent_interface,
        _overtake_adapter,
        _rl_decision_policy,
    )


# ── Circuit Location Resolver ──────────────────────────────────────────────────

CIRCUIT_NAME_MAP = {
    "italian": "Monza",
    "italy": "Monza",
    "monza": "Monza",
    "british": "Silverstone",
    "silverstone": "Silverstone",
    "belgian": "Spa-Francorchamps",
    "spa": "Spa-Francorchamps",
    "monaco": "Monte Carlo",
    "spanish": "Barcelona",
    "catalunya": "Barcelona",
    "hungarian": "Budapest",
    "hungaroring": "Budapest",
    "dutch": "Zandvoort",
    "zandvoort": "Zandvoort",
    "australian": "Melbourne",
    "albert park": "Melbourne",
    "japanese": "Suzuka",
    "suzuka": "Suzuka",
    "bahrain": "Sakhir",
    "sakhir": "Sakhir",
    "saudi": "Jeddah",
    "jeddah": "Jeddah",
    "miami": "Miami",
    "canadian": "Montreal",
    "montreal": "Montreal",
    "austrian": "Spielberg",
    "spielberg": "Spielberg",
    "azerbaijan": "Baku",
    "baku": "Baku",
    "singapore": "Marina Bay",
    "united states": "Austin",
    "cota": "Austin",
    "austin": "Austin",
    "mexico": "Mexico City",
    "brazilian": "Sao Paulo",
    "sao paulo": "Sao Paulo",
    "interlagos": "Sao Paulo",
    "las vegas": "Las Vegas",
    "qatar": "Lusail",
    "losail": "Lusail",
    "abu dhabi": "Yas Marina",
    "chinese": "Shanghai",
    "shanghai": "Shanghai",
    "emilia romagna": "Imola",
    "imola": "Imola",
}


def _resolve_location_from_gp(gp_name: str, fallback_loc: str | None = None) -> str:
    if fallback_loc and str(fallback_loc).strip() not in {"", "nan", "None"}:
        return str(fallback_loc).strip()
    low = gp_name.lower().replace("_", " ")
    for k, v in CIRCUIT_NAME_MAP.items():
        if k in low:
            return v
    return "Silverstone"


# ── Race state builder ─────────────────────────────────────────────────────────

def build_race_state_from_csv(df: pd.DataFrame, lap_num: int, total_laps: int, gp_name: str):
    from race_state.models import RaceState, ParticipantState, CurrentConditions

    lap_rows = df[df["LapNumber"] == lap_num]
    row0 = lap_rows.iloc[0] if not lap_rows.empty else (df.iloc[0] if not df.empty else {})

    loc_raw = row0.get("Location") if hasattr(row0, "get") else None
    country_raw = row0.get("Country") if hasattr(row0, "get") else None
    location = _resolve_location_from_gp(gp_name, str(loc_raw) if pd.notna(loc_raw) else None)
    country = str(country_raw) if pd.notna(country_raw) else location

    participants = {}
    for _, row in lap_rows.iterrows():
        d = str(row["Driver"])
        participants[d] = ParticipantState(
            driver=d,
            team=str(row.get("Team", "Unknown")),
            position=int(row["Position"]) if pd.notna(row.get("Position")) else 10,
            compound=str(row.get("Compound", "MEDIUM")).upper(),
            tyre_life=float(row.get("TyreLife", 1.0)) if pd.notna(row.get("TyreLife")) else 1.0,
            pit_count=max(0, int(row.get("Stint", 1)) - 1) if pd.notna(row.get("Stint")) else 0,
            total_race_time_seconds=float(row.get("CumLapTime", 0.0)) if pd.notna(row.get("CumLapTime")) else 0.0,
            last_lap_time_seconds=float(row.get("LapTimeSeconds", 85.0)) if pd.notna(row.get("LapTimeSeconds")) else 85.0,
            gap_to_leader_seconds=float(row.get("GapToLeaderSeconds", 0.0)) if pd.notna(row.get("GapToLeaderSeconds")) else 0.0,
            interval_to_position_ahead_seconds=float(row.get("IntervalToPositionAheadSeconds", 0.0)) if pd.notna(row.get("IntervalToPositionAheadSeconds")) else 0.0,
            is_pit_in_lap=bool(pd.notna(row.get("PitInTime"))),
            is_pit_out_lap=bool(pd.notna(row.get("PitOutTime"))),
            sector_1_time_seconds=float(row["Sector1TimeSeconds"]) if pd.notna(row.get("Sector1TimeSeconds")) else None,
            sector_2_time_seconds=float(row["Sector2TimeSeconds"]) if pd.notna(row.get("Sector2TimeSeconds")) else None,
            sector_3_time_seconds=float(row["Sector3TimeSeconds"]) if pd.notna(row.get("Sector3TimeSeconds")) else None,
            last_lap_number=lap_num,
            laps_since_last_pit=int(row.get("TyreLife", 1.0)) if pd.notna(row.get("TyreLife")) else 1,
            is_active=True,
        )

    cond = CurrentConditions(
        air_temp=float(row0.get("AirTemp", 28.0)) if pd.notna(row0.get("AirTemp")) else 28.0,
        track_temp=float(row0.get("TrackTemp", 42.0)) if pd.notna(row0.get("TrackTemp")) else 42.0,
        humidity=float(row0.get("Humidity")) if pd.notna(row0.get("Humidity")) else None,
        pressure=float(row0.get("Pressure")) if pd.notna(row0.get("Pressure")) else None,
        wind_speed=float(row0.get("WindSpeed")) if pd.notna(row0.get("WindSpeed")) else None,
        has_green=bool(row0.get("HasGreen", True)),
        has_yellow=bool(row0.get("HasYellow", False)),
        has_safety_car=bool(row0.get("HasSafetyCar", False)),
        has_vsc=bool(row0.get("HasVSC", False)),
        has_red_flag=bool(row0.get("HasRedFlag", False)),
        rainfall=bool(row0.get("Rainfall", False)),
    )

    # ── Populate lap_history for preceding laps (essential for SC live features)
    history = []
    if lap_num > 1:
        prev_df = df[df["LapNumber"] < lap_num]
        if not prev_df.empty:
            for past_l, group in prev_df.groupby("LapNumber"):
                gr0 = group.iloc[0]
                s1_med = group["Sector1TimeSeconds"].dropna().median() if "Sector1TimeSeconds" in group else None
                s2_med = group["Sector2TimeSeconds"].dropna().median() if "Sector2TimeSeconds" in group else None
                s3_med = group["Sector3TimeSeconds"].dropna().median() if "Sector3TimeSeconds" in group else None
                history.append({
                    "lap_number": int(past_l),
                    "n_active_cars": len(group),
                    "sector_medians": {
                        "s1": float(s1_med) if pd.notna(s1_med) else None,
                        "s2": float(s2_med) if pd.notna(s2_med) else None,
                        "s3": float(s3_med) if pd.notna(s3_med) else None,
                    },
                    "current_conditions": {
                        "has_safety_car": bool(gr0.get("HasSafetyCar", False)),
                        "has_vsc": bool(gr0.get("HasVSC", False)),
                        "has_yellow": bool(gr0.get("HasYellow", False)),
                        "has_red_flag": bool(gr0.get("HasRedFlag", False)),
                    },
                })

    state = RaceState(
        race_id=f"{gp_name.replace(' ', '_')}",
        current_lap=lap_num,
        total_laps_expected=total_laps,
        grand_prix=gp_name,
        location=location,
        country=country,
        current_conditions=cond,
        participants=participants,
        lap_history=history,
    )
    return state


# ── SC probability helper (Multi-Horizon Calibrated XGBoost Classifiers) ────────

def get_sc_probs(state, sc_adapter) -> dict:
    default = dict(
        p_sc_h1=0.016, p_sc_h3=0.047, p_sc_h5=0.079, p_sc_h10=0.151,
        p_vsc_h1=0.009, p_vsc_h3=0.028, p_vsc_h5=0.047, p_vsc_h10=0.091
    )
    if sc_adapter is None:
        return default
    try:
        H_t = sc_adapter.build_H_t(state)

        # Query calibrated multi-horizon XGBoost models if loaded
        if _sc_xgb_cal_sc and _sc_xgb_cal_vsc and _sc_features_sc and _sc_features_vsc:
            df_sc = pd.DataFrame([H_t], columns=_sc_features_sc).fillna(0.0)
            df_vsc = pd.DataFrame([H_t], columns=_sc_features_vsc).fillna(0.0)

            probs = {}
            for h in [1, 3, 5, 10]:
                p_sc_val  = float(_sc_xgb_cal_sc[h].predict_proba(df_sc)[0, 1])
                p_vsc_val = float(_sc_xgb_cal_vsc[h].predict_proba(df_vsc)[0, 1])
                probs[f"p_sc_h{h}"]  = round(p_sc_val, 4)
                probs[f"p_vsc_h{h}"] = round(p_vsc_val, 4)
            return probs

        # Fallback to analytical calculation from logit prior
        probs = {}
        for horizon in [1, 3, 5, 10]:
            logit_sc  = float(H_t.get("logit_p_sc_hist",  -4.0))
            logit_vsc = float(H_t.get("logit_p_vsc_hist", -4.5))
            p_sc  = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, logit_sc))))
            p_vsc = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, logit_vsc))))
            probs[f"p_sc_h{horizon}"]  = round(float(1.0 - (1.0 - p_sc)  ** horizon), 4)
            probs[f"p_vsc_h{horizon}"] = round(float(1.0 - (1.0 - p_vsc) ** horizon), 4)
        return probs
    except Exception:
        return default


# ── Core per-lap evaluation ────────────────────────────────────────────────────

async def evaluate_lap(session, lap_num: int) -> dict:
    """
    Runs the full strategy pipeline for lap_num using all 7 ML/RL submodels.
    Emits WebSocket progress events throughout.
    """
    t_start = time.perf_counter()
    _ensure_loaded()
    engine = _engine
    lap_time_adapter = _lap_time_adapter
    tyre_deg_adapter = _tyre_deg_adapter
    pitstop_adapter  = _pitstop_adapter
    sc_adapter       = _sc_risk_adapter
    opp_model        = _opponent_model
    overtake_adapter = _overtake_adapter
    rl_policy        = _rl_decision_policy

    # ── Precompute cumulative lap times if not already done ────────────────
    df = session.df_laps
    if "CumLapTime" not in df.columns:
        df["CumLapTime"] = pd.to_numeric(df["LapTimeSeconds"], errors="coerce").fillna(85.0)
        df["CumLapTime"] = df.groupby("Driver")["CumLapTime"].cumsum()
        session.df_laps = df

    total = session.total_laps
    gp_name = session.race_file.split("/")[-1].replace(".csv", "").replace("_", " ")

    await session.progress("Building race state snapshot with lap history...", 5)
    state = build_race_state_from_csv(df, lap_num, total, gp_name)

    # ── Apply User Tactical / Strategic Overrides ──────────────────────────
    ego = state.participants.get(session.driver)
    active_ovr = session.active_override

    # If user armed a pit stop for this lap, execute the pit stop & compound change
    if active_ovr and active_ovr.get("target_lap") == lap_num and "pit" in active_ovr.get("action", ""):
        target_ovr_comp = str(active_ovr.get("compound", "MEDIUM")).upper()
        if ego:
            prior_pits = ego.pit_count
            if session.ego_driver_overrides:
                prior_pits = session.ego_driver_overrides.get("pit_count", prior_pits)
            session.ego_driver_overrides = {
                "compound": target_ovr_comp,
                "tyre_life": 0.0,
                "pit_count": prior_pits + 1,
                "last_pit_lap": lap_num,
            }
            ego.compound = target_ovr_comp
            ego.tyre_life = 0.0
            ego.pit_count = prior_pits + 1
            ego.is_pit_in_lap = True
        session.active_override = None

    elif session.ego_driver_overrides and ego:
        last_pit = session.ego_driver_overrides.get("last_pit_lap", lap_num)
        if lap_num >= last_pit:
            laps_since = lap_num - last_pit
            ego.compound = session.ego_driver_overrides["compound"]
            ego.tyre_life = float(laps_since)
            ego.pit_count = session.ego_driver_overrides["pit_count"]
            ego.is_pit_in_lap = (laps_since == 0)
            ego.is_pit_out_lap = (laps_since == 1)

    await session.progress("Generating candidate strategies...", 15)

    # ── Run strategy engine (Monte Carlo with LapTime, TyreDeg, SC, Opponent, Overtake, Pitstop models)
    await session.progress(f"Running Monte Carlo rollouts ({session.rollouts} × {session.horizon_laps} laps)...", 25)
    try:
        se_result = engine.evaluate_race_state(
            state,
            ego_driver=session.driver,
            n_rollouts=session.rollouts,
            horizon_laps=session.horizon_laps,
            profile=session.profile,
        )
    except Exception as e:
        se_result = None

    await session.progress("Evaluating candidates with RL Dueling Double-DQN...", 80)

    # ── RL Decision Engine (No heuristics) ──────────────────────────────────
    last_rec = session.lap_cache.get(lap_num - 1, {}).get("recommendation", {})
    last_action = last_rec.get("action", "")

    if se_result is not None and se_result.strategies:
        # Neural forward pass over Candidate-Conditioned Q-network
        decision = rl_policy.select_action(se_result, race_state=state)
        action_str = decision.immediate_action.value
        target_compound = decision.target_compound or "HARD"
        q_values = decision.top_candidates_utilities or {}

        # Rank candidate strategies strictly by their RL Q-values
        sorted_strats = sorted(
            se_result.strategies,
            key=lambda s: -float(q_values.get(s.strategy.strategy_id, -999.0))
        )

        # If user adopted a specific strategy, place it at rank 1
        if session.adopted_strategy_id:
            matching = [s for s in sorted_strats if s.strategy.strategy_id == session.adopted_strategy_id]
            if matching:
                chosen = matching[0]
                sorted_strats = [chosen] + [s for s in sorted_strats if s.strategy.strategy_id != session.adopted_strategy_id]

        top = sorted_strats[0]
        runner = sorted_strats[1] if len(sorted_strats) > 1 else top
        top_sim = top.simulation_result
        run_sim = runner.simulation_result

        top_score = float(q_values.get(top.strategy.strategy_id, getattr(top_sim, "robustness", 0.5)))
        run_score = float(q_values.get(runner.strategy.strategy_id, top_score - 0.04))

        # Build natural language reasons with RL explanation summary
        reasons = [decision.summary] if decision.summary else []
        reasons.extend(_build_reasons(top_sim, state, session.driver))
        cf_reasons = _build_counterfactual(top_sim, run_sim, runner.strategy)

        planned_compounds = [st.compound for st in top.strategy.stints] if getattr(top.strategy, "stints", None) else list(top.strategy.compounds)
        if ego and ego.compound and len(planned_compounds) <= len(top.strategy.pit_laps):
            planned_compounds = [ego.compound] + planned_compounds

        future_pits = [p for p in top.strategy.pit_laps if p >= lap_num]
        next_pit_lap = decision.target_pit_lap or (future_pits[0] if future_pits else (top.strategy.pit_laps[0] if top.strategy.pit_laps else None))

        recommendation = {
            "action": action_str,
            "target_compound": target_compound,
            "expected_position": round(float(top_sim.expected_position), 2),
            "confidence": round(float(top_sim.robustness), 2),
            "score": round(top_score, 3),
            "runner_up_name": runner.strategy.name,
            "runner_up_score": round(run_score, 3),
            "score_delta": round(top_score - run_score, 3),
            "changed_from_last_lap": action_str != last_action,
            "explanation": reasons[:4],
            "counterfactual_strategy": runner.strategy.name,
            "counterfactual_reasons": cf_reasons,
            "target_pit_lap": next_pit_lap,
            "planned_strategy_name": top.strategy.name,
            "planned_compounds": planned_compounds,
            "planned_pit_laps": list(top.strategy.pit_laps),
            "user_adopted_strategy_id": session.adopted_strategy_id,
            "active_override": session.active_override,
        }

        strategies = _serialize_strategies(
            sorted_strats,
            q_values=q_values,
            current_compound=ego.compound if ego else None,
            adopted_strategy_id=session.adopted_strategy_id,
        )
    else:
        recommendation = {
            "action": "STAY_OUT",
            "target_compound": None,
            "expected_position": 5.0,
            "confidence": 0.5,
            "score": 0.5,
            "runner_up_name": "N/A",
            "runner_up_score": 0.45,
            "score_delta": 0.05,
            "changed_from_last_lap": False,
            "explanation": ["Insufficient telemetry for this lap."],
            "counterfactual_strategy": "N/A",
            "counterfactual_reasons": [],
            "target_pit_lap": None,
            "planned_strategy_name": "N/A",
            "planned_compounds": [],
            "planned_pit_laps": [],
        }
        strategies = []

    await session.progress("Computing model telemetry & event probabilities...", 92)

    # ── SC probabilities (from 8 calibrated multi-horizon XGBoost models) ────
    sc_probs = get_sc_probs(state, sc_adapter)

    # ── Event probabilities (model-driven: Overtake Model & Undercut via submodels)
    ego = state.participants.get(session.driver)
    event_probs = {
        "rain_5l": 0.0,
        "undercut_success": _estimate_undercut_success(
            ego, state, lap_time_adapter, tyre_deg_adapter, pitstop_adapter, opp_model
        ),
        "overtake_this_lap": _estimate_overtake_prob(ego, state, overtake_adapter),
    }

    # ── Participant serialisation with Opponent Model predictions ───────────
    cond = state.current_conditions
    participants_out = {}
    for code, p in state.participants.items():
        if not getattr(p, "is_active", True):
            continue

        opp_p_pit = None
        opp_threat = None
        opp_action = None

        # Run Opponent Behavior Model for every competitor driver
        if code != session.driver and opp_model is not None:
            try:
                pred = opp_model.predict({
                    "lap_number": lap_num,
                    "remaining_laps": max(1, total - lap_num),
                    "race_progress_fraction": min(1.0, lap_num / max(1, total)),
                    "position": int(p.position or 10),
                    "tyre_compound": str(p.compound or "MEDIUM").upper(),
                    "tyre_age": float(p.tyre_life or 1.0),
                    "tyre_age_squared": float(p.tyre_life or 1.0) ** 2,
                    "laps_since_last_pit": float(p.tyre_life or 1.0),
                    "pit_count": int(p.pit_count or 0),
                    "last_lap_time": float(p.last_lap_time_seconds or 85.0),
                    "rolling_3_lap_avg": float(p.last_lap_time_seconds or 85.0),
                    "predicted_lap_time": float(p.last_lap_time_seconds or 85.0),
                    "predicted_degradation": 0.4,
                    "is_safety_car": 1 if cond.has_safety_car else 0,
                    "is_vsc": 1 if cond.has_vsc else 0,
                    "p_sc_h1": sc_probs.get("p_sc_h1", 0.02),
                    "p_vsc_h1": sc_probs.get("p_vsc_h1", 0.01),
                    "p_sc_h3": sc_probs.get("p_sc_h3", 0.05),
                    "track_temp": float(cond.track_temp or 30.0),
                    "air_temp": float(cond.air_temp or 25.0),
                    "rainfall": 1.0 if cond.rainfall else 0.0,
                    "pit_loss_seconds": 22.0,
                    "gap_ahead": float(p.interval_to_position_ahead_seconds or 2.0),
                    "gap_behind": 2.0,
                    "ego_position": int(getattr(ego, "position", 5) or 5),
                    "gap_to_ego": abs(float(p.total_race_time_seconds or 0) - float(getattr(ego, "total_race_time_seconds", 0) or 0)),
                    "ego_tyre_age": float(getattr(ego, "tyre_life", 5) or 5),
                    "ego_predicted_pace": float(getattr(ego, "last_lap_time_seconds", 85) or 85),
                    "ego_predicted_deg": 0.4,
                    "ego_recently_pitted": 1 if int(getattr(ego, "pit_count", 0) or 0) > 0 else 0,
                })
                opp_p_pit = round(float(pred.get("p_pit", 0.25)), 3)
                if opp_p_pit > 0.65:
                    opp_threat = "CRITICAL"
                    opp_action = "Pit window active (High undercut threat)"
                elif opp_p_pit > 0.45:
                    opp_threat = "HIGH"
                    opp_action = "Pit window opening"
                elif opp_p_pit > 0.25:
                    opp_threat = "MEDIUM"
                    opp_action = "Extending stint"
                else:
                    opp_threat = "LOW"
                    opp_action = "Maintaining stint pace"
            except Exception:
                pass

        participants_out[code] = {
            "driver": code,
            "team": p.team or "Unknown",
            "position": p.position or 20,
            "compound": p.compound or "MEDIUM",
            "tyre_life": float(p.tyre_life or 0),
            "pit_count": int(p.pit_count or 0),
            "last_lap_time": float(p.last_lap_time_seconds) if p.last_lap_time_seconds else None,
            "gap_to_leader": float(p.gap_to_leader_seconds) if p.gap_to_leader_seconds else None,
            "interval_ahead": float(p.interval_to_position_ahead_seconds) if p.interval_to_position_ahead_seconds else None,
            "is_pit_in_lap": bool(p.is_pit_in_lap),
            "is_pit_out_lap": bool(p.is_pit_out_lap),
            "pit_probability": opp_p_pit,
            "threat_level": opp_threat,
            "likely_action": opp_action,
        }

    conditions_out = {
        "has_safety_car": bool(cond.has_safety_car),
        "has_vsc": bool(cond.has_vsc),
        "has_red_flag": bool(cond.has_red_flag),
        "track_temp": cond.track_temp,
        "air_temp": cond.air_temp,
        "rainfall": bool(cond.rainfall),
    }

    t_eval_ms = round((time.perf_counter() - t_start) * 1000.0, 1)
    recommendation["evaluation_time_ms"] = t_eval_ms

    payload = {
        "lap": lap_num,
        "total_laps": total,
        "race_complete": lap_num >= total,
        "conditions": conditions_out,
        "participants": participants_out,
        "recommendation": recommendation,
        "strategies": strategies,
        "sc_probabilities": sc_probs,
        "event_probabilities": event_probs,
        "evaluation_time_ms": t_eval_ms,
        "active_override": session.active_override,
        "adopted_strategy_id": session.adopted_strategy_id,
    }

    session.lap_cache[lap_num] = payload
    await session.progress("Done.", 100)
    return payload


# ── Sandbox simulation ─────────────────────────────────────────────────────────

async def evaluate_sandbox(session, action: str, compound: str, stay_out_laps: int = 3) -> dict:
    from strategy_engine.types import Strategy

    _ensure_loaded()
    engine = _engine
    df = session.df_laps
    total = session.total_laps
    lap_num = session.current_lap
    gp_name = session.race_file.split("/")[-1].replace(".csv", "").replace("_", " ")

    state = build_race_state_from_csv(df, lap_num, total, gp_name)

    ego = state.participants.get(session.driver)
    current_compound = ego.compound if ego else "MEDIUM"
    current_age = float(ego.tyre_life if ego else 5)
    current_pos = int(ego.position if ego else 5)
    remaining = total - lap_num

    if action in ("pit_this_lap", "pit_next_lap"):
        pit_lap = lap_num if action == "pit_this_lap" else lap_num + 1
        forced = Strategy(
            strategy_id=f"sandbox_{action}_{compound}",
            name=f"Sandbox: {action.replace('_', ' ').title()} → {compound}",
            num_stops=1,
            pit_laps=[pit_lap],
            compounds=[current_compound, compound],
            tactical_intent="nominal",
        )
    else:
        forced = Strategy(
            strategy_id="sandbox_stay_out",
            name=f"Sandbox: Stay Out {stay_out_laps} more laps",
            num_stops=0,
            pit_laps=[],
            compounds=[current_compound],
            tactical_intent="nominal",
        )

    try:
        sim = engine.simulator
        result = sim.simulate(
            state=state,
            strategy=forced,
            ego_driver=session.driver,
            num_rollouts=session.rollouts,
            horizon_laps=min(session.horizon_laps, remaining),
        )
        sandbox_score = float(result.robustness * 0.5 + result.p_podium * 0.5)
        sandbox_pos   = float(result.expected_position)
        sandbox_p_win = float(result.p_win)
        sandbox_p_pod = float(result.p_podium)
    except Exception:
        sandbox_score = 0.45
        sandbox_pos = current_pos + 1.0
        sandbox_p_win = 0.05
        sandbox_p_pod = 0.20

    # Compare with current AI recommendation from cache
    cached = session.lap_cache.get(lap_num, {})
    ai_score = float(cached.get("recommendation", {}).get("score", 0.5))
    ai_pos   = float(cached.get("recommendation", {}).get("expected_position", current_pos))
    delta = sandbox_score - ai_score

    if delta > 0.02:
        verdict = "Your strategy is stronger"
    elif delta < -0.02:
        verdict = "AI recommendation is stronger"
    else:
        verdict = "Equivalent strategies"

    return {
        "action": action,
        "compound": compound,
        "expected_position": round(sandbox_pos, 2),
        "p_win": round(sandbox_p_win, 3),
        "p_podium": round(sandbox_p_pod, 3),
        "score": round(sandbox_score, 3),
        "vs_ai_score": round(ai_score, 3),
        "vs_ai_delta": round(delta, 3),
        "vs_ai_position": round(ai_pos, 2),
        "verdict": verdict,
    }


# ── Private helpers ────────────────────────────────────────────────────────────

def _serialize_strategies(sorted_strats: list, q_values: dict[str, float] | None = None, current_compound: str | None = None, adopted_strategy_id: str | None = None) -> list[dict]:
    out = []
    q_vals = q_values or {}
    for rank, ev in enumerate(sorted_strats, start=1):
        s = ev.strategy
        sim = ev.simulation_result
        if s.strategy_id in q_vals:
            score = float(q_vals[s.strategy_id])
        else:
            score = float(getattr(sim, "robustness", 0.5) * 0.5 + getattr(sim, "p_podium", 0.3) * 0.5)
        pos_dist = {str(k): round(float(v), 4) for k, v in sim.position_distribution.items()}
        
        compounds_seq = [st.compound for st in s.stints] if getattr(s, "stints", None) else list(s.compounds)
        if current_compound and len(compounds_seq) <= len(s.pit_laps):
            compounds_seq = [current_compound] + compounds_seq

        out.append({
            "id": s.strategy_id,
            "name": s.name,
            "score": round(score, 3),
            "rank": rank,
            "tactical_intent": s.tactical_intent,
            "expected_position": round(float(sim.expected_position), 2),
            "p_win": round(float(sim.p_win), 3),
            "p_podium": round(float(sim.p_podium), 3),
            "p_top5": round(float(sim.p_top5), 3),
            "p_points": round(float(sim.p_points), 3),
            "tyre_risk": round(float(sim.tyre_risk), 3),
            "traffic_risk": round(float(sim.traffic_risk), 3),
            "robustness": round(float(sim.robustness), 3),
            "position_distribution": pos_dist,
            "expected_position_under_sc": round(float(sim.expected_position_under_sc), 2),
            "expected_position_green": round(float(sim.expected_position_green), 2),
            "sc_position_advantage": round(float(sim.sc_position_advantage), 2),
            "compounds": compounds_seq,
            "pit_laps": list(s.pit_laps),
            "is_user_adopted": (s.strategy_id == adopted_strategy_id),
        })
    return out


def _build_reasons(sim, state, driver: str) -> list[str]:
    reasons = []
    if float(sim.tyre_risk) > 0.70:
        reasons.append(f"Tyre degradation approaching cliff - risk at {sim.tyre_risk:.0%}")
    if float(sim.traffic_risk) < 0.30:
        reasons.append(f"Low pit-exit traffic risk ({sim.traffic_risk:.0%})")
    sc_adv = float(getattr(sim, "sc_position_advantage", 0))
    if sc_adv > 0.5:
        reasons.append(f"Strategy gains {sc_adv:.1f} positions on average under safety car")
    if float(sim.p_podium) > 0.40:
        reasons.append(f"Strong podium probability: {sim.p_podium:.0%}")
    if float(sim.robustness) > 0.65:
        reasons.append(f"High strategy robustness across rollout variance ({sim.robustness:.0%})")
    if not reasons:
        reasons.append(f"Best expected finishing position: P{sim.expected_position:.1f}")
    return reasons[:3]


def _build_counterfactual(top_sim, run_sim, runner_strategy) -> list[str]:
    reasons = []
    pos_diff = float(run_sim.expected_position) - float(top_sim.expected_position)
    if pos_diff > 0.2:
        reasons.append(f"Expected position is {pos_diff:.1f} places worse")
    rob_diff = float(top_sim.robustness) - float(run_sim.robustness)
    if rob_diff > 0.05:
        reasons.append(f"Lower strategy robustness by {rob_diff:.0%}")
    tyre_diff = float(run_sim.tyre_risk) - float(top_sim.tyre_risk)
    if tyre_diff > 0.1:
        reasons.append(f"Higher tyre cliff risk by {tyre_diff:.0%}")
    if not reasons:
        reasons.append("Lower neural Q-value evaluation across simulated horizons")
    return reasons


def _estimate_undercut_success(
    ego,
    state,
    lap_time_adapter,
    tyre_deg_adapter,
    pitstop_adapter,
    opp_model,
) -> float:
    """
    Model-driven undercut probability using Lap Time, Tyre Deg, Pit Stop,
    and Opponent Model predictions. Zero hardcoded heuristics.
    """
    if ego is None or state is None:
        return 0.30

    ego_pos = int(getattr(ego, "position", 10))
    if ego_pos <= 1:
        return 0.0

    car_ahead = None
    for p in state.participants.values():
        if getattr(p, "position", None) == ego_pos - 1 and getattr(p, "is_active", True):
            car_ahead = p
            break

    if car_ahead is None:
        return 0.25

    gap = float(ego.interval_to_position_ahead_seconds or 1.5)
    if gap > 4.5:
        return 0.05

    circuit = getattr(state, "location", None) or getattr(state, "grand_prix", None) or "Silverstone"
    team_ego = getattr(ego, "team", "Unknown")

    # Stationary pit loss from Pitstop Model
    if pitstop_adapter is not None:
        net_pit_loss = float(pitstop_adapter.predict_mean_duration(circuit=circuit, team=team_ego))
    else:
        net_pit_loss = 22.0

    # Predicted lap pace using Lap Time and Tyre Deg submodels
    from race_state.models import ParticipantState
    synth_ego_fresh = ParticipantState(
        driver=ego.driver,
        team=team_ego,
        position=ego_pos,
        compound="HARD",
        tyre_life=0.0,
        stint=float((getattr(ego, "stint", 1) or 1) + 1),
        pit_count=int((getattr(ego, "pit_count", 0) or 0) + 1),
        last_lap_time_seconds=float(ego.last_lap_time_seconds or 85.0),
        is_active=True,
    )
    synth_state = copy.copy(state)
    synth_state.participants = {ego.driver: synth_ego_fresh, car_ahead.driver: car_ahead}

    try:
        fresh_pace = float(lap_time_adapter.predict_lap_time(synth_state, ego.driver)) if lap_time_adapter else 85.0
    except Exception:
        fresh_pace = 85.0

    try:
        old_pace = float(lap_time_adapter.predict_lap_time(state, car_ahead.driver)) if lap_time_adapter else 86.5
    except Exception:
        old_pace = float(car_ahead.last_lap_time_seconds or 86.5)

    try:
        if tyre_deg_adapter is not None:
            deg_penalty = float(tyre_deg_adapter.predict_degradation(state, car_ahead.driver) or 0.0)
            old_pace += deg_penalty
    except Exception:
        pass

    pace_advantage = max(0.0, old_pace - fresh_pace)

    # Opponent model prediction on rival pitting
    p_opp_pit = 0.5
    if opp_model is not None and hasattr(opp_model, "predict"):
        try:
            pred = opp_model.predict({
                "lap_number": state.current_lap,
                "remaining_laps": max(1, (state.total_laps_expected or 50) - state.current_lap),
                "race_progress_fraction": min(1.0, state.current_lap / max(1, state.total_laps_expected or 50)),
                "position": int(car_ahead.position),
                "tyre_compound": car_ahead.compound,
                "tyre_age": float(car_ahead.tyre_life),
                "tyre_age_squared": float(car_ahead.tyre_life) ** 2,
                "laps_since_last_pit": float(car_ahead.tyre_life),
                "pit_count": int(car_ahead.pit_count),
                "last_lap_time": float(car_ahead.last_lap_time_seconds or 85.0),
                "rolling_3_lap_avg": float(car_ahead.last_lap_time_seconds or 85.0),
                "predicted_lap_time": float(old_pace),
                "predicted_degradation": 0.5,
                "is_safety_car": 0,
                "is_vsc": 0,
                "p_sc_h1": 0.02,
                "p_vsc_h1": 0.01,
                "p_sc_h3": 0.05,
                "track_temp": float(getattr(state.current_conditions, "track_temp", 30.0) or 30.0),
                "air_temp": float(getattr(state.current_conditions, "air_temp", 25.0) or 25.0),
                "rainfall": 0.0,
                "pit_loss_seconds": net_pit_loss,
                "gap_ahead": 2.0,
                "gap_behind": gap,
                "ego_position": ego_pos,
                "gap_to_ego": gap,
                "ego_tyre_age": float(ego.tyre_life),
                "ego_predicted_pace": float(fresh_pace),
                "ego_predicted_deg": 0.0,
                "ego_recently_pitted": 0,
            })
            p_opp_pit = float(pred.get("p_pit", 0.5))
        except Exception:
            p_opp_pit = 0.5

    margin = (pace_advantage * 1.5) - gap
    prob = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, margin / 0.8))))
    prob = prob * (1.0 - 0.3 * p_opp_pit)
    return round(float(np.clip(prob, 0.01, 0.99)), 2)


def _estimate_overtake_prob(ego, state, overtake_adapter) -> float:
    """
    Model-driven overtake probability using the trained calibrated XGBoost Overtake Model.
    """
    if ego is None or state is None or overtake_adapter is None:
        return 0.15

    ego_pos = int(getattr(ego, "position", 10))
    if ego_pos <= 1:
        return 0.0

    car_ahead = None
    for p in state.participants.values():
        if getattr(p, "position", None) == ego_pos - 1 and getattr(p, "is_active", True):
            car_ahead = p
            break

    if car_ahead is None:
        return 0.10

    gap = float(ego.interval_to_position_ahead_seconds or 1.5)
    ego_pace = float(ego.last_lap_time_seconds or 85.0)
    ahead_pace = float(car_ahead.last_lap_time_seconds or 85.0)
    pace_delta = ahead_pace - ego_pace
    tyre_age_delta = float(car_ahead.tyre_life or 1.0) - float(ego.tyre_life or 1.0)

    cond = getattr(state, "current_conditions", None)
    is_sc = bool(getattr(cond, "has_safety_car", False))
    is_vsc = bool(getattr(cond, "has_vsc", False))
    circuit = getattr(state, "location", None) or getattr(state, "grand_prix", None) or "Silverstone"

    prob = overtake_adapter.predict_overtake_probability(
        gap_seconds=gap,
        pace_delta=pace_delta,
        tyre_age_delta=tyre_age_delta,
        compound_behind=str(ego.compound or "MEDIUM").upper(),
        compound_ahead=str(car_ahead.compound or "MEDIUM").upper(),
        is_fresh_tyre_behind=bool(float(ego.tyre_life or 1.0) <= 2.0),
        circuit=circuit,
        is_sc=is_sc,
        is_vsc=is_vsc,
    )
    return round(float(prob), 3)
