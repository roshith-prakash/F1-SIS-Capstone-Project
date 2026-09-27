"""
src/strategy_engine/simulator.py
================================
Component 2: Race Scenario Simulator (Monte Carlo Engine).

Simulates stochastic race evolution across N Monte Carlo rollouts over a
receding horizon H, conditioning on the candidate strategy and race state snapshot.

Integrates (all with graceful fallback if adapter not loaded):
- Lap Time Model (LapTimeAdapter): circuit-specific baseline dry pace
- Tyre Degradation Model (TyreDegAdapter): per-lap XGBoost degradation prediction
  accounting for TrackTemp, AirTemp, team, tyre age, rolling 3-lap trend
- Safety Car / VSC Risk Model (SCRiskAdapter): hazard-based caution sampling
- Opponent Model (OpponentModel via MonteCarloOpponentInterface): calibrated
  P(PIT|state_t) from the trained XGBoost + Bayesian updater, built from
  full opponent state vectors (tyre age, gap, pace trend, SC probability, etc.)

Strict Architectural Boundary:
- Simulates future race states forward from current lap t.
- Zero future-information leakage: never accesses ground-truth laps > t.
- Evaluates candidate plans and returns stochastic outcome distributions.
"""

from __future__ import annotations

import copy
import math
import random
from typing import Any, Optional
import numpy as np

try:
    from race_state.models import CurrentConditions, ParticipantState, RaceState, normalize_driver
except ImportError:
    from src.race_state.models import CurrentConditions, ParticipantState, RaceState, normalize_driver

try:
    from lap_time.adapter import LapTimeAdapter
except ImportError:
    LapTimeAdapter = None

try:
    from tyre_deg.adapter import TyreDegAdapter
except ImportError:
    TyreDegAdapter = None

try:
    from sc_risk.adapter import SCRiskAdapter
except ImportError:
    SCRiskAdapter = None

try:
    from opponent_model.mc_interface import MonteCarloOpponentInterface
    from opponent_model.model import OpponentModel
except ImportError:
    MonteCarloOpponentInterface = None
    OpponentModel = None

try:
    from overtake.adapter import OvertakeAdapter
except ImportError:
    try:
        from src.overtake.adapter import OvertakeAdapter
    except ImportError:
        OvertakeAdapter = None

from .types import RolloutOutcome, SimulationResult, Strategy, StrategyEngineConfig



class RaceScenarioSimulator:
    """
    Stochastic Monte Carlo Race Scenario Simulator.
    """

    def __init__(
        self,
        config: StrategyEngineConfig | None = None,
        lap_time_adapter: Any = None,
        tyre_deg_adapter: Any = None,
        sc_risk_adapter: Any = None,
        opponent_interface: Any = None,
        overtake_adapter: Any = None,
        default_rollouts: int = 150,
        random_seed: int | None = 42,
    ):
        self.config = config or StrategyEngineConfig()
        self.default_rollouts = default_rollouts or self.config.default_n_rollouts
        self.random_seed = random_seed

        # Upstream model integrations
        self.lap_time_adapter = lap_time_adapter
        self.tyre_deg_adapter = tyre_deg_adapter
        self.sc_risk_adapter = sc_risk_adapter
        self.opponent_interface = opponent_interface
        self.overtake_adapter = overtake_adapter

        # Compound baseline pace offsets relative to Medium
        self._compound_pace_offset = {
            "SOFT": -0.65,    # Soft is ~0.65s faster when new
            "MEDIUM": 0.0,    # Baseline reference
            "HARD": +0.55,    # Hard is ~0.55s slower when new
        }

        # Compound degradation rate multipliers (seconds/lap wear slope)
        self._compound_deg_rate = {
            "SOFT": 0.085,
            "MEDIUM": 0.045,
            "HARD": 0.025,
        }

        # Simulation memoization caches for model predictions across rollouts
        self._deg_cache: dict[tuple[str, str, int], float | None] = {}
        self._opp_pit_cache: dict[tuple[str, str, int, int, bool], float | None] = {}

    def clear_cache(self) -> None:
        """Clear model inference memoization caches."""
        self._deg_cache.clear()
        self._opp_pit_cache.clear()
        if self.overtake_adapter is not None and hasattr(self.overtake_adapter, "clear_cache"):
            self.overtake_adapter.clear_cache()

    # ------------------------------------------------------------------
    # Adapter Integration Helpers
    # ------------------------------------------------------------------

    def _get_tyre_deg_from_adapter(
        self,
        state: RaceState | None,
        car: dict[str, Any],
        sim_lap: int,
        total_laps: int,
    ) -> float | None:
        """
        Query TyreDegAdapter for predicted tyre degradation (seconds) for a simulated car.

        Builds a synthetic ParticipantState matching the car's current simulated state
        and calls predict_degradation(). The adapter uses its trained XGBoost model which
        accounts for TrackTemp, AirTemp, team identity, tyre age, and rolling pace trends.

        Returns None if adapter unavailable or errors — caller falls back to hardcoded formula.
        """
        if self.tyre_deg_adapter is None:
            return None
        cache_key = (
            str(car.get("driver", "")),
            str(car.get("compound", "")),
            int(car.get("tyre_age", 0)),
        )
        if cache_key in self._deg_cache:
            return self._deg_cache[cache_key]
        try:
            synth_participant = ParticipantState(
                driver=car["driver"],
                team=car.get("team", "Racing Team"),
                position=int(car.get("position", 5)),
                compound=car["compound"],
                tyre_life=float(car["tyre_age"]),
                stint=float(car.get("stint", 1)),
                pit_count=int(car.get("pit_count", 0)),
                last_lap_time_seconds=float(car.get("last_lap_time", 85.0)),
                laps_since_last_pit=int(car["tyre_age"]),
                is_active=True,
            )
            # Lightweight synthetic RaceState for the current simulated lap
            current_cond = getattr(state, "current_conditions", None) if state is not None else None
            synth_state = RaceState(
                current_lap=sim_lap,
                total_laps_expected=total_laps,
                grand_prix=getattr(state, "grand_prix", None) if state is not None else None,
                location=getattr(state, "location", None) if state is not None else None,
                current_conditions=current_cond if isinstance(current_cond, CurrentConditions) else CurrentConditions(),
            )
            synth_state.participants[car["driver"]] = synth_participant
            deg = self.tyre_deg_adapter.predict_degradation(synth_state, car["driver"])
            val = float(deg) if deg is not None else None
            self._deg_cache[cache_key] = val
            return val
        except Exception:
            self._deg_cache[cache_key] = None
            return None

    def _get_opponent_pit_prob(
        self,
        opp: dict[str, Any],
        ego: dict[str, Any],
        state: RaceState | None,
        sim_lap: int,
        total_laps: int,
        sc_prob: float,
    ) -> float | None:
        """
        Query OpponentModel for calibrated P(PIT|state_t) for an opponent car.

        Builds the full feature dict required by OpponentModel.predict() — including
        tyre state, gap context, SC probability, ego state, and derived strategic
        features (laps_past_nominal, tyre_life_fraction, cost_of_staying, etc. are
        computed inside OpponentModel.predict via compute_derived_features).

        Returns None if model unavailable or errors — caller falls back to heuristic.
        """
        if self.opponent_interface is None:
            return None
        opp_model = getattr(self.opponent_interface, "opponent_model", None)
        if opp_model is None:
            return None
        cache_key = (
            str(opp.get("driver", "")),
            str(opp.get("compound", "MEDIUM")),
            int(opp.get("tyre_age", 1)),
            sim_lap,
            bool(sc_prob > 0.05),
        )
        if cache_key in self._opp_pit_cache:
            return self._opp_pit_cache[cache_key]
        try:
            remaining = max(1, total_laps - sim_lap)
            compound = opp.get("compound", "MEDIUM")
            tyre_age = float(opp.get("tyre_age", 1.0))
            nominal_stint = {"SOFT": 18, "MEDIUM": 28, "HARD": 38}.get(compound, 28)
            pred_deg = self._compound_deg_rate.get(compound, 0.045) * tyre_age
            pred_pace = float(opp.get("last_lap_time", 85.0))

            opp_state_dict: dict[str, Any] = {
                # Race state
                "lap_number": sim_lap,
                "remaining_laps": remaining,
                "race_progress_fraction": min(1.0, sim_lap / max(1, total_laps)),
                "position": int(opp.get("position", 5)),
                # Tyre state
                "tyre_compound": compound,
                "tyre_age": tyre_age,
                "tyre_age_squared": tyre_age ** 2,
                "laps_since_last_pit": tyre_age,
                "pit_count": int(opp.get("pit_count", 0)),
                # Lap time features
                "last_lap_time": pred_pace,
                "rolling_3_lap_avg": pred_pace,
                "predicted_lap_time": pred_pace,
                "predicted_degradation": pred_deg,
                "predicted_lap_time_t1": pred_pace,
                "predicted_degradation_t1": self._compound_deg_rate.get(compound, 0.045) * (tyre_age + 1),
                # Safety car features
                "is_safety_car": 0,
                "is_vsc": 0,
                "p_sc_h1": float(np.clip(sc_prob, 0.0, 1.0)),
                "p_vsc_h1": float(np.clip(sc_prob * 0.5, 0.0, 1.0)),
                "p_sc_h3": float(np.clip(1.0 - (1.0 - sc_prob) ** 3, 0.0, 1.0)),
                # Weather (dry default; RaceState conditions override if available)
                "track_temp": 30.0,
                "air_temp": 25.0,
                "rainfall": 0.0,
                "pit_loss_seconds": self.config.pit_loss_green_seconds,
                # Gap context (approximate from accumulated race time delta)
                "gap_ahead": max(0.0, float(opp.get("gap_to_leader", 5.0)) - 2.0),
                "gap_behind": 2.0,
                # Ego reference state
                "ego_position": int(ego.get("position", 5)),
                "gap_to_ego": abs(
                    float(opp.get("cumulative_race_time", 0.0)) -
                    float(ego.get("cumulative_race_time", 0.0))
                ),
                "ego_tyre_age": float(ego.get("tyre_age", 5.0)),
                "ego_predicted_pace": float(ego.get("last_lap_time", 85.0)),
                "ego_predicted_deg": self._compound_deg_rate.get(
                    ego.get("compound", "MEDIUM"), 0.045
                ) * float(ego.get("tyre_age", 5.0)),
                "ego_recently_pitted": 1 if int(ego.get("pit_count", 0)) > 0 else 0,
                # Strategic context for derived feature computation (inside OpponentModel.predict)
                "laps_past_nominal": max(0.0, tyre_age - nominal_stint),
                "tyre_life_fraction": min(1.0, tyre_age / max(1, nominal_stint)),
                "field_median_lap_time": pred_pace,
            }

            # Override weather if RaceState has current conditions
            cond = getattr(state, "current_conditions", None)
            if cond is not None:
                if getattr(cond, "track_temp", None) is not None:
                    opp_state_dict["track_temp"] = float(cond.track_temp)
                if getattr(cond, "air_temp", None) is not None:
                    opp_state_dict["air_temp"] = float(cond.air_temp)
                if getattr(cond, "has_safety_car", False):
                    opp_state_dict["is_safety_car"] = 1
                if getattr(cond, "has_vsc", False):
                    opp_state_dict["is_vsc"] = 1

            pred = opp_model.predict(opp_state_dict)
            p_pit = float(pred.get("p_pit", 0.05))
            val = max(1e-4, min(1.0 - 1e-4, p_pit))
            self._opp_pit_cache[cache_key] = val
            return val
        except Exception:
            self._opp_pit_cache[cache_key] = None
            return None

    def _init_simulation_state(
        self,
        state: RaceState,
        ego_driver: str,
    ) -> tuple[dict[str, dict[str, Any]], str, list[str]]:
        """
        Extract active cars, baseline timings, and classify field into tiers.
        """
        ego_code = normalize_driver(ego_driver) or "EGO"
        total_laps = state.total_laps_expected or 57
        current_lap = state.current_lap or 1

        cars: dict[str, dict[str, Any]] = {}
        for d_code, p in state.participants.items():
            if not p.is_active:
                continue
            pos = int(p.position) if p.position is not None else 10
            gap_to_lead = float(p.gap_to_leader_seconds) if p.gap_to_leader_seconds is not None else float((pos - 1) * 2.0)
            compound = str(p.compound or "MEDIUM").upper()
            tyre_life = float(p.tyre_life if p.tyre_life is not None else 1.0)
            pit_count = int(p.pit_count if p.pit_count is not None else 0)

            # Establish initial cumulative race time proxy
            # Base lap time: use recent lap time or default ~85.0s
            last_lt = float(p.last_lap_time_seconds) if p.last_lap_time_seconds is not None else 85.0
            cum_time = float(p.total_race_time_seconds) if p.total_race_time_seconds is not None else (gap_to_lead + current_lap * 85.0)

            cars[d_code] = {
                "driver": d_code,
                "team": str(p.team or "Racing Team"),
                "position": pos,
                "gap_to_leader": gap_to_lead,
                "cumulative_race_time": cum_time,
                "compound": compound,
                "tyre_age": tyre_life,
                "stint": int(p.stint or 1),
                "pit_count": pit_count,
                "last_lap_time": last_lt,
                "is_active": True,
            }

        # If ego car is somehow missing from participants, add with defaults
        if ego_code not in cars:
            cars[ego_code] = {
                "driver": ego_code,
                "team": "Ego Team",
                "position": 5,
                "gap_to_leader": 8.0,
                "cumulative_race_time": 8.0 + current_lap * 85.0,
                "compound": "MEDIUM",
                "tyre_age": 10.0,
                "stint": 1,
                "pit_count": 0,
                "last_lap_time": 85.0,
                "is_active": True,
            }

        ego_pos = cars[ego_code]["position"]

        # Identify Tier 1 rivals: cars within config window of ego position
        tier1_drivers = [
            d for d, c in cars.items()
            if d != ego_code and abs(c["position"] - ego_pos) <= max(self.config.tier1_window_ahead, self.config.tier1_window_behind)
        ]

        return cars, ego_code, tier1_drivers

    def _estimate_lap_pace(
        self,
        car: dict[str, Any],
        lap_num: int,
        total_laps: int,
        circuit_base: float,
        is_sc: bool,
        is_vsc: bool,
        is_pitting: bool,
        rng: np.random.RandomState,
        state: RaceState | None = None,
    ) -> float:
        """
        Estimate the total lap time for a car on a specific simulated lap.
        Integrates Lap Time baseline, fuel load penalty, tyre degradation, and caution regime.

        Tyre degradation is sourced from TyreDegAdapter (XGBoost model) when available,
        falling back to the hardcoded formula if the adapter is not loaded or fails.
        """
        compound = car["compound"]
        tyre_age = car["tyre_age"]

        # 1. Base dry pace + compound pace offset
        comp_offset = self._compound_pace_offset.get(compound, 0.0)
        base = circuit_base + comp_offset

        # 2. Fuel consumption weight penalty (burns off linearly)
        fuel_penalty = max(0.0, (total_laps - lap_num) * self.config.fuel_penalty_per_lap_seconds)

        # 3. Tyre degradation pace loss
        # --- Try TyreDegAdapter (XGBoost model with TrackTemp, AirTemp, team, rolling trend) ---
        adapter_deg = (
            self._get_tyre_deg_from_adapter(state, car, lap_num, total_laps)
            if state is not None else None
        )
        if adapter_deg is not None:
            tyre_deg = float(adapter_deg)
        else:
            # Fallback: hardcoded empirical formula
            deg_slope = self._compound_deg_rate.get(compound, 0.045)
            tyre_deg = deg_slope * tyre_age + 0.008 * (tyre_age ** 1.3)

        # Cliff penalty if tyre age exceeds compound limit (applies regardless of deg source)
        max_age = self.config.get_max_tyre_age(compound)
        if tyre_age > max_age:
            cliff_excess = tyre_age - max_age
            tyre_deg += 0.35 * (cliff_excess ** 1.5)

        # 4. Caution regime delta
        caution_delta = 0.0
        if is_sc:
            caution_delta = self.config.sc_lap_time_delta_seconds
        elif is_vsc:
            caution_delta = self.config.vsc_lap_time_delta_seconds

        # 5. Pit lane time loss
        pit_delta = 0.0
        if is_pitting:
            pit_delta = self.config.pit_loss_sc_seconds if (is_sc or is_vsc) else self.config.pit_loss_green_seconds

        # 6. Driver consistency noise
        noise = rng.normal(0.0, self.config.lap_time_noise_std)

        total_lap_time = base + fuel_penalty + tyre_deg + caution_delta + pit_delta + noise
        return float(max(40.0, total_lap_time))

    def _get_sc_hazard_prob(self, state: RaceState, lap_num: int) -> float:
        """
        Obtain Safety Car probability for the simulated lap.
        Uses SCRiskAdapter if available, or circuit progress prior fallback.
        """
        if self.sc_risk_adapter is not None:
            try:
                H_t = self.sc_risk_adapter.build_H_t(state)
                # Historical prior logit or live feature proxy
                logit = float(H_t.get("logit_p_sc_hist", -3.5))
                # Sigmoid transform: 1 / (1 + exp(-logit))
                p = 1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, logit))))
                return float(np.clip(p, 0.01, 0.15))
            except Exception:
                pass
        # Empirical baseline F1 SC probability per lap (~2.5% - 3.5%)
        return 0.028

    def _run_single_rollout(
        self,
        rollout_id: int,
        initial_cars: dict[str, dict[str, Any]],
        ego_driver: str,
        strategy: Strategy,
        horizon_laps: int,
        current_lap: int,
        total_laps: int,
        circuit_base: float,
        sc_hazard_prob: float,
        tier1_drivers: list[str],
        rng: np.random.RandomState,
        py_random: random.Random,
        state: RaceState | None = None,
    ) -> RolloutOutcome:
        """
        Execute a single forward stochastic rollout over the horizon.
        Zero future-information leakage: step-by-step state propagation.
        """
        cars = {d: dict(c) for d, c in initial_cars.items()}
        initial_ego_pos = cars[ego_driver]["position"]

        sc_remaining_laps = 0
        vsc_remaining_laps = 0
        sc_ever_deployed = False
        vsc_ever_deployed = False
        opponent_pitted_in_rollout = False
        pit_stops_executed = 0

        # Create planned pit lookup for ego: lap -> target_compound
        ego_pit_plan: dict[int, str] = {}
        for stint in strategy.stints:
            if stint.start_lap > current_lap:
                ego_pit_plan[stint.start_lap] = stint.compound
        # Also map from strategy.pit_laps directly if stints didn't cover
        if not ego_pit_plan and strategy.pit_laps:
            for idx, pl in enumerate(strategy.pit_laps):
                target_comp = strategy.compounds[idx] if idx < len(strategy.compounds) else "HARD"
                ego_pit_plan[pl] = target_comp

        # Safety Car restart tracking
        sc_restart_counter = 0
        was_sc = False

        # Roll forward lap-by-lap
        for step in range(1, horizon_laps + 1):
            sim_lap = current_lap + step
            if sim_lap > total_laps:
                break

            # -------------------------------------------------------------
            # 1. Safety Car / Caution State Transition
            # -------------------------------------------------------------
            is_sc = sc_remaining_laps > 0
            is_vsc = vsc_remaining_laps > 0

            if was_sc and not is_sc:
                sc_restart_counter = 2
            is_sc_restart = (sc_restart_counter > 0)
            if sc_restart_counter > 0 and not is_sc:
                sc_restart_counter -= 1
            was_sc = is_sc

            if is_sc:
                sc_remaining_laps -= 1
            elif is_vsc:
                vsc_remaining_laps -= 1
            else:
                # Stochastic deployment check
                if py_random.random() < sc_hazard_prob:
                    is_sc = True
                    sc_ever_deployed = True
                    # Sample SC duration from discrete uniform [min, max]
                    sc_remaining_laps = py_random.randint(
                        self.config.sc_duration_min_laps,
                        self.config.sc_duration_max_laps,
                    ) - 1

                    # Safety car compresses the field
                    leader_time = min(c["cumulative_race_time"] for c in cars.values())
                    sorted_cars = sorted(cars.values(), key=lambda c: c["cumulative_race_time"])
                    for idx, c_dict in enumerate(sorted_cars):
                        c_dict["cumulative_race_time"] = leader_time + idx * self.config.sc_field_bunching_gap_seconds
                elif py_random.random() < (sc_hazard_prob * 0.5):
                    # VSC event (shorter)
                    is_vsc = True
                    vsc_ever_deployed = True
                    vsc_remaining_laps = 1

            # -------------------------------------------------------------
            # 2. Ego Pit Decision Execution
            # -------------------------------------------------------------
            ego_pitting = (sim_lap in ego_pit_plan) or (step == 1 and any(pl <= current_lap for pl in strategy.pit_laps))
            if ego_pitting:
                new_compound = ego_pit_plan.get(sim_lap)
                if not new_compound and strategy.compounds:
                    new_compound = strategy.compounds[0]
                new_compound = new_compound or "HARD"

                cars[ego_driver]["compound"] = new_compound
                cars[ego_driver]["tyre_age"] = 0.0
                cars[ego_driver]["pit_count"] += 1
                pit_stops_executed += 1

            # -------------------------------------------------------------
            # 3. Opponent Pit Decisions (Stochastic via Opponent Interface)
            # -------------------------------------------------------------
            for opp_code in tier1_drivers:
                opp = cars[opp_code]
                opp_pitting = False

                if is_sc and opp["tyre_age"] >= self.config.sc_opponent_pit_tyre_age:
                    # SC rule override: rival pits unconditionally under SC if tyres are old
                    opp_pitting = True
                elif self.opponent_interface is not None:
                    # Use OpponentModel: build full state dict and get calibrated P(PIT)
                    p_pit = self._get_opponent_pit_prob(
                        opp=opp,
                        ego=cars[ego_driver],
                        state=state,
                        sim_lap=sim_lap,
                        total_laps=total_laps,
                        sc_prob=sc_hazard_prob,
                    )
                    if p_pit is None:
                        # OpponentModel failed — fall back to sigmoid heuristic
                        median_life = self.config.get_max_tyre_age(opp["compound"]) - 6
                        x = (opp["tyre_age"] - median_life) / 3.0
                        p_pit = 1.0 / (1.0 + math.exp(-max(-5.0, min(5.0, x))))
                    action = self.opponent_interface.sample_action(p_pit)
                    opp_pitting = (action == "PIT")
                else:
                    # Heuristic sampling fallback
                    median_life = self.config.get_max_tyre_age(opp["compound"]) - 6
                    if opp["tyre_age"] >= median_life:
                        p_pit = 0.25 + 0.10 * (opp["tyre_age"] - median_life)
                        opp_pitting = (py_random.random() < min(0.90, p_pit))

                if opp_pitting:
                    opponent_pitted_in_rollout = True
                    # Switch to Hard or Medium
                    curr_c = opp["compound"]
                    next_c = "HARD" if curr_c != "HARD" else "MEDIUM"
                    opp["compound"] = next_c
                    opp["tyre_age"] = 0.0
                    opp["pit_count"] += 1
                    opp["_pitted_this_lap"] = True
                else:
                    opp["_pitted_this_lap"] = False

            # -------------------------------------------------------------
            # 4. Lap Time Simulation & Cumulative Time Update
            # -------------------------------------------------------------
            lap_times: dict[str, float] = {}
            for d_code, car in cars.items():
                is_pitting = ego_pitting if d_code == ego_driver else car.get("_pitted_this_lap", False)
                lt = self._estimate_lap_pace(
                    car=car,
                    lap_num=sim_lap,
                    total_laps=total_laps,
                    circuit_base=circuit_base,
                    is_sc=is_sc,
                    is_vsc=is_vsc,
                    is_pitting=is_pitting,
                    rng=rng,
                    state=state,
                )
                lap_times[d_code] = lt
                car["cumulative_race_time"] += lt
                if not is_pitting:
                    car["tyre_age"] += 1.0
                car["last_lap_time"] = lt

            # -------------------------------------------------------------
            # 5. Position Resolution & Overtaking / Dirty Air Dynamics
            # -------------------------------------------------------------
            sorted_cars = sorted(cars.values(), key=lambda c: c["cumulative_race_time"])
            for rank_idx, c_dict in enumerate(sorted_cars):
                c_dict["position"] = rank_idx + 1

            # Check dirty air penalty and model-driven overtaking when two cars are within 0.8s
            for i in range(len(sorted_cars) - 1):
                car_ahead = sorted_cars[i]
                car_behind = sorted_cars[i + 1]
                gap = car_behind["cumulative_race_time"] - car_ahead["cumulative_race_time"]

                if gap < 0.8 and not is_sc:
                    p_overtake = self._get_overtake_probability(
                        car_ahead=car_ahead,
                        car_behind=car_behind,
                        gap=gap,
                        state=state,
                        is_sc_restart=is_sc_restart,
                        is_sc=is_sc,
                    )
                    if py_random.random() < p_overtake:
                        # Successful overtake: swap times slightly to reflect pass
                        car_behind["cumulative_race_time"] = car_ahead["cumulative_race_time"] - 0.1
                        car_ahead["cumulative_race_time"] += 0.2
                    else:
                        # Trailing car is stuck in dirty air; receives wake penalty
                        dirty_air = self._get_dirty_air_penalty(car_behind=car_behind, gap=gap, state=state)
                        car_behind["cumulative_race_time"] += dirty_air

        # Re-sort final positions at horizon end
        final_sorted = sorted(cars.values(), key=lambda c: c["cumulative_race_time"])
        for rank_idx, c_dict in enumerate(final_sorted):
            c_dict["position"] = rank_idx + 1

        final_ego_pos = cars[ego_driver]["position"]
        pos_delta = initial_ego_pos - final_ego_pos
        final_ego_time = cars[ego_driver]["cumulative_race_time"]

        return RolloutOutcome(
            rollout_id=rollout_id,
            final_position=final_ego_pos,
            position_delta=pos_delta,
            total_race_time_seconds=final_ego_time,
            sc_deployed=sc_ever_deployed,
            vsc_deployed=vsc_ever_deployed,
            opponent_pitted=opponent_pitted_in_rollout,
            final_tyre_age=int(cars[ego_driver]["tyre_age"]),
            pit_stops_executed=pit_stops_executed,
        )

    def simulate_strategy(
        self,
        state: RaceState,
        ego_driver: str,
        strategy: Strategy,
        horizon_laps: int | None = None,
        n_rollouts: int | None = None,
        seed: int | None = None,
    ) -> SimulationResult:
        """
        Simulate N stochastic Monte Carlo rollouts for a single candidate strategy
        from current RaceState through race completion.
        Deterministic when seed is fixed; stochastic across different seeds.
        """
        num_rollouts = n_rollouts or self.default_rollouts
        current_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        remaining_laps = max(1, total_laps - current_lap)

        if horizon_laps is not None:
            h_laps = horizon_laps
        elif self.config.default_horizon_laps is not None:
            h_laps = self.config.default_horizon_laps
        elif strategy.horizon_laps is not None:
            h_laps = strategy.horizon_laps
        else:
            h_laps = remaining_laps

        effective_seed = seed if seed is not None else self.random_seed

        # Initialize isolated random number generators for determinism
        rng = np.random.RandomState(effective_seed)
        py_random = random.Random(effective_seed)

        # Baseline circuit pace
        circuit_base = 85.0
        if self.lap_time_adapter is not None:
            try:
                circuit = self.lap_time_adapter.resolve_circuit(state)
                circuit_base = float(self.lap_time_adapter.meta["circuit_dry_bases"].get(circuit, 85.0))
            except Exception:
                circuit_base = 85.0

        current_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        sc_hazard_prob = self._get_sc_hazard_prob(state, current_lap)

        initial_cars, ego_code, tier1_drivers = self._init_simulation_state(state, ego_driver)

        rollouts: list[RolloutOutcome] = []
        for r_id in range(1, num_rollouts + 1):
            outcome = self._run_single_rollout(
                rollout_id=r_id,
                initial_cars=initial_cars,
                ego_driver=ego_code,
                strategy=strategy,
                horizon_laps=h_laps,
                current_lap=current_lap,
                total_laps=total_laps,
                circuit_base=circuit_base,
                sc_hazard_prob=sc_hazard_prob,
                tier1_drivers=tier1_drivers,
                rng=rng,
                py_random=py_random,
                state=state,
            )
            rollouts.append(outcome)

        # -----------------------------------------------------------------
        # Aggregate Outcome Distributions
        # -----------------------------------------------------------------
        positions = np.array([r.final_position for r in rollouts], dtype=float)
        race_times = np.array([r.total_race_time_seconds for r in rollouts], dtype=float)

        expected_pos = float(np.mean(positions))
        std_pos = float(np.std(positions))
        median_pos = float(np.median(positions))
        p10 = float(np.percentile(positions, 10))
        p90 = float(np.percentile(positions, 90))

        expected_time = float(np.mean(race_times))
        std_time = float(np.std(race_times))

        # Position frequency distribution
        pos_counts: dict[int, float] = {}
        for p_val in range(1, 21):
            count = int(np.sum(positions == p_val))
            if count > 0:
                pos_counts[p_val] = float(count / num_rollouts)

        # Probabilities
        p_win = float(np.mean(positions == 1))
        p_podium = float(np.mean(positions <= 3))
        p_top5 = float(np.mean(positions <= 5))
        p_points = float(np.mean(positions <= 10))
        p_gain = float(np.mean([r.position_delta > 0 for r in rollouts]))

        # SC event sensitivities
        sc_rollouts = [r for r in rollouts if r.sc_deployed or r.vsc_deployed]
        green_rollouts = [r for r in rollouts if not (r.sc_deployed or r.vsc_deployed)]

        p_sc_affected = float(len(sc_rollouts) / num_rollouts)
        exp_pos_sc = float(np.mean([r.final_position for r in sc_rollouts])) if sc_rollouts else expected_pos
        exp_pos_green = float(np.mean([r.final_position for r in green_rollouts])) if green_rollouts else expected_pos
        sc_advantage = float(exp_pos_green - exp_pos_sc) # Positive = better under SC

        # Tyre wear risk: max tyre age achieved vs threshold
        max_seen_age = max(r.final_tyre_age for r in rollouts)
        target_compound = strategy.compounds[-1] if strategy.compounds else "MEDIUM"
        max_allowable_age = self.config.get_max_tyre_age(target_compound)
        tyre_risk = float(np.clip(max_seen_age / max_allowable_age, 0.0, 1.0))

        # Traffic risk proxy: frequency of finishing behind slower rival
        traffic_risk = float(np.clip(std_pos / 3.0, 0.05, 0.95))

        # Strategy robustness: 1.0 - (std_pos / 5.0)
        robustness = float(np.clip(1.0 - (std_pos / 5.0), 0.1, 0.95))

        return SimulationResult(
            strategy_id=strategy.strategy_id,
            n_rollouts=num_rollouts,
            horizon_laps=h_laps,
            expected_position=expected_pos,
            median_position=median_pos,
            std_position=std_pos,
            position_p10=p10,
            position_p90=p90,
            position_distribution=pos_counts,
            p_win=p_win,
            p_podium=p_podium,
            p_top5=p_top5,
            p_points=p_points,
            p_gain_positions=p_gain,
            expected_time_seconds=expected_time,
            std_time_seconds=std_time,
            p_sc_affected=p_sc_affected,
            expected_position_under_sc=exp_pos_sc,
            expected_position_green=exp_pos_green,
            sc_position_advantage=sc_advantage,
            tyre_risk=tyre_risk,
            traffic_risk=traffic_risk,
            robustness=robustness,
            rollouts=rollouts,
        )

    def simulate_all(
        self,
        state: RaceState,
        ego_driver: str,
        strategies: list[Strategy],
        horizon_laps: int | None = None,
        n_rollouts: int | None = None,
        seed: int | None = None,
    ) -> dict[str, SimulationResult]:
        """Simulate all candidate strategies through full remaining-race horizon."""
        results: dict[str, SimulationResult] = {}
        base_seed = seed if seed is not None else self.random_seed
        current_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        remaining_laps = max(1, total_laps - current_lap)

        if horizon_laps is not None:
            effective_horizon = horizon_laps
        elif self.config.default_horizon_laps is not None:
            effective_horizon = self.config.default_horizon_laps
        else:
            effective_horizon = remaining_laps

        for idx, strat in enumerate(strategies):
            strat_seed = (base_seed + idx * 1000) if base_seed is not None else None
            res = self.simulate_strategy(
                state=state,
                ego_driver=ego_driver,
                strategy=strat,
                horizon_laps=effective_horizon,
                n_rollouts=n_rollouts,
                seed=strat_seed,
            )
            results[strat.strategy_id] = res

        return results

    def _get_overtake_probability(
        self,
        car_ahead: dict[str, Any],
        car_behind: dict[str, Any],
        gap: float,
        state: RaceState | None,
        is_sc_restart: bool = False,
        is_sc: bool = False,
    ) -> float:
        """
        Query OvertakeAdapter for P(overtake). Falls back to pace_delta
        threshold heuristic if adapter is unavailable.
        """
        pace_delta = car_ahead["last_lap_time"] - car_behind["last_lap_time"]
        tyre_age_delta = float(car_ahead.get("tyre_age", 0.0) - car_behind.get("tyre_age", 0.0))
        compound_behind = str(car_behind.get("compound", "MEDIUM")).strip().upper()
        compound_ahead = str(car_ahead.get("compound", "MEDIUM")).strip().upper()
        is_fresh = bool(car_behind.get("tyre_age", 0.0) <= 2.0)
        circuit = getattr(state, "location", "unknown") if state else "unknown"

        if self.overtake_adapter is not None:
            try:
                return float(self.overtake_adapter.predict_overtake_probability(
                    gap_seconds=gap,
                    pace_delta=pace_delta,
                    tyre_age_delta=tyre_age_delta,
                    compound_behind=compound_behind,
                    compound_ahead=compound_ahead,
                    is_fresh_tyre_behind=is_fresh,
                    circuit=circuit,
                    is_sc_restart=is_sc_restart,
                    is_sc=is_sc,
                ))
            except Exception:
                pass

        # Fallback: sigmoid on pace delta centred at threshold
        x = (pace_delta - self.config.overtake_pace_advantage_threshold) / 0.35
        return float(1.0 / (1.0 + math.exp(-max(-6.0, min(6.0, x)))))

    def _get_dirty_air_penalty(
        self,
        car_behind: dict[str, Any],
        gap: float,
        state: RaceState | None,
    ) -> float:
        """
        Query OvertakeAdapter for dirty air wake penalty. Falls back to
        config.dirty_air_penalty_seconds if adapter is unavailable.
        """
        circuit = getattr(state, "location", "unknown") if state else "unknown"
        if self.overtake_adapter is not None:
            try:
                return float(self.overtake_adapter.predict_dirty_air_penalty(
                    gap_seconds=gap,
                    circuit=circuit,
                ))
            except Exception:
                pass
        return float(self.config.dirty_air_penalty_seconds)

