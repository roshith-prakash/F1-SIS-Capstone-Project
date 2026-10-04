"""
Gymnasium environment wrapper for F1-SIS Strategy Decision Engine.
Adapts lap-by-lap race progression into standard reinforcement learning MDP:
(State, Action, Reward, Next_State, Done).

Supports:
- Baseline physics simulation with stochastic noise
- Machine Learning physics integration (XGBoost Tyre Deg, Empirical Pitstop, SC Risk Hazard, Overtake Dynamics)
- Dynamic Reactive Opponent AI Policies
- Curriculum Learning stage generation
"""

from __future__ import annotations
import copy
from pathlib import Path
import random
from typing import Any
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from race_state.models import CurrentConditions, ParticipantState, RaceState, normalize_driver
from strategy_engine.engine import StrategyEngine
from strategy_engine.types import StrategyEngineConfig
from ..types import ImmediateAction
from ..mapper import ActionMapper
from ..state_encoder import StateEncoder
from .reward import StrategyRewardCalculator
from .opponent_policy import ReactiveOpponentPolicy, create_random_opponent_policies


class F1StrategyEnv(gym.Env):
    """
    Step-by-step F1 Race Strategy Gymnasium Environment.
    At each step:
        - Agent observes normalized state S_t and a variable set of candidate strategies.
        - Action A_t selects one of K candidate strategies (discrete index 0 to K-1).
        - Action is mapped to an immediate single-lap tactical command (STAY_OUT or PIT_COMPOUND).
        - Environment advances race by 1 lap, updating tyre degradation, lap times, gaps, and positions.
        - Returns next_state, reward, terminated, truncated, info.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        engine: StrategyEngine | None = None,
        base_state: RaceState | None = None,
        ego_driver: str = "VER",
        max_candidates: int = 10,
        rollouts_per_step: int = 15,
        horizon_laps: int = 10,
        reward_calculator: StrategyRewardCalculator | None = None,
        state_encoder: StateEncoder | None = None,
        seed: int = 42,
        tyre_deg_adapter: Any = None,
        pitstop_adapter: Any = None,
        sc_risk_adapter: Any = None,
        overtake_adapter: Any = None,
        lap_time_adapter: Any = None,
        opponent_model: Any = None,
        opponent_policies: dict[str, ReactiveOpponentPolicy] | None = None,
        default_opponent_policy: ReactiveOpponentPolicy | None = None,
        use_ml_physics: bool = False,
    ):
        super().__init__()
        self.ego_driver = normalize_driver(ego_driver) or "VER"
        self.max_candidates = max_candidates
        self.rollouts_per_step = rollouts_per_step
        self.horizon_laps = horizon_laps
        self.reward_calculator = reward_calculator or StrategyRewardCalculator()
        self.state_encoder = state_encoder or StateEncoder()
        self.random_seed = seed
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        self.use_ml_physics = use_ml_physics

        # Adapters
        self.tyre_deg_adapter = tyre_deg_adapter
        self.pitstop_adapter = pitstop_adapter
        self.sc_risk_adapter = sc_risk_adapter
        self.overtake_adapter = overtake_adapter
        self.lap_time_adapter = lap_time_adapter
        self.opponent_model = opponent_model

        # Opponent policies
        self.opponent_policies = opponent_policies or {}
        self.default_opponent_policy = default_opponent_policy or ReactiveOpponentPolicy(personality="balanced")

        # Auto-load adapters when use_ml_physics is True and adapter is None
        if self.use_ml_physics:
            self._lazy_load_adapters()

        # Inference memoization caches for RL step performance
        self._deg_cache: dict[tuple[str, str, int], float] = {}
        self._opp_pit_cache: dict[tuple[str, str, int, int], bool] = {}

        # Initialize StrategyEngine if not provided
        if engine is not None:
            self.engine = engine
        else:
            cfg = StrategyEngineConfig(
                default_horizon_laps=self.horizon_laps,
                default_n_rollouts=self.rollouts_per_step,
                default_random_seed=seed,
            )
            self.engine = StrategyEngine(config=cfg)

        self.base_state = base_state or self._create_default_initial_state()
        self.current_state: RaceState = copy.deepcopy(self.base_state)

        # Observation space matching StateEncoder dimensions (13 or 16)
        obs_dim = getattr(self.state_encoder, "state_dim", 13)
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(obs_dim,), dtype=np.float32
        )

        # Action space: index selecting a candidate strategy (0 to max_candidates - 1)
        self.action_space = spaces.Discrete(self.max_candidates)

        self.current_candidates: list[dict[str, Any]] = []
        self.candidate_features: np.ndarray = np.zeros((0, 9), dtype=np.float32)

    def _lazy_load_adapters(self) -> None:
        """Loads repository ML submodels gracefully with fallback."""
        if self.tyre_deg_adapter is None:
            try:
                from tyre_deg.adapter import TyreDegAdapter
                self.tyre_deg_adapter = TyreDegAdapter()
            except Exception:
                pass

        if self.pitstop_adapter is None:
            try:
                from pitstop.adapter import PitstopAdapter
                self.pitstop_adapter = PitstopAdapter()
            except Exception:
                pass

        if self.sc_risk_adapter is None:
            try:
                from sc_risk.adapter import SCRiskAdapter
                sc_prior_path = Path("models") / "SC Estimation" / "sc_vsc_historical_prior.csv"
                if sc_prior_path.exists():
                    self.sc_risk_adapter = SCRiskAdapter(str(sc_prior_path))
            except Exception:
                pass

        if self.overtake_adapter is None:
            try:
                from overtake.adapter import OvertakeAdapter
                self.overtake_adapter = OvertakeAdapter()
            except Exception:
                pass

        if self.opponent_model is None:
            try:
                from opponent_model.model import OpponentModel
                self.opponent_model = OpponentModel.load()
            except Exception:
                pass

    @classmethod
    def create_stage_initial_state(cls, stage: int) -> RaceState:
        """
        Factory producing initial race states tailored to curriculum stages.
        """
        if stage == 1:
            # Stage 1: Sprint Endgame (Lap 42/52, 10 laps remaining, 3 cars)
            cond = CurrentConditions(air_temp=24.0, track_temp=36.0, has_safety_car=False, has_vsc=False)
            state = RaceState(
                race_id="2024_Stage1_Sprint",
                current_lap=42,
                total_laps_expected=52,
                grand_prix="British Grand Prix",
                location="Silverstone",
                current_conditions=cond,
            )
            grid = [
                ("NOR", "McLaren", 1, 0.0, 3600.0, "HARD", 20.0),
                ("VER", "Red Bull Racing", 2, 1.2, 3601.2, "MEDIUM", 12.0),
                ("HAM", "Mercedes", 3, 3.5, 3603.5, "HARD", 22.0),
            ]
            for d_code, team, pos, gap, ctime, comp, life in grid:
                state.participants[d_code] = ParticipantState(
                    driver=d_code,
                    team=team,
                    position=pos,
                    gap_to_leader_seconds=gap,
                    total_race_time_seconds=ctime,
                    compound=comp,
                    tyre_life=life,
                    stint=2,
                    pit_count=1,
                    last_lap_time_seconds=85.5,
                    is_active=True,
                )
            return state

        elif stage == 2:
            # Stage 2: Pit Window (Lap 16/52, 5 cars)
            cond = CurrentConditions(air_temp=25.0, track_temp=38.0, has_safety_car=False, has_vsc=False)
            state = RaceState(
                race_id="2024_Stage2_PitWindow",
                current_lap=16,
                total_laps_expected=52,
                grand_prix="British Grand Prix",
                location="Silverstone",
                current_conditions=cond,
            )
            grid = [
                ("NOR", "McLaren", 1, 0.0, 1360.0, "MEDIUM", 16.0),
                ("VER", "Red Bull Racing", 2, 1.5, 1361.5, "MEDIUM", 16.0),
                ("HAM", "Mercedes", 3, 3.8, 1363.8, "MEDIUM", 16.0),
                ("LEC", "Ferrari", 4, 6.5, 1366.5, "HARD", 16.0),
                ("PIA", "McLaren", 5, 9.2, 1369.2, "HARD", 16.0),
            ]
            for d_code, team, pos, gap, ctime, comp, life in grid:
                state.participants[d_code] = ParticipantState(
                    driver=d_code,
                    team=team,
                    position=pos,
                    gap_to_leader_seconds=gap,
                    total_race_time_seconds=ctime,
                    compound=comp,
                    tyre_life=life,
                    stint=1,
                    pit_count=0,
                    last_lap_time_seconds=85.2,
                    is_active=True,
                )
            return state

        elif stage == 3:
            # Stage 3: SC Chaos (Lap 22/52, Safety Car active)
            cond = CurrentConditions(air_temp=23.0, track_temp=32.0, has_safety_car=True, has_vsc=False)
            state = RaceState(
                race_id="2024_Stage3_SCChaos",
                current_lap=22,
                total_laps_expected=52,
                grand_prix="British Grand Prix",
                location="Silverstone",
                current_conditions=cond,
            )
            grid = [
                ("NOR", "McLaren", 1, 0.0, 1920.0, "MEDIUM", 22.0),
                ("VER", "Red Bull Racing", 2, 0.6, 1920.6, "MEDIUM", 22.0),
                ("HAM", "Mercedes", 3, 1.2, 1921.2, "MEDIUM", 22.0),
                ("LEC", "Ferrari", 4, 1.8, 1921.8, "HARD", 22.0),
                ("PIA", "McLaren", 5, 2.4, 1922.4, "HARD", 22.0),
            ]
            for d_code, team, pos, gap, ctime, comp, life in grid:
                state.participants[d_code] = ParticipantState(
                    driver=d_code,
                    team=team,
                    position=pos,
                    gap_to_leader_seconds=gap,
                    total_race_time_seconds=ctime,
                    compound=comp,
                    tyre_life=life,
                    stint=1,
                    pit_count=0,
                    last_lap_time_seconds=110.0,
                    is_active=True,
                )
            return state

        else:
            # Stage 4: Full Grand Prix from lap 1
            return cls._create_default_initial_state_static()

    @staticmethod
    def _create_default_initial_state_static() -> RaceState:
        cond = CurrentConditions(
            air_temp=25.0,
            track_temp=35.0,
            has_safety_car=False,
            has_vsc=False,
        )
        state = RaceState(
            race_id="2024_British_GP",
            current_lap=1,
            total_laps_expected=52,
            grand_prix="British Grand Prix",
            location="Silverstone",
            current_conditions=cond,
        )
        grid = [
            ("NOR", "McLaren", 1, 0.0, 0.0, "MEDIUM", 0.0),
            ("VER", "Red Bull Racing", 2, 1.5, 1.5, "MEDIUM", 0.0),
            ("HAM", "Mercedes", 3, 3.2, 3.2, "MEDIUM", 0.0),
            ("LEC", "Ferrari", 4, 5.0, 5.0, "HARD", 0.0),
            ("PIA", "McLaren", 5, 7.5, 7.5, "HARD", 0.0),
        ]
        for d_code, team, pos, gap, ctime, comp, life in grid:
            state.participants[d_code] = ParticipantState(
                driver=d_code,
                team=team,
                position=pos,
                gap_to_leader_seconds=gap,
                total_race_time_seconds=ctime,
                compound=comp,
                tyre_life=life,
                stint=1,
                pit_count=0,
                last_lap_time_seconds=85.0,
                is_active=True,
            )
        return state

    def _create_default_initial_state(self) -> RaceState:
        """Constructs a default competitive 5-car grid for Silverstone."""
        return self._create_default_initial_state_static()

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Resets the environment to the beginning of the race."""
        if seed is not None:
            self.random_seed = seed
            self.rng = np.random.RandomState(seed)
            self.py_rng = random.Random(seed)

        self._deg_cache.clear()
        self._opp_pit_cache.clear()
        if self.overtake_adapter is not None and hasattr(self.overtake_adapter, "clear_cache"):
            self.overtake_adapter.clear_cache()

        self.current_state = copy.deepcopy(self.base_state)

        # Randomize opponent policies at reset for diverse strategic behaviors
        active_opponents = [
            code for code in self.current_state.participants.keys() if code != self.ego_driver
        ]
        if not self.opponent_policies or len(self.opponent_policies) != len(active_opponents):
            self.opponent_policies = create_random_opponent_policies(
                active_opponents, rng=self.py_rng
            )

        # Update candidates for the starting state
        self._refresh_candidates()

        obs = self.state_encoder.encode_state(self.current_state, self.ego_driver)
        info = self._get_info()
        return obs, info

    def _refresh_candidates(self) -> None:
        """Queries StrategyEngine to produce evaluated alternatives for current state."""
        try:
            res = self.engine.evaluate_race_state(
                state=self.current_state,
                ego_driver=self.ego_driver,
                horizon_laps=self.horizon_laps,
                n_rollouts=self.rollouts_per_step,
            )
            payload = res.to_decision_engine_payload()
            self.current_candidates = payload.get("strategies", [])
        except Exception:
            # Fallback candidate if simulation errors
            cur_lap = self.current_state.current_lap or 1
            self.current_candidates = [
                {
                    "strategy_id": "FALLBACK_1",
                    "pit_laps": [cur_lap + 15],
                    "compounds": ["MEDIUM", "HARD"],
                    "expected_position": 2.0,
                    "p_win": 0.3,
                    "p_podium": 0.7,
                    "p_top5": 0.95,
                    "tyre_risk": 0.2,
                    "traffic_risk": 0.2,
                    "robustness": 0.8,
                }
            ]

        cur_lap = self.current_state.current_lap or 1
        self.candidate_features = self.state_encoder.encode_candidates_batch(
            self.current_candidates, cur_lap
        )

    def _get_info(self) -> dict[str, Any]:
        """Builds step info payload with candidate features and valid action masks."""
        k = len(self.current_candidates)
        mask = np.zeros(self.max_candidates, dtype=bool)
        mask[: min(k, self.max_candidates)] = True

        ego_p = self.current_state.participants.get(self.ego_driver)
        pos = int(ego_p.position) if ego_p and ego_p.position is not None else 10

        return {
            "current_lap": self.current_state.current_lap or 1,
            "ego_position": pos,
            "candidates": self.current_candidates,
            "candidate_features": self.candidate_features,
            "valid_mask": mask,
            "action_count": min(k, self.max_candidates),
        }

    def _calc_deg_penalty(self, p: ParticipantState) -> float:
        """Calculates tyre degradation using TyreDegAdapter (if ML physics active) or linear wear."""
        comp = str(p.compound or "MEDIUM").upper()
        life = float(p.tyre_life or 0.0)
        deg_factor = 0.05 if comp == "MEDIUM" else 0.08 if comp == "SOFT" else 0.03
        fallback_deg = life * deg_factor

        if self.use_ml_physics and self.tyre_deg_adapter is not None:
            cache_key = (str(p.driver), comp, int(life))
            if cache_key in self._deg_cache:
                return self._deg_cache[cache_key]
            try:
                deg = self.tyre_deg_adapter.predict_degradation(self.current_state, p.driver)
                val = float(deg) if deg is not None else fallback_deg
                self._deg_cache[cache_key] = val
                return val
            except Exception:
                return fallback_deg

        return fallback_deg

    def _calc_pit_loss(self, p: ParticipantState, is_sc: bool) -> float:
        """Calculates pit stop duration via PitstopAdapter or standard constants."""
        fallback_pit = 14.0 if is_sc else 22.0
        if self.use_ml_physics and self.pitstop_adapter is not None:
            try:
                pit_loss = self.pitstop_adapter.sample_pit_duration(
                    circuit=self.current_state.location or "Silverstone",
                    team=p.team or "Red Bull Racing",
                    is_sc=is_sc,
                    is_vsc=bool(self.current_state.current_conditions.has_vsc),
                    rng=self.rng,
                )
                return float(pit_loss)
            except Exception:
                return fallback_pit
        return fallback_pit

    def _simulate_lap_transition(
        self, action: ImmediateAction, target_compound: str | None
    ) -> None:
        """
        Performs one-lap physics and timing updates for all active participants.
        Integrates TyreDegAdapter, PitstopAdapter, SCRiskAdapter, OvertakeAdapter,
        and ReactiveOpponentPolicy.
        """
        cur_lap = self.current_state.current_lap or 1
        total_laps = self.current_state.total_laps_expected or 52
        is_sc = bool(self.current_state.current_conditions.has_safety_car)
        ego_pitted = action != ImmediateAction.STAY_OUT

        # 1. Update ego driver
        ego = self.current_state.participants.get(self.ego_driver)
        if ego is not None:
            if ego_pitted and target_compound:
                ego.compound = target_compound
                ego.tyre_life = 0.0
                ego.pit_count = int(ego.pit_count or 0) + 1
                pit_loss = self._calc_pit_loss(ego, is_sc)
            else:
                ego.tyre_life = float(ego.tyre_life or 0.0) + 1.0
                pit_loss = 0.0

            deg_penalty = self._calc_deg_penalty(ego)
            pace = 85.0 + deg_penalty + pit_loss + self.rng.normal(0.0, 0.2)
            if is_sc:
                pace += 25.0
            ego.last_lap_time_seconds = pace
            ego.total_race_time_seconds = float(ego.total_race_time_seconds or 0.0) + pace

        # 2. Update opponent drivers
        for code, p in self.current_state.participants.items():
            if code == self.ego_driver or not p.is_active:
                continue

            comp = str(p.compound or "MEDIUM").upper()
            life = float(p.tyre_life or 0.0)
            opp_pits = False

            # Query ReactiveOpponentPolicy
            opp_policy = self.opponent_policies.get(code, self.default_opponent_policy)
            opp_pits = opp_policy.decide_pit(
                opp_driver=code,
                opp_life=life,
                opp_compound=comp,
                opp_gap_ahead=float(p.interval_to_position_ahead_seconds or 5.0),
                opp_gap_behind=float(p.gap_behind_seconds or 5.0),
                current_lap=cur_lap,
                total_laps=total_laps,
                ego_driver=self.ego_driver,
                ego_pitted_this_lap=ego_pitted,
                is_sc=is_sc,
                py_rng=self.py_rng,
            )

            # Check ML OpponentModel if policy did not trigger pit
            if not opp_pits and self.use_ml_physics and self.opponent_model is not None:
                cache_key = (code, comp, int(life), cur_lap)
                if cache_key in self._opp_pit_cache:
                    opp_pits = self._opp_pit_cache[cache_key]
                else:
                    try:
                        pred = self.opponent_model.predict({
                            "lap_number": cur_lap,
                            "remaining_laps": max(1, total_laps - cur_lap),
                            "race_progress_fraction": min(1.0, cur_lap / max(1, total_laps)),
                            "position": int(p.position or 5),
                            "tyre_compound": comp,
                            "tyre_age": life,
                            "tyre_age_squared": life ** 2,
                            "laps_since_last_pit": life,
                            "gap_ahead_seconds": float(p.interval_to_position_ahead_seconds or 5.0),
                            "gap_behind_seconds": float(p.gap_behind_seconds or 5.0),
                            "predicted_lap_time": float(p.last_lap_time_seconds or 85.0),
                            "predicted_deg": self._calc_deg_penalty(p),
                            "sc_probability": 0.03,
                            "pit_loss_seconds": 22.0,
                        })
                        p_pit = float(pred.get("p_pit", 0.05)) if isinstance(pred, dict) else 0.05
                        opp_pits = self.py_rng.random() < p_pit
                        self._opp_pit_cache[cache_key] = opp_pits
                    except Exception:
                        opp_pits = False

            if opp_pits:
                p.compound = "HARD" if comp == "MEDIUM" else "MEDIUM"
                p.tyre_life = 0.0
                p.pit_count = int(p.pit_count or 0) + 1
                opp_pit_loss = self._calc_pit_loss(p, is_sc)
            else:
                p.tyre_life = life + 1.0
                opp_pit_loss = 0.0

            opp_deg = self._calc_deg_penalty(p)
            opp_pace = 85.0 + opp_deg + opp_pit_loss + self.rng.normal(0.0, 0.2)
            if is_sc:
                opp_pace += 25.0
            p.last_lap_time_seconds = opp_pace
            p.total_race_time_seconds = float(p.total_race_time_seconds or 0.0) + opp_pace

        # 3. Re-sort field by cumulative total race time
        active_participants = [
            p for p in self.current_state.participants.values() if p.is_active
        ]
        active_participants.sort(key=lambda p: float(p.total_race_time_seconds or 0.0))

        # 4. Overtake dynamics & dirty air resolution pass
        if self.use_ml_physics and self.overtake_adapter is not None and not is_sc:
            for i in range(1, len(active_participants)):
                behind = active_participants[i]
                ahead = active_participants[i - 1]
                gap = float(behind.total_race_time_seconds or 0.0) - float(ahead.total_race_time_seconds or 0.0)

                # Close proximity (< 1.5 seconds)
                if 0.0 <= gap < 1.5:
                    pace_delta = float(ahead.last_lap_time_seconds or 85.0) - float(behind.last_lap_time_seconds or 85.0)
                    try:
                        p_overtake = self.overtake_adapter.predict_overtake_probability(
                            gap_seconds=max(0.1, gap),
                            pace_delta=pace_delta,
                            tyre_age_delta=float(ahead.tyre_life or 0) - float(behind.tyre_life or 0),
                            compound_behind=str(behind.compound or "MEDIUM"),
                            compound_ahead=str(ahead.compound or "MEDIUM"),
                            is_fresh_tyre_behind=(behind.tyre_life or 0) <= 2,
                            circuit=self.current_state.location or "Silverstone",
                        )
                    except Exception:
                        p_overtake = 0.45

                    # If overtake not achieved, apply dirty air wake penalty to car behind
                    if self.py_rng.random() >= p_overtake:
                        try:
                            dirty_air = self.overtake_adapter.predict_dirty_air_penalty(
                                gap_seconds=max(0.1, gap),
                                circuit=self.current_state.location or "Silverstone",
                            )
                        except Exception:
                            dirty_air = 0.25
                        behind.total_race_time_seconds = float(behind.total_race_time_seconds or 0.0) + float(dirty_air)

            # Re-sort after dirty air adjustments
            active_participants.sort(key=lambda p: float(p.total_race_time_seconds or 0.0))

        # 5. Assign updated positions and gaps
        leader_time = float(active_participants[0].total_race_time_seconds or 0.0)
        for rank, p in enumerate(active_participants, start=1):
            p.position = rank
            p.gap_to_leader_seconds = float(p.total_race_time_seconds or 0.0) - leader_time
            if rank > 1:
                ahead_p = active_participants[rank - 2]
                p.interval_to_position_ahead_seconds = (
                    float(p.total_race_time_seconds or 0.0)
                    - float(ahead_p.total_race_time_seconds or 0.0)
                )
            else:
                p.interval_to_position_ahead_seconds = 0.0

            if rank < len(active_participants):
                behind_p = active_participants[rank]
                p.gap_behind_seconds = (
                    float(behind_p.total_race_time_seconds or 0.0)
                    - float(p.total_race_time_seconds or 0.0)
                )
            else:
                p.gap_behind_seconds = 20.0

        # 6. Inform tyre degradation adapter of lap observation
        if self.use_ml_physics and self.tyre_deg_adapter is not None:
            if hasattr(self.tyre_deg_adapter, "observe_lap"):
                try:
                    self.tyre_deg_adapter.observe_lap(self.current_state)
                except Exception:
                    pass

        # 7. Safety car stochastic transition
        if not is_sc:
            sc_prob = 0.03
            if self.use_ml_physics and self.sc_risk_adapter is not None:
                try:
                    H_t = self.sc_risk_adapter.build_H_t(self.current_state)
                    sc_prob = float(H_t.get("p_sc", 0.03)) if isinstance(H_t, dict) else 0.03
                except Exception:
                    sc_prob = 0.03
            if self.py_rng.random() < sc_prob:
                self.current_state.current_conditions.has_safety_car = True
        elif is_sc and self.py_rng.random() < 0.35:
            self.current_state.current_conditions.has_safety_car = False

        # Advance lap counter
        self.current_state.current_lap = cur_lap + 1

    def step(
        self, action_idx: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Advances the race by one lap based on the candidate chosen by the agent.
        """
        cur_lap = self.current_state.current_lap or 1
        total_laps = self.current_state.total_laps_expected or 52

        # 1. Resolve selected candidate strategy
        if self.current_candidates and 0 <= action_idx < len(self.current_candidates):
            selected_cand = self.current_candidates[action_idx]
        elif self.current_candidates:
            selected_cand = self.current_candidates[0]
        else:
            selected_cand = {"pit_laps": [], "compounds": ["MEDIUM"]}

        # 2. Map to immediate tactical action
        action, target_lap, target_comp = ActionMapper.map_to_immediate_action(
            cur_lap, selected_cand
        )

        ego_p = self.current_state.participants.get(self.ego_driver)
        prev_pos = int(ego_p.position) if ego_p and ego_p.position is not None else 10

        is_pitting = action != ImmediateAction.STAY_OUT

        # 3. Simulate 1-lap state transition
        self._simulate_lap_transition(action=action, target_compound=target_comp)

        # 4. Compute reward
        curr_pos = int(ego_p.position) if ego_p and ego_p.position is not None else prev_pos
        terminated = bool((self.current_state.current_lap or 1) >= total_laps)
        truncated = False

        reward = self.reward_calculator.calculate_step_reward(
            prev_position=prev_pos,
            curr_position=curr_pos,
            is_pit_lap=is_pitting,
            is_terminal=terminated,
            final_position=curr_pos,
        )

        # 5. Generate candidates for next state if not done
        if not terminated:
            self._refresh_candidates()

        obs = self.state_encoder.encode_state(self.current_state, self.ego_driver)
        info = self._get_info()
        info["immediate_action"] = action.value

        return obs, reward, terminated, truncated, info
