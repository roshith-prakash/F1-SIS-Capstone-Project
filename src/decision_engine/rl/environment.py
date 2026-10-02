"""
Gymnasium environment wrapper for F1-SIS Strategy Decision Engine.
Adapts lap-by-lap race progression into standard reinforcement learning MDP:
(State, Action, Reward, Next_State, Done).
"""

from __future__ import annotations
import copy
import random
from typing import Any, Callable
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

        # Observation space: 13-dim normalized state vector
        self.observation_space = spaces.Box(
            low=0.0, high=2.0, shape=(13,), dtype=np.float32
        )

        # Action space: index selecting a candidate strategy (0 to max_candidates - 1)
        self.action_space = spaces.Discrete(self.max_candidates)

        self.current_candidates: list[dict[str, Any]] = []
        self.candidate_features: np.ndarray = np.zeros((0, 9), dtype=np.float32)

    def _create_default_initial_state(self) -> RaceState:
        """Constructs a default competitive 5-car grid for Silverstone."""
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

        self.current_state = copy.deepcopy(self.base_state)
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

    def _simulate_lap_transition(
        self, action: ImmediateAction, target_compound: str | None
    ) -> None:
        """
        Performs one-lap physics and timing updates for all active participants.
        """
        cur_lap = self.current_state.current_lap or 1
        is_sc = bool(self.current_state.current_conditions.has_safety_car)

        # Update ego driver
        ego = self.current_state.participants.get(self.ego_driver)
        if ego is not None:
            if action != ImmediateAction.STAY_OUT and target_compound:
                ego.compound = target_compound
                ego.tyre_life = 0.0
                ego.pit_count = int(ego.pit_count or 0) + 1
                pit_loss = 14.0 if is_sc else 22.0
            else:
                ego.tyre_life = float(ego.tyre_life or 0.0) + 1.0
                pit_loss = 0.0

            # Lap pace calculation (base + deg + noise)
            deg_factor = 0.05 if ego.compound == "MEDIUM" else 0.08 if ego.compound == "SOFT" else 0.03
            deg_penalty = (ego.tyre_life or 0.0) * deg_factor
            pace = 85.0 + deg_penalty + pit_loss + self.rng.normal(0.0, 0.2)
            if is_sc:
                pace += 25.0
            ego.last_lap_time_seconds = pace
            ego.total_race_time_seconds = float(ego.total_race_time_seconds or 0.0) + pace

        # Update opponent drivers
        for code, p in self.current_state.participants.items():
            if code == self.ego_driver or not p.is_active:
                continue

            comp = str(p.compound or "MEDIUM").upper()
            life = float(p.tyre_life or 0.0)

            # Heuristic pit decision for opponents: pit if tyre is older than nominal threshold
            nominal_life = 30 if comp == "MEDIUM" else 20 if comp == "SOFT" else 40
            opp_pits = False
            if life >= nominal_life and (cur_lap < (self.current_state.total_laps_expected or 52) - 5):
                opp_pits = True
                p.compound = "HARD" if comp == "MEDIUM" else "MEDIUM"
                p.tyre_life = 0.0
                p.pit_count = int(p.pit_count or 0) + 1
            else:
                p.tyre_life = life + 1.0

            opp_pit_loss = (14.0 if is_sc else 22.0) if opp_pits else 0.0
            opp_deg = (p.tyre_life or 0.0) * (0.05 if p.compound == "MEDIUM" else 0.03)
            opp_pace = 85.0 + opp_deg + opp_pit_loss + self.rng.normal(0.0, 0.2)
            if is_sc:
                opp_pace += 25.0
            p.last_lap_time_seconds = opp_pace
            p.total_race_time_seconds = float(p.total_race_time_seconds or 0.0) + opp_pace

        # Re-sort field by cumulative total race time to update positions and gaps
        active_participants = [
            p for p in self.current_state.participants.values() if p.is_active
        ]
        active_participants.sort(key=lambda p: float(p.total_race_time_seconds or 0.0))

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

        # Safety car stochastic transition: ~3% chance per lap
        if not is_sc and self.py_rng.random() < 0.03:
            self.current_state.current_conditions.has_safety_car = True
        elif is_sc and self.py_rng.random() < 0.35:  # SC ends
            self.current_state.current_conditions.has_safety_car = False

        # Advance lap counter
        self.current_state.current_lap = cur_lap + 1
