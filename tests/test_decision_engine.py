"""
tests/test_decision_engine.py
=============================
Comprehensive unit and integration test suite for the F1-SIS Decision Engine.
Tests:
- Action mapping (STAY_OUT vs PIT_{COMPOUND})
- Feature normalization and state encoding
- Deterministic Baseline Policy with risk profiles and explainability
- Integration with StrategyEngine outputs
- Reinforcement Learning components (Reward, DQN, Replay Buffer, Gym Env)
- Policy Evaluator
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.models import CurrentConditions, ParticipantState, RaceState
from strategy_engine.engine import StrategyEngine
from strategy_engine.types import StrategyEngineConfig, StrategyEngineResult
from decision_engine.types import ImmediateAction, DecisionExplanation
from decision_engine.mapper import ActionMapper
from decision_engine.state_encoder import StateEncoder
from decision_engine.baseline import BaselineDecisionPolicy
from decision_engine.rl.reward import StrategyRewardCalculator, F1_POINTS_MAP
from decision_engine.rl.dqn import CandidateConditionedQNetwork, RLDecisionPolicy
from decision_engine.rl.replay_buffer import VariableCandidateReplayBuffer
from decision_engine.rl.environment import F1StrategyEnv
from decision_engine.rl.trainer import DQNTrainer
from decision_engine.evaluator import PolicyEvaluator, PolicyEvaluationResult


class TestDecisionEngineTypesAndMapper(unittest.TestCase):
    def test_immediate_action_enums(self):
        self.assertEqual(ImmediateAction.STAY_OUT.value, "STAY_OUT")
        self.assertEqual(ImmediateAction.PIT_SOFT.value, "PIT_SOFT")
        self.assertEqual(ImmediateAction.PIT_MEDIUM.value, "PIT_MEDIUM")
        self.assertEqual(ImmediateAction.PIT_HARD.value, "PIT_HARD")

    def test_action_mapper_stay_out(self):
        strat = {
            "strategy_id": "S1",
            "pit_laps": [25],
            "compounds": ["MEDIUM", "HARD"],
        }
        # Lap 10: not a pit lap
        action, target_lap, target_comp = ActionMapper.map_to_immediate_action(10, strat)
        self.assertEqual(action, ImmediateAction.STAY_OUT)
        self.assertEqual(target_lap, 25)
        self.assertEqual(target_comp, "HARD")

    def test_action_mapper_pit_lap(self):
        strat = {
            "strategy_id": "S1",
            "pit_laps": [25],
            "compounds": ["MEDIUM", "HARD"],
        }
        # Lap 25: pit lap
        action, target_lap, target_comp = ActionMapper.map_to_immediate_action(25, strat)
        self.assertEqual(action, ImmediateAction.PIT_HARD)
        self.assertEqual(target_lap, 25)
        self.assertEqual(target_comp, "HARD")

    def test_action_mapper_two_stops(self):
        strat = {
            "strategy_id": "S2",
            "pit_laps": [18, 38],
            "compounds": ["SOFT", "MEDIUM", "HARD"],
        }
        # Lap 18: first stop for MEDIUM
        action1, lap1, comp1 = ActionMapper.map_to_immediate_action(18, strat)
        self.assertEqual(action1, ImmediateAction.PIT_MEDIUM)
        self.assertEqual(comp1, "MEDIUM")

        # Lap 25: between stops
        action2, lap2, comp2 = ActionMapper.map_to_immediate_action(25, strat)
        self.assertEqual(action2, ImmediateAction.STAY_OUT)
        self.assertEqual(lap2, 38)
        self.assertEqual(comp2, "HARD")

        # Lap 38: second stop for HARD
        action3, lap3, comp3 = ActionMapper.map_to_immediate_action(38, strat)
        self.assertEqual(action3, ImmediateAction.PIT_HARD)
        self.assertEqual(comp3, "HARD")


class TestStateEncoder(unittest.TestCase):
    def setUp(self):
        self.encoder = StateEncoder()

    def test_encode_state(self):
        cond = CurrentConditions(
            air_temp=24.0,
            track_temp=40.0,
            has_safety_car=True,
            has_vsc=False,
        )
        state = RaceState(
            race_id="2024_Test_GP",
            current_lap=25,
            total_laps_expected=50,
            current_conditions=cond,
        )
        state.participants["VER"] = ParticipantState(
            driver="VER",
            position=2,
            tyre_life=12.0,
            compound="MEDIUM",
            interval_to_position_ahead_seconds=1.5,
            gap_behind_seconds=4.0,
        )

        s_vec = self.encoder.encode_state(state, ego_driver="VER")
        self.assertEqual(s_vec.shape, (13,))
        self.assertAlmostEqual(s_vec[0], 0.5)  # lap 25/50 = 0.5
        self.assertEqual(s_vec[1], 1.0)       # is_sc = 1.0
        self.assertEqual(s_vec[2], 0.0)       # is_vsc = 0.0
        self.assertAlmostEqual(s_vec[3], 0.5)  # track_temp: (40-20)/40 = 0.5
        self.assertAlmostEqual(s_vec[4], 0.9)  # pos 2: 1 - 2/20 = 0.9

    def test_encode_candidate(self):
        cand = {
            "strategy_id": "S1",
            "pit_laps": [20],
            "compounds": ["MEDIUM", "HARD"],
            "expected_position": 2.5,
            "p_win": 0.45,
            "p_podium": 0.85,
            "p_top5": 0.95,
            "tyre_risk": 0.15,
            "traffic_risk": 0.20,
            "robustness": 0.75,
        }
        c_vec = self.encoder.encode_candidate(cand, current_lap=20)
        self.assertEqual(c_vec.shape, (9,))
        self.assertEqual(c_vec[7], 1.0)  # is_immediate_pit = 1.0 on lap 20

    def test_encode_candidates_batch(self):
        cands = [
            {"expected_position": 1.0, "p_win": 0.6, "pit_laps": [20]},
            {"expected_position": 3.0, "p_win": 0.2, "pit_laps": [30]},
        ]
        batch = self.encoder.encode_candidates_batch(cands, current_lap=10)
        self.assertEqual(batch.shape, (2, 9))


class TestBaselineDecisionPolicy(unittest.TestCase):
    def setUp(self):
        self.candidates = [
            {
                "strategy_id": "S_AGGRESSIVE",
                "pit_laps": [15],
                "compounds": ["SOFT", "MEDIUM"],
                "expected_position": 1.8,
                "p_win": 0.55,
                "p_podium": 0.80,
                "p_top5": 0.90,
                "tyre_risk": 0.45,
                "traffic_risk": 0.35,
                "robustness": 0.40,
            },
            {
                "strategy_id": "S_CONSERVATIVE",
                "pit_laps": [25],
                "compounds": ["MEDIUM", "HARD"],
                "expected_position": 2.2,
                "p_win": 0.25,
                "p_podium": 0.85,
                "p_top5": 0.98,
                "tyre_risk": 0.10,
                "traffic_risk": 0.15,
                "robustness": 0.85,
            },
        ]

    def test_risk_profile_preferences(self):
        # Aggressive policy should prefer S_AGGRESSIVE due to high p_win
        pol_agg = BaselineDecisionPolicy(risk_profile="aggressive")
        expl_agg = pol_agg.select_action({"current_lap": 10, "strategies": self.candidates})
        self.assertEqual(expl_agg.selected_strategy_id, "S_AGGRESSIVE")

        # Conservative policy should prefer S_CONSERVATIVE due to low tyre risk & high robustness
        pol_con = BaselineDecisionPolicy(risk_profile="conservative")
        expl_con = pol_con.select_action({"current_lap": 10, "strategies": self.candidates})
        self.assertEqual(expl_con.selected_strategy_id, "S_CONSERVATIVE")

    def test_explainability_summary(self):
        pol = BaselineDecisionPolicy(risk_profile="balanced")
        expl = pol.select_action({"current_lap": 15, "strategies": self.candidates})
        self.assertIn("selected under balanced profile", expl.summary)
        self.assertIn("Key advantages", expl.summary)
        self.assertIn(expl.selected_strategy_id, expl.top_candidates_utilities)

    def test_fallback_on_empty(self):
        pol = BaselineDecisionPolicy()
        expl = pol.select_action({"current_lap": 10, "strategies": []})
        self.assertEqual(expl.selected_strategy_id, "FALLBACK")
        self.assertEqual(expl.immediate_action, ImmediateAction.STAY_OUT)


class TestStrategyEngineIntegration(unittest.TestCase):
    def test_end_to_end_from_strategy_engine(self):
        # 1. Setup mock RaceState
        cond = CurrentConditions(air_temp=25.0, track_temp=35.0)
        state = RaceState(
            race_id="2024_British_GP",
            current_lap=20,
            total_laps_expected=52,
            current_conditions=cond,
        )
        state.participants["VER"] = ParticipantState(
            driver="VER",
            position=1,
            tyre_life=20.0,
            compound="MEDIUM",
            gap_to_leader_seconds=0.0,
            total_race_time_seconds=1700.0,
            is_active=True,
        )
        state.participants["NOR"] = ParticipantState(
            driver="NOR",
            position=2,
            tyre_life=20.0,
            compound="MEDIUM",
            gap_to_leader_seconds=2.0,
            total_race_time_seconds=1702.0,
            is_active=True,
        )

        # 2. Run StrategyEngine to get evaluated alternatives
        cfg = StrategyEngineConfig(default_horizon_laps=6, default_n_rollouts=15, default_random_seed=42)
        engine = StrategyEngine(config=cfg)
        se_result = engine.evaluate_race_state(state=state, ego_driver="VER")

        self.assertIsInstance(se_result, StrategyEngineResult)
        self.assertGreaterEqual(len(se_result.strategies), 1)

        # 3. Feed directly into Decision Engine BaselinePolicy
        decision_policy = BaselineDecisionPolicy(risk_profile="balanced")
        explanation = decision_policy.select_action(se_result)

        self.assertIsInstance(explanation, DecisionExplanation)
        self.assertIn(explanation.immediate_action, list(ImmediateAction))
        self.assertGreater(len(explanation.top_candidates_utilities), 0)
        self.assertTrue(len(explanation.summary) > 10)


class TestRLComponents(unittest.TestCase):
    def test_reward_calculator(self):
        calc = StrategyRewardCalculator(pit_penalty=-0.05, position_gain_weight=0.1)

        # Intermediate lap: gain 1 position, stay out
        r1 = calc.calculate_step_reward(
            prev_position=3, curr_position=2, is_pit_lap=False, is_terminal=False
        )
        self.assertAlmostEqual(r1, 0.1)

        # Pit lap: lose 2 positions, pit penalty
        r2 = calc.calculate_step_reward(
            prev_position=2, curr_position=4, is_pit_lap=True, is_terminal=False
        )
        # delta = 2 - 4 = -2 -> -0.2 + (-0.05) = -0.25
        self.assertAlmostEqual(r2, -0.25)

        # Terminal lap: win race (P1 = 25 pts -> 1.0)
        r3 = calc.calculate_step_reward(
            prev_position=1, curr_position=1, is_pit_lap=False, is_terminal=True, final_position=1
        )
        self.assertAlmostEqual(r3, 1.0)

    def test_dqn_network_and_policy(self):
        q_net = CandidateConditionedQNetwork(state_dim=13, candidate_dim=9, embed_dim=32, hidden_dim=32)

        # Batch forward test
        batch_size = 4
        k = 5
        s = torch.randn(batch_size, 13)
        c = torch.randn(batch_size, k, 9)
        mask = torch.ones(batch_size, k, dtype=torch.bool)
        mask[0, 3:] = False  # Mask slots 3, 4 for first sample

        q_vals = q_net(s, c, mask)
        self.assertEqual(q_vals.shape, (batch_size, k))
        self.assertEqual(q_vals[0, 3].item(), -1e9)
        self.assertEqual(q_vals[0, 4].item(), -1e9)

        # RL policy inference
        rl_pol = RLDecisionPolicy(q_net=q_net)
        state_vec = np.random.randn(13).astype(np.float32)
        cands_vec = np.random.randn(3, 9).astype(np.float32)

        idx, scores = rl_pol.select_candidate_index(state_vec, cands_vec, epsilon=0.0)
        self.assertIn(idx, [0, 1, 2])
        self.assertEqual(len(scores), 3)

    def test_replay_buffer_dynamic_padding(self):
        buffer = VariableCandidateReplayBuffer(capacity=100)

        # Push transition with 2 candidates
        s1 = np.ones(13, dtype=np.float32)
        c1 = np.ones((2, 9), dtype=np.float32)
        buffer.push(s1, c1, action=0, reward=0.5, next_state=s1, next_candidates=c1, done=False)

        # Push transition with 4 candidates
        c2 = np.ones((4, 9), dtype=np.float32)
        buffer.push(s1, c2, action=2, reward=1.0, next_state=s1, next_candidates=c2, done=True)

        self.assertEqual(len(buffer), 2)

        # Sample batch
        batch = buffer.sample(batch_size=2)
        # Should pad to max_k = 4
        self.assertEqual(batch["candidates"].shape, (2, 4, 9))
        mask_sums = sorted([batch["candidates_mask"][0].sum().item(), batch["candidates_mask"][1].sum().item()])
        self.assertEqual(mask_sums, [2, 4])



class TestGymEnvironmentAndEvaluator(unittest.TestCase):
    def test_gym_environment_step(self):
        # Fast test with small rollouts
        env = F1StrategyEnv(rollouts_per_step=10, horizon_laps=5, seed=42)
        obs, info = env.reset(seed=42)

        self.assertEqual(obs.shape, (13,))
        self.assertIn("candidates", info)
        self.assertGreater(len(info["candidates"]), 0)

        # Take step
        next_obs, reward, terminated, truncated, next_info = env.step(action_idx=0)
        self.assertEqual(next_obs.shape, (13,))
        self.assertIsInstance(reward, float)
        self.assertIn(next_info["immediate_action"], ["STAY_OUT", "PIT_SOFT", "PIT_MEDIUM", "PIT_HARD"])
        self.assertEqual(next_info["current_lap"], 2)

    def test_policy_evaluator(self):
        # Quick 2-episode evaluation
        pol = BaselineDecisionPolicy(risk_profile="balanced")
        evaluator = PolicyEvaluator(
            env_factory=lambda seed: F1StrategyEnv(rollouts_per_step=10, horizon_laps=5, seed=seed)
        )
        res = evaluator.evaluate_policy(pol, policy_name="Baseline_Balanced", n_episodes=2, base_seed=42)

        self.assertIsInstance(res, PolicyEvaluationResult)
        self.assertEqual(res.total_races, 2)
        self.assertEqual(len(res.finishing_positions), 2)
        self.assertIn("Evaluation: Baseline_Balanced", res.summary())

    def test_dqn_trainer_fast(self):
        # Test 1 warm-up episode and 1 training episode
        env = F1StrategyEnv(rollouts_per_step=10, horizon_laps=5, seed=42)
        trainer = DQNTrainer(env=env, batch_size=4)
        policy = trainer.train(n_episodes=1, warm_up_episodes=1)
        self.assertIsInstance(policy, RLDecisionPolicy)



if __name__ == "__main__":
    unittest.main()
