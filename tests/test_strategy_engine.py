"""
tests/test_strategy_engine.py
=============================
Integration tests for the complete Strategy Engine pipeline.

Verifies the strict architectural boundary:
- Strategy Engine produces evaluated strategy alternatives and simulated outcome distributions.
- Strategy Engine does NOT select, recommend, rank as a decision, or declare a "best" strategy.
- Decision Engine receives the evaluated alternatives payload.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.models import CurrentConditions, ParticipantState, RaceState
from strategy_engine.engine import StrategyEngine
from strategy_engine.types import StrategyEngineConfig, StrategyEngineResult


class TestStrategyEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.config = StrategyEngineConfig(
            default_horizon_laps=8,
            default_n_rollouts=30,
            default_random_seed=42,
        )
        self.engine = StrategyEngine(config=self.config)

    def _create_mock_race_state(self, current_lap: int = 22) -> RaceState:
        cond = CurrentConditions(air_temp=25.0, track_temp=35.0, has_safety_car=False, has_vsc=False)
        state = RaceState(
            race_id="2024_British_GP",
            current_lap=current_lap,
            total_laps_expected=52,
            grand_prix="British Grand Prix",
            location="Silverstone",
            current_conditions=cond,
        )

        # Field: P1 NOR, P2 VER (ego), P3 HAM, P4 LEC, P5 PIA
        drivers = [
            ("NOR", "McLaren", 1, 0.0, 1870.0, "MEDIUM", 22.0),
            ("VER", "Red Bull Racing", 2, 1.8, 1871.8, "MEDIUM", 22.0),
            ("HAM", "Mercedes", 3, 4.2, 1874.2, "MEDIUM", 22.0),
            ("LEC", "Ferrari", 4, 8.5, 1878.5, "HARD", 12.0),
            ("PIA", "McLaren", 5, 12.0, 1882.0, "HARD", 15.0),
        ]

        for d_code, team, pos, gap, ctime, comp, life in drivers:
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
                last_lap_time_seconds=85.5,
                is_active=True,
            )

        return state

    def test_full_pipeline_from_race_state(self):
        """Full execution from RaceState to StrategyEngineResult returning evaluated alternatives."""
        state = self._create_mock_race_state(current_lap=22)

        result = self.engine.evaluate_race_state(
            state=state,
            ego_driver="VER",
            horizon_laps=8,
            n_rollouts=30,
            profile="balanced",
            seed=42,
        )

        self.assertIsInstance(result, StrategyEngineResult)
        self.assertEqual(result.ego_driver, "VER")
        self.assertEqual(result.current_lap, 22)
        self.assertEqual(result.horizon_laps, 8)
        self.assertGreater(result.candidate_count, 0)
        self.assertEqual(len(result.strategies), result.candidate_count)

        # Verify each candidate has complete simulated outcome statistics
        for ev in result.strategies:
            self.assertIsNotNone(ev.strategy.strategy_id)
            self.assertIsInstance(ev.strategy.pit_laps, list)
            self.assertIsInstance(ev.strategy.compounds, list)
            sim = ev.simulation_result
            self.assertGreater(sim.n_rollouts, 0)
            self.assertGreater(sim.expected_position, 0.0)
            self.assertGreaterEqual(sim.p_podium, 0.0)
            self.assertLessEqual(sim.p_podium, 1.0)
            self.assertGreaterEqual(sim.p_win, 0.0)
            self.assertLessEqual(sim.p_win, 1.0)
            self.assertGreater(sim.expected_time_seconds, 0.0)

    def test_strict_architectural_boundary_no_recommendation(self):
        """
        Verify that StrategyEngine does NOT contain or return a recommended strategy,
        best strategy, runner-up comparison, or decision-level recommendation.
        """
        state = self._create_mock_race_state(current_lap=22)

        result = self.engine.evaluate_race_state(
            state=state,
            ego_driver="VER",
            horizon_laps=6,
            n_rollouts=25,
            seed=100,
        )

        # Result object must NOT have recommendation or runner-up attributes
        self.assertFalse(hasattr(result, "recommended_strategy"), "StrategyEngineResult must not have recommended_strategy.")
        self.assertFalse(hasattr(result, "runner_up_strategy"), "StrategyEngineResult must not have runner_up_strategy.")
        self.assertFalse(hasattr(result, "score_delta_to_runner_up"), "StrategyEngineResult must not have score_delta_to_runner_up.")
        self.assertFalse(hasattr(result, "recommended_immediate_action"), "StrategyEngineResult must not have recommended_immediate_action.")

        # Summary must NOT declare a strategy recommendation
        self.assertNotIn("is recommended", result.explanation_summary.lower())
        self.assertNotIn("best strategy", result.explanation_summary.lower())

        # Metadata must include boundary notice
        self.assertIn("architectural_boundary_notice", result.metadata)

    def test_decision_engine_payload_structure(self):
        """Downstream payload contains strictly evaluated strategy alternatives and simulation metadata."""
        state = self._create_mock_race_state(current_lap=22)
        result = self.engine.evaluate_race_state(state=state, ego_driver="VER", n_rollouts=20)
        payload = result.to_decision_engine_payload()

        self.assertEqual(payload["current_lap"], 22)
        self.assertEqual(payload["ego_driver"], "VER")

        # Must NOT contain decision-level recommendation fields
        self.assertNotIn("recommended_strategy_id", payload)
        self.assertNotIn("strategy_recommendation_type", payload)
        self.assertNotIn("margin_to_runner_up", payload)

        # Must contain evaluated alternatives and simulation metadata
        self.assertIn("strategies", payload)
        self.assertIn("simulation_metadata", payload)
        self.assertIsInstance(payload["strategies"], list)
        self.assertGreater(len(payload["strategies"]), 0)

        # Verify strategy schema matches requirement
        first_strat = payload["strategies"][0]
        required_keys = [
            "strategy_id", "pit_laps", "compounds", "expected_position",
            "position_std", "expected_race_time", "race_time_std",
            "p_win", "p_podium", "p_top5", "tyre_risk", "traffic_risk",
            "robustness",
        ]
        for k in required_keys:
            self.assertIn(k, first_strat, f"Missing key '{k}' in strategy payload")

        # Verify forbidden ranking/decision fields are strictly absent
        self.assertNotIn("composite_score", first_strat, "composite_score must be removed from production payload.")
        self.assertNotIn("immediate_action", first_strat, "immediate_action must be removed from production payload.")

        # Verify simulation_metadata schema
        meta = payload["simulation_metadata"]
        self.assertIn("rollouts_per_strategy", meta)
        self.assertIn("horizon_laps", meta)
        self.assertIn("seed", meta)

    def test_neutral_deterministic_candidate_ordering(self):
        """Strategies must be returned in deterministic neutral candidate order (S01, S02, ...) not score order."""
        state = self._create_mock_race_state(current_lap=20)
        state.total_laps_expected = 53
        res = self.engine.evaluate_race_state(state=state, ego_driver="VER", n_rollouts=15, seed=42)
        payload = res.to_decision_engine_payload()

        strat_ids = [s["strategy_id"] for s in payload["strategies"]]
        self.assertEqual(strat_ids, sorted(strat_ids), "Strategies must be in natural candidate order S01, S02, ...")

        # Verify across all evaluated strategies that composite_score and immediate_action are absent
        for s in payload["strategies"]:
            self.assertNotIn("composite_score", s)
            self.assertNotIn("immediate_action", s)
            self.assertNotIn("rank", s)

        for ev in res.strategies:
            self.assertFalse(hasattr(ev, "composite_score"))
            self.assertFalse(hasattr(ev, "rank"))
            self.assertFalse(hasattr(ev.strategy, "immediate_action"))

    def test_receding_horizon_lap_stepping(self):
        """Simulate receding-horizon stepping across two consecutive laps."""
        state_lap22 = self._create_mock_race_state(current_lap=22)
        res_22 = self.engine.evaluate_race_state(state=state_lap22, ego_driver="VER", n_rollouts=20, seed=42)

        # Advance to Lap 23
        state_lap23 = self._create_mock_race_state(current_lap=23)
        for p in state_lap23.participants.values():
            p.tyre_life = (p.tyre_life or 0.0) + 1.0

        res_23 = self.engine.evaluate_race_state(state=state_lap23, ego_driver="VER", n_rollouts=20, seed=42)

        self.assertEqual(res_22.current_lap, 22)
        self.assertEqual(res_23.current_lap, 23)
        self.assertGreater(len(res_22.strategies), 0)
        self.assertGreater(len(res_23.strategies), 0)

    def test_full_remaining_race_horizon_derivation(self):
        """
        Verify the production Strategy Engine derives full remaining-race horizon from RaceState:
        - Lap 20 / 53-lap race -> 33-lap horizon
        - Lap 30 / 53-lap race -> 23-lap horizon
        - Lap 52 / 53-lap race -> 1-lap horizon
        - No 10-lap truncation in default production behavior
        - Simulation reaches race completion (current_lap + horizon_laps == total_laps_expected)
        """
        # Production engine with default configuration (no hardcoded horizon)
        prod_engine = StrategyEngine()
        self.assertIsNone(prod_engine.default_horizon, "Production default_horizon must be None (full remaining race).")

        # Test case 1: Lap 20 of 53
        state_20 = self._create_mock_race_state(current_lap=20)
        state_20.total_laps_expected = 53
        res_20 = prod_engine.evaluate_race_state(state=state_20, ego_driver="VER", n_rollouts=10, seed=42)
        self.assertEqual(res_20.horizon_laps, 33, "Lap 20/53 must produce a 33-lap simulation horizon.")
        self.assertEqual(res_20.current_lap + res_20.horizon_laps, 53, "Simulation must reach race completion.")
        self.assertNotEqual(res_20.horizon_laps, 10, "Production must not truncate to 10 laps.")

        # Test case 2: Lap 30 of 53
        state_30 = self._create_mock_race_state(current_lap=30)
        state_30.total_laps_expected = 53
        res_30 = prod_engine.evaluate_race_state(state=state_30, ego_driver="VER", n_rollouts=10, seed=42)
        self.assertEqual(res_30.horizon_laps, 23, "Lap 30/53 must produce a 23-lap simulation horizon.")
        self.assertEqual(res_30.current_lap + res_30.horizon_laps, 53, "Simulation must reach race completion.")

        # Test case 3: Lap 52 of 53
        state_52 = self._create_mock_race_state(current_lap=52)
        state_52.total_laps_expected = 53
        res_52 = prod_engine.evaluate_race_state(state=state_52, ego_driver="VER", n_rollouts=10, seed=42)
        self.assertEqual(res_52.horizon_laps, 1, "Lap 52/53 must produce a 1-lap simulation horizon.")
        self.assertEqual(res_52.current_lap + res_52.horizon_laps, 53, "Simulation must reach race completion.")

    def test_race_finish_metric_semantics(self):
        """
        Verify that production Strategy Engine metrics represent race-completion outcomes:
        - expected_race_time covers the full race distance
        - expected_position, p_podium, and p_top5 evaluate the checkered flag order
        """
        prod_engine = StrategyEngine()
        state = self._create_mock_race_state(current_lap=20)
        state.total_laps_expected = 53

        res = prod_engine.evaluate_race_state(state=state, ego_driver="VER", n_rollouts=15, seed=42)
        self.assertEqual(res.horizon_laps, 33)

        payload = res.to_decision_engine_payload()
        self.assertEqual(payload["simulation_metadata"]["horizon_laps"], 33)

        for s_data in payload["strategies"]:
            # Expected race time must reflect 53 laps of racing (~4500s+), not just 10 laps (~2550s)
            self.assertGreater(s_data["expected_race_time"], 4000.0, "expected_race_time must reflect race completion.")
            self.assertGreaterEqual(s_data["expected_position"], 1.0)
            self.assertLessEqual(s_data["expected_position"], 20.0)
            self.assertGreaterEqual(s_data["p_podium"], 0.0)
            self.assertLessEqual(s_data["p_podium"], 1.0)
            self.assertGreaterEqual(s_data["p_top5"], 0.0)
            self.assertLessEqual(s_data["p_top5"], 1.0)

    def test_independent_candidate_evaluation(self):
        """Different candidate strategies must remain independently evaluated without selection."""
        prod_engine = StrategyEngine()
        state = self._create_mock_race_state(current_lap=20)
        state.total_laps_expected = 53

        res = prod_engine.evaluate_race_state(state=state, ego_driver="VER", n_rollouts=15, seed=42)
        strategies = res.strategies
        self.assertGreater(len(strategies), 1, "Must generate multiple candidate alternatives.")

        # Candidate IDs must be distinct
        strat_ids = [s.strategy.strategy_id for s in strategies]
        self.assertEqual(len(strat_ids), len(set(strat_ids)), "Strategy IDs must be unique.")

        # Verify no selection has occurred
        self.assertFalse(hasattr(res, "best_strategy"))
        self.assertFalse(hasattr(res, "recommended_strategy"))
        self.assertFalse(hasattr(res, "selected_strategy"))

    def test_seed_reproducibility(self):
        """Same seed must produce 100% reproducible numerical evaluations."""
        prod_engine = StrategyEngine()
        state = self._create_mock_race_state(current_lap=20)
        state.total_laps_expected = 53

        res1 = prod_engine.evaluate_race_state(state=state, ego_driver="VER", n_rollouts=15, seed=12345)
        res2 = prod_engine.evaluate_race_state(state=state, ego_driver="VER", n_rollouts=15, seed=12345)

        for s1, s2 in zip(res1.strategies, res2.strategies):
            self.assertEqual(s1.strategy.strategy_id, s2.strategy.strategy_id)
            self.assertEqual(s1.simulation_result.expected_position, s2.simulation_result.expected_position)
            self.assertEqual(s1.simulation_result.expected_time_seconds, s2.simulation_result.expected_time_seconds)
            self.assertEqual(s1.simulation_result.p_podium, s2.simulation_result.p_podium)
            self.assertEqual(s1.simulation_result.p_top5, s2.simulation_result.p_top5)

    def test_factory_load_with_default_models(self):
        """Verify StrategyEngine.load_with_default_models wires cleanly without crashing."""
        engine = StrategyEngine.load_with_default_models()
        self.assertIsNotNone(engine.candidate_generator)
        self.assertIsNotNone(engine.simulator)
        self.assertIsNotNone(engine.evaluator)
        self.assertIsNotNone(engine.formatter)
        self.assertIsNotNone(engine.simulator.overtake_adapter)
        self.assertIsNotNone(engine.simulator.pitstop_adapter)


if __name__ == "__main__":
    unittest.main()
