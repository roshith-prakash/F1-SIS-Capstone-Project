"""
tests/test_strategy_evaluator.py
================================
Automated unit tests for Strategy Evaluator and Risk Profiles.
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from strategy_engine.evaluator import StrategyEvaluator
from strategy_engine.types import (
    SimulationResult,
    Strategy,
    StrategyEngineConfig,
)


class TestStrategyEvaluator(unittest.TestCase):
    def setUp(self):
        self.config = StrategyEngineConfig()
        self.evaluator = StrategyEvaluator(config=self.config)

    def _make_dummy_sim_result(
        self,
        strategy_id: str,
        expected_pos: float = 2.0,
        std_pos: float = 0.5,
        p90_pos: float = 3.0,
        expected_time: float = 5200.0,
        p_podium: float = 0.85,
        tyre_risk: float = 0.20,
        sc_advantage: float = 0.5,
    ) -> SimulationResult:
        return SimulationResult(
            strategy_id=strategy_id,
            n_rollouts=100,
            horizon_laps=10,
            expected_position=expected_pos,
            median_position=expected_pos,
            std_position=std_pos,
            position_p10=1.0,
            position_p90=p90_pos,
            p_win=0.30,
            p_podium=p_podium,
            p_top5=0.95,
            p_points=1.0,
            p_gain_positions=0.40,
            expected_time_seconds=expected_time,
            std_time_seconds=4.0,
            p_sc_affected=0.15,
            expected_position_under_sc=expected_pos - sc_advantage,
            expected_position_green=expected_pos,
            sc_position_advantage=sc_advantage,
            tyre_risk=tyre_risk,
            traffic_risk=0.15,
            robustness=0.85,
        )

    def test_evaluation_packaging_and_genuine_metrics(self):
        """StrategyEvaluator packages genuine Monte Carlo outcome and risk metrics without composite score."""
        strat = Strategy(strategy_id="S01", name="Test", num_stops=1, pit_laps=[25], compounds=["HARD"])
        sim_res = self._make_dummy_sim_result("S01")

        ev = self.evaluator.evaluate_strategy(strat, sim_res, profile_name="balanced")
        self.assertEqual(ev.strategy.strategy_id, "S01")
        self.assertEqual(ev.simulation_result.expected_position, 2.0)
        self.assertEqual(ev.simulation_result.std_position, 0.5)
        self.assertEqual(ev.simulation_result.expected_time_seconds, 5200.0)
        self.assertEqual(ev.simulation_result.p_podium, 0.85)
        self.assertEqual(ev.simulation_result.tyre_risk, 0.20)
        self.assertEqual(ev.simulation_result.traffic_risk, 0.15)
        self.assertEqual(ev.simulation_result.robustness, 0.85)

        # Must NOT expose composite score, rank, or immediate action
        self.assertFalse(hasattr(ev, "composite_score"), "StrategyEvaluation must not expose composite_score")
        self.assertFalse(hasattr(ev, "rank"), "StrategyEvaluation must not expose rank")
        self.assertFalse(hasattr(ev.strategy, "immediate_action"), "Strategy must not expose immediate_action")

    def test_evaluator_returns_deterministic_neutral_order(self):
        """
        Evaluations must be returned in neutral deterministic candidate order (S01, S02, S03)
        regardless of their expected positions or times. No ranking or score sorting.
        """
        strat1 = Strategy(strategy_id="S01", name="S1", num_stops=1, pit_laps=[20], compounds=["HARD"])
        strat2 = Strategy(strategy_id="S02", name="S2", num_stops=1, pit_laps=[24], compounds=["HARD"])
        strat3 = Strategy(strategy_id="S03", name="S3", num_stops=2, pit_laps=[15, 35], compounds=["MEDIUM", "SOFT"])

        sims = {
            "S01": self._make_dummy_sim_result("S01", expected_pos=4.0),
            "S02": self._make_dummy_sim_result("S02", expected_pos=1.5),  # Best position, but must not be placed first
            "S03": self._make_dummy_sim_result("S03", expected_pos=2.5),
        }

        evals = self.evaluator.evaluate_all([strat1, strat2, strat3], sims, profile_name="balanced")
        self.assertEqual(len(evals), 3)

        # Must maintain neutral candidate ID order: S01, S02, S03
        returned_ids = [e.strategy.strategy_id for e in evals]
        self.assertEqual(returned_ids, ["S01", "S02", "S03"], "Must return in neutral candidate order S01, S02, S03.")

        # Must NOT have rank or composite score
        for e in evals:
            self.assertFalse(hasattr(e, "rank"))
            self.assertFalse(hasattr(e, "composite_score"))

    def test_empty_input_handling(self):
        """Evaluating empty strategy or simulation lists should return empty without error."""
        self.assertEqual(self.evaluator.evaluate_all([], {}), [])


if __name__ == "__main__":
    unittest.main()
