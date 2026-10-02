"""
tests/test_strategy_simulator.py
================================
Automated unit tests for Race Scenario Simulator (Monte Carlo Engine).
"""

from __future__ import annotations

import unittest
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from race_state.models import ParticipantState, RaceState
from strategy_engine.simulator import RaceScenarioSimulator
from strategy_engine.types import Strategy, StrategyEngineConfig, StintPlan


class MockOpponentInterface:
    """Mock interface tracking calls to sample_action."""
    def __init__(self):
        self.call_count = 0
        self.sampled_actions = []

    def sample_action(self, p_pit: float) -> str:
        self.call_count += 1
        # Deterministic toggle based on probability threshold
        action = "PIT" if p_pit > 0.40 else "STAY"
        self.sampled_actions.append(action)
        return action


class TestRaceScenarioSimulator(unittest.TestCase):
    def setUp(self):
        self.config = StrategyEngineConfig(
            default_horizon_laps=10,
            default_n_rollouts=50,
            pit_loss_green_seconds=22.0,
            pit_loss_sc_seconds=12.0,
            lap_time_noise_std=0.20,
        )
        self.mock_opp = MockOpponentInterface()
        self.simulator = RaceScenarioSimulator(
            config=self.config,
            opponent_interface=self.mock_opp,
            default_rollouts=50,
            random_seed=42,
        )

    def _create_sample_state(self, current_lap: int = 20, total_laps: int = 57) -> RaceState:
        state = RaceState(
            current_lap=current_lap,
            total_laps_expected=total_laps,
            grand_prix="Italian Grand Prix",
            location="Monza",
        )
        # Ego car (VER, P2)
        state.participants["VER"] = ParticipantState(
            driver="VER",
            team="Red Bull Racing",
            position=2,
            gap_to_leader_seconds=2.0,
            total_race_time_seconds=1702.0,
            compound="MEDIUM",
            tyre_life=20.0,
            stint=1,
            pit_count=0,
            last_lap_time_seconds=85.0,
            is_active=True,
        )
        # Leader (NOR, P1)
        state.participants["NOR"] = ParticipantState(
            driver="NOR",
            team="McLaren",
            position=1,
            gap_to_leader_seconds=0.0,
            total_race_time_seconds=1700.0,
            compound="MEDIUM",
            tyre_life=20.0,
            stint=1,
            pit_count=0,
            last_lap_time_seconds=84.9,
            is_active=True,
        )
        # Rival behind (LEC, P3)
        state.participants["LEC"] = ParticipantState(
            driver="LEC",
            team="Ferrari",
            position=3,
            gap_to_leader_seconds=4.5,
            total_race_time_seconds=1704.5,
            compound="MEDIUM",
            tyre_life=20.0,
            stint=1,
            pit_count=0,
            last_lap_time_seconds=85.2,
            is_active=True,
        )
        return state

    def test_fixed_seed_reproducibility(self):
        """Simulation results must be 100% identical when seed is fixed."""
        state = self._create_sample_state()
        strat = Strategy(
            strategy_id="S01",
            name="1-Stop Medium->Hard",
            num_stops=1,
            pit_laps=[25],
            compounds=["HARD"],
        )

        res1 = self.simulator.simulate_strategy(state, "VER", strat, horizon_laps=10, n_rollouts=40, seed=12345)
        res2 = self.simulator.simulate_strategy(state, "VER", strat, horizon_laps=10, n_rollouts=40, seed=12345)

        self.assertEqual(res1.expected_position, res2.expected_position)
        self.assertEqual(res1.std_position, res2.std_position)
        self.assertEqual(res1.expected_time_seconds, res2.expected_time_seconds)
        self.assertEqual(res1.p_podium, res2.p_podium)
        self.assertEqual(len(res1.rollouts), len(res2.rollouts))
        for r1, r2 in zip(res1.rollouts, res2.rollouts):
            self.assertEqual(r1.final_position, r2.final_position)
            self.assertEqual(r1.total_race_time_seconds, r2.total_race_time_seconds)

    def test_different_seeds_produce_variance(self):
        """Different seeds should generate statistically varying outcomes."""
        state = self._create_sample_state()
        strat = Strategy(
            strategy_id="S01",
            name="1-Stop Medium->Hard",
            num_stops=1,
            pit_laps=[22],
            compounds=["HARD"],
        )

        res_a = self.simulator.simulate_strategy(state, "VER", strat, horizon_laps=10, n_rollouts=50, seed=111)
        res_b = self.simulator.simulate_strategy(state, "VER", strat, horizon_laps=10, n_rollouts=50, seed=999)

        # Total times or position distributions should differ slightly due to stochastic noise
        self.assertNotEqual(res_a.expected_time_seconds, res_b.expected_time_seconds)

    def test_rollout_count_matches(self):
        """Simulator generates exactly the requested number of rollouts."""
        state = self._create_sample_state()
        strat = Strategy(strategy_id="S01", name="Test", num_stops=1, pit_laps=[24], compounds=["HARD"])

        for count in [25, 60]:
            res = self.simulator.simulate_strategy(state, "VER", strat, n_rollouts=count, seed=42)
            self.assertEqual(res.n_rollouts, count)
            self.assertEqual(len(res.rollouts), count)

    def test_tyre_age_progression_and_reset(self):
        """Tyre age must advance on STAY laps and reset to 0 after pitting."""
        state = self._create_sample_state(current_lap=20)

        # Strategy A: Pit on lap 21
        strat_pit_now = Strategy(
            strategy_id="PIT_NOW",
            name="Pit Now",
            num_stops=1,
            pit_laps=[21],
            compounds=["HARD"],
        )
        res_pit = self.simulator.simulate_strategy(state, "VER", strat_pit_now, horizon_laps=5, n_rollouts=20, seed=42)

        # In 5 laps, pitting on lap 21 means tyre age at end should be ~4 laps (0 on lap 21, +1 for laps 22, 23, 24, 25)
        self.assertTrue(all(r.final_tyre_age <= 5 for r in res_pit.rollouts))

        # Strategy B: Stay out all 5 laps
        strat_stay = Strategy(
            strategy_id="STAY_OUT",
            name="Stay Out",
            num_stops=0,
            pit_laps=[],
            compounds=[],
        )
        res_stay = self.simulator.simulate_strategy(state, "VER", strat_stay, horizon_laps=5, n_rollouts=20, seed=42)
        # Initial tyre age was 20. Under clean green laps, advances to 25. Under SC/VSC, wear is discounted (< 25).
        for r in res_stay.rollouts:
            if not r.sc_deployed and not r.vsc_deployed:
                self.assertEqual(r.final_tyre_age, 25)
            else:
                self.assertLess(r.final_tyre_age, 25)

    def test_opponent_action_sampling(self):
        """Opponent interface is invoked during rollout execution."""
        state = self._create_sample_state()
        strat = Strategy(strategy_id="S01", name="Test", num_stops=1, pit_laps=[24], compounds=["HARD"])

        self.mock_opp.call_count = 0
        self.simulator.simulate_strategy(state, "VER", strat, horizon_laps=5, n_rollouts=10, seed=42)

        self.assertGreater(self.mock_opp.call_count, 0)
        self.assertIn("STAY", self.mock_opp.sampled_actions)

    def test_no_future_information_leakage(self):
        """
        Simulator must only condition on the state at lap t.
        Simulating lap t should be independent of any future external state.
        """
        state_a = self._create_sample_state(current_lap=20)
        strat = Strategy(strategy_id="S01", name="Test", num_stops=1, pit_laps=[24], compounds=["HARD"])

        res_a = self.simulator.simulate_strategy(state_a, "VER", strat, horizon_laps=8, n_rollouts=30, seed=777)

        # Construct identical state at lap 20, but with different historical metadata that is unrelated
        state_b = self._create_sample_state(current_lap=20)
        state_b.lap_history = [{"dummy": "unrelated_past_data"}]

        res_b = self.simulator.simulate_strategy(state_b, "VER", strat, horizon_laps=8, n_rollouts=30, seed=777)

        self.assertEqual(res_a.expected_position, res_b.expected_position)
        self.assertEqual(res_a.expected_time_seconds, res_b.expected_time_seconds)

    def test_default_horizon_derives_full_remaining_race(self):
        """
        When horizon_laps is None, the simulator must derive the horizon as
        remaining_laps = total_laps_expected - current_lap and reach race completion.
        """
        config = StrategyEngineConfig(default_n_rollouts=10)
        sim = RaceScenarioSimulator(config=config, random_seed=42)

        # 1. Lap 20 of 53 -> 33 laps remaining
        state_20 = self._create_sample_state(current_lap=20, total_laps=53)
        strat = Strategy(strategy_id="S01", name="Test", num_stops=1, pit_laps=[25], compounds=["HARD"])
        res_20 = sim.simulate_strategy(state_20, "VER", strat, horizon_laps=None, n_rollouts=10)
        self.assertEqual(res_20.horizon_laps, 33)
        self.assertIsNotNone(state_20.current_lap)
        self.assertEqual((state_20.current_lap or 0) + res_20.horizon_laps, 53)

        # 2. Lap 30 of 53 -> 23 laps remaining
        state_30 = self._create_sample_state(current_lap=30, total_laps=53)
        res_30 = sim.simulate_strategy(state_30, "VER", strat, horizon_laps=None, n_rollouts=10)
        self.assertEqual(res_30.horizon_laps, 23)
        self.assertIsNotNone(state_30.current_lap)
        self.assertEqual((state_30.current_lap or 0) + res_30.horizon_laps, 53)

        # 3. Lap 52 of 53 -> 1 lap remaining
        state_52 = self._create_sample_state(current_lap=52, total_laps=53)
        res_52 = sim.simulate_strategy(state_52, "VER", strat, horizon_laps=None, n_rollouts=10)
        self.assertEqual(res_52.horizon_laps, 1)
        self.assertIsNotNone(state_52.current_lap)
        self.assertEqual((state_52.current_lap or 0) + res_52.horizon_laps, 53)

    def test_pitstop_adapter_integration_in_simulator(self):
        """Verify that PitstopAdapter dynamically drives pit time loss in simulator rollouts."""
        try:
            from src.pitstop.adapter import PitstopAdapter
        except ImportError:
            from pitstop.adapter import PitstopAdapter

        adapter = PitstopAdapter()
        config = StrategyEngineConfig(default_n_rollouts=10)
        sim = RaceScenarioSimulator(config=config, pitstop_adapter=adapter, random_seed=42)

        # Candidate strategy that pits on lap 21
        strat_pit = Strategy(
            strategy_id="PIT_S01",
            name="Pit Stop",
            num_stops=1,
            pit_laps=[21],
            compounds=["MEDIUM", "HARD"],
            stints=[
                StintPlan(stint_number=1, compound="MEDIUM", target_laps=20, start_lap=1, end_lap=20),
                StintPlan(stint_number=2, compound="HARD", target_laps=10, start_lap=21, end_lap=30),
            ]
        )

        # 1. Simulate on Melbourne (fast pit lane: ~18s)
        state_melb = self._create_sample_state(current_lap=20, total_laps=30)
        state_melb.location = "Melbourne"
        state_melb.grand_prix = "Australian Grand Prix"
        res_melb = sim.simulate_strategy(state_melb, "VER", strat_pit, horizon_laps=5, n_rollouts=15, seed=42)

        # 2. Simulate on Silverstone (slow pit lane: ~30s)
        state_silver = self._create_sample_state(current_lap=20, total_laps=30)
        state_silver.location = "Silverstone"
        state_silver.grand_prix = "British Grand Prix"
        res_silver = sim.simulate_strategy(state_silver, "VER", strat_pit, horizon_laps=5, n_rollouts=15, seed=42)

        # Silverstone race time must be strictly longer due to physical pit lane length (~11s difference)
        self.assertGreater(
            res_silver.expected_time_seconds,
            res_melb.expected_time_seconds + 5.0,
            "Silverstone pit stop must take substantially longer than Melbourne in simulator.",
        )

    def test_safety_car_wear_discount_and_lap_time_delta(self):
        """Verify empirical SC (+40% base, 0.50 wear, 0.50 deg) and VSC (+24% base, 0.70 wear, 0.70 deg) logic."""
        rng = np.random.default_rng(42)
        car = {
            "driver": "VER",
            "team": "Red Bull Racing",
            "compound": "MEDIUM",
            "tyre_age": 10.0,
            "cumulative_race_time": 1000.0,
            "last_lap_time": 85.0,
            "position": 1,
            "pit_count": 0,
        }
        state = self._create_sample_state()

        # 1. Base 85.0s circuit
        lt_green = self.simulator._estimate_lap_pace(
            car=car.copy(), lap_num=21, total_laps=57, circuit_base=85.0,
            is_sc=False, is_vsc=False, is_pitting=False, rng=rng, state=state,
        )

        # SC pace: caution_delta = 85.0 * 0.40 = 34.0s, deg suppressed by 0.50
        rng_sc = np.random.default_rng(42)
        lt_sc = self.simulator._estimate_lap_pace(
            car=car.copy(), lap_num=21, total_laps=57, circuit_base=85.0,
            is_sc=True, is_vsc=False, is_pitting=False, rng=rng_sc, state=state,
        )
        self.assertGreater(lt_sc - lt_green, 33.0)
        self.assertLess(lt_sc - lt_green, 35.0)

        # VSC pace: caution_delta = 85.0 * 0.24 = 20.4s, deg suppressed by 0.70
        rng_vsc = np.random.default_rng(42)
        lt_vsc = self.simulator._estimate_lap_pace(
            car=car.copy(), lap_num=21, total_laps=57, circuit_base=85.0,
            is_sc=False, is_vsc=True, is_pitting=False, rng=rng_vsc, state=state,
        )
        self.assertGreater(lt_vsc - lt_green, 19.5)
        self.assertLess(lt_vsc - lt_green, 21.5)

        # 2. Verify circuit proportionality: short circuit (Austria 67.5s) vs long circuit (Spa 107.3s)
        rng_austria = np.random.default_rng(42)
        lt_sc_austria = self.simulator._estimate_lap_pace(
            car=car.copy(), lap_num=21, total_laps=57, circuit_base=67.5,
            is_sc=True, is_vsc=False, is_pitting=False, rng=rng_austria, state=state,
        )
        rng_austria_green = np.random.default_rng(42)
        lt_green_austria = self.simulator._estimate_lap_pace(
            car=car.copy(), lap_num=21, total_laps=57, circuit_base=67.5,
            is_sc=False, is_vsc=False, is_pitting=False, rng=rng_austria_green, state=state,
        )

        rng_spa = np.random.default_rng(42)
        lt_sc_spa = self.simulator._estimate_lap_pace(
            car=car.copy(), lap_num=21, total_laps=57, circuit_base=107.3,
            is_sc=True, is_vsc=False, is_pitting=False, rng=rng_spa, state=state,
        )
        rng_spa_green = np.random.default_rng(42)
        lt_green_spa = self.simulator._estimate_lap_pace(
            car=car.copy(), lap_num=21, total_laps=57, circuit_base=107.3,
            is_sc=False, is_vsc=False, is_pitting=False, rng=rng_spa_green, state=state,
        )

        delta_austria = lt_sc_austria - lt_green_austria
        delta_spa = lt_sc_spa - lt_green_spa

        # Austria delta (~27s) must be substantially smaller than Spa delta (~43s)
        self.assertAlmostEqual(delta_austria, 67.5 * 0.40, delta=1.5)
        self.assertAlmostEqual(delta_spa, 107.3 * 0.40, delta=1.5)
        self.assertGreater(delta_spa, delta_austria + 12.0)


if __name__ == "__main__":
    unittest.main()

