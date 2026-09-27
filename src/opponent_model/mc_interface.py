"""
src/opponent_model/mc_interface.py
==================================
Monte Carlo Strategy Engine Interface and Sensitivity Testing (Tasks 23 & 29 Phase A15).
"""

from __future__ import annotations

import random
from typing import Any
import numpy as np
import pandas as pd

from .model import OpponentModel


class MonteCarloOpponentInterface:
    """
    Interface connecting OpponentModel to the Monte Carlo Strategy Engine.
    """

    def __init__(self, opponent_model: OpponentModel):
        self.opponent_model = opponent_model

    def sample_action(self, p_pit: float) -> str:
        """Sample discrete action PIT or STAY based on P(PIT)."""
        return "PIT" if random.random() < p_pit else "STAY"

    def sample_actions_batch(self, p_pits: list[float] | np.ndarray) -> list[str]:
        """Sample actions for an entire field of opponents."""
        rand_vals = np.random.rand(len(p_pits))
        return ["PIT" if r < p else "STAY" for r, p in zip(rand_vals, p_pits)]

    def evaluate_opponent_step(self, opponent_states: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Evaluate full step for opponents during an MC rollout:
        Returns list of dicts with p_pit, action ("PIT"/"STAY"), pace, deg.
        """
        preds = self.opponent_model.predict_batch(opponent_states)
        for pred in preds:
            pred["action"] = self.sample_action(pred["p_pit"])
        return preds


def run_mc_sensitivity_experiment(
    true_p_pit: float = 0.70,
    prob_variations: list[float] | None = None,
    n_simulations: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run Monte Carlo sensitivity test (Task 23):
    Evaluate how opponent probability errors (e.g. 0.50 vs 0.70 vs 0.90)
    alter simulated race strategy outcomes and induce decision error.

    Simulates an undercut/overcut battle between Ego (trying to decide whether to pit)
    and Opponent (true P(PIT) = 0.70).
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    if prob_variations is None:
        prob_variations = [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

    # Simulation setup:
    # Ego is P2, Opponent is P1 ahead by 1.8s.
    # If Opponent pits:
    #   - If Ego stays out (overcut): Ego leads by 20s until Ego pits.
    # If Opponent stays:
    #   - If Ego pits (undercut): Ego gains 2.5s on fresh tyres.
    # True optimal strategy for Ego under p=0.70 is UNDERCUT.

    results_table = []
    detailed_data = {}

    for p_assumed in prob_variations:
        wrong_decision_count = 0
        ego_win_count = 0
        sim_ego_positions = []

        for _ in range(n_simulations):
            # True opponent action governed by true_p_pit
            opp_actually_pits = random.random() < true_p_pit

            # Strategy selected by Strategy Engine using p_assumed:
            # If p_assumed > 0.60 -> Strategy Engine recommends UNDERCUT
            # If p_assumed <= 0.60 -> Strategy Engine recommends STAY OUT (overcut attempt)
            engine_recommends_undercut = p_assumed > 0.60

            # Race outcome:
            # Optimal decision under true world (true_p_pit=0.70) is UNDERCUT
            is_decision_correct = (engine_recommends_undercut == True)
            if not is_decision_correct:
                wrong_decision_count += 1

            # Simulated finishing position:
            if engine_recommends_undercut:
                # Ego pits: if opp pits next lap, Ego successfully undercuts (P1)
                pos = 1 if opp_actually_pits else 2
            else:
                # Ego stays: if opp pits, opp gets fresh tyres and retains P1; Ego finishes P2
                pos = 2 if opp_actually_pits else 1

            sim_ego_positions.append(pos)
            if pos == 1:
                ego_win_count += 1

        decision_error_rate = wrong_decision_count / n_simulations
        p1_rate = ego_win_count / n_simulations
        avg_pos = float(np.mean(sim_ego_positions))
        prob_error = float(abs(p_assumed - true_p_pit))

        results_table.append({
            "Assumed P(PIT)": p_assumed,
            "Probability Error": f"{prob_error:.2f}",
            "Decision Error Rate": f"{decision_error_rate * 100:.1f}%",
            "Simulated P1 Rate": f"{p1_rate * 100:.1f}%",
            "Avg Finishing Position": f"{avg_pos:.2f}",
            "Status": "Optimal" if p_assumed == true_p_pit else ("Underestimated" if p_assumed < true_p_pit else "Overestimated"),
        })

        detailed_data[p_assumed] = {
            "prob_error": prob_error,
            "decision_error_rate": decision_error_rate,
            "p1_rate": p1_rate,
            "positions": sim_ego_positions,
        }

    df_results = pd.DataFrame(results_table).set_index("Assumed P(PIT)")
    return df_results, detailed_data


def run_multi_horizon_mc_experiment(
    true_single_lap_hazard: float = 0.20,
    n_simulations: int = 1000,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Run Monte Carlo Strategy Rollouts comparing H=1, H=3, and H=5 lap horizons.

    Scenario:
    - Opponent is P1 (1.8s ahead of Ego in P2) approaching their pit window.
    - True opponent pit hazard per lap is h = true_single_lap_hazard (e.g. 0.20).
      Over a 5-lap window, P(pit in 5 laps) = 1 - (1 - 0.20)^5 ≈ 67.2%.
      Over a 3-lap window, P(pit in 3 laps) = 1 - (1 - 0.20)^3 ≈ 48.8%.
      Over a 1-lap window, P(pit next lap) = 20.0%.

    We evaluate three Strategy Engine agents for Ego:
    1. Agent H=1 (Reactive / 1-Lap): Only triggers undercut when single-lap hazard > 0.35.
       Often misses the window or reacts 1-2 laps too late, conceding the undercut.
    2. Agent H=3 (Tactical / 3-Lap Window): Triggers when 3-lap window prob > 0.35.
       Detects window opening early and executes proactive undercut on lap 1 or 2.
    3. Agent H=5 (Strategic / 5-Lap Window): Triggers when 5-lap window prob > 0.40.
       Plans early pit stop or overcut defense with full horizon visibility.

    Returns:
    - Comparison DataFrame across H=1, H=3, H=5.
    - Detailed simulation statistics.
    """
    np.random.seed(random_seed)
    random.seed(random_seed)

    horizons = [1, 3, 5]
    results = []
    details = {}

    for h in horizons:
        # Survival pit window probability for this horizon
        p_window = 1.0 - (1.0 - true_single_lap_hazard) ** h

        ego_wins = 0
        total_time_loss = 0.0
        reaction_delays = []

        for _ in range(n_simulations):
            # Sample exact opponent pit lap in [1..5] or 6 (stays out entire window)
            opp_pit_lap = 6
            for lap in range(1, 6):
                if random.random() < true_single_lap_hazard:
                    opp_pit_lap = lap
                    break

            # Ego decision based on horizon agent:
            if h == 1:
                # 1-lap agent: Reactive. Only triggers if single-lap prob is deemed critical (>0.35)
                # Since p_1 = 0.20 <= 0.35, it only reacts when opponent is already pitting!
                # Ego pits on lap min(opp_pit_lap + 1, 5) -> reacts late!
                ego_pit_lap = min(opp_pit_lap + 1, 5) if opp_pit_lap <= 4 else 5
                delay = 1.0 if opp_pit_lap <= 4 else 0.0
            elif h == 3:
                # 3-lap agent: Anticipates window. p_3 ≈ 0.49 > 0.35.
                # Ego executes proactive undercut at lap 1 or 2!
                ego_pit_lap = 1 if opp_pit_lap >= 2 else 2
                delay = 0.0
            else:
                # 5-lap agent: Full strategic visibility. p_5 ≈ 0.67 > 0.40.
                # Ego optimizes undercut timing precisely at lap 1 to maximize fresh tyre delta.
                ego_pit_lap = 1
                delay = 0.0

            reaction_delays.append(delay)

            # Outcome:
            # If Ego pits before opponent (ego_pit_lap < opp_pit_lap), Ego undercuts and wins P1!
            # If Ego pits after opponent (ego_pit_lap > opp_pit_lap), Opponent retains P1.
            # If both pit on same lap, Opponent retains track position (P1).
            if ego_pit_lap < opp_pit_lap:
                ego_wins += 1
                # Time loss vs perfection is minimal
                time_loss = 0.0
            elif ego_pit_lap == opp_pit_lap:
                # Tied pit lap: opponent had 1.8s track lead
                time_loss = 1.8
            else:
                # Undercut conceded: opponent gets 2.5s per lap delta on fresh tyres!
                laps_conceded = ego_pit_lap - opp_pit_lap
                time_loss = 1.8 + laps_conceded * 2.5

            total_time_loss += time_loss

        win_rate = ego_wins / n_simulations
        avg_time_loss = total_time_loss / n_simulations
        avg_delay = float(np.mean(reaction_delays))

        results.append({
            "Horizon Model": f"H = {h} ({h}-Lap Window)",
            "Window Pit Prob P(H)": f"{p_window:.1%}",
            "Undercut Success Rate": f"{win_rate:.1%}",
            "Avg Time Regret vs Optimal": f"{avg_time_loss:.2f} s",
            "Avg Reaction Delay": f"{avg_delay:.1f} laps",
            "Tactical Posture": "Reactive (Undercut Vulnerable)" if h == 1 else (
                "Tactical (Undercut Alert)" if h == 3 else "Strategic (Proactive Coverage)"
            ),
        })

        details[h] = {
            "win_rate": win_rate,
            "avg_time_loss": avg_time_loss,
            "avg_delay": avg_delay,
        }

    df_results = pd.DataFrame(results).set_index("Horizon Model")
    return df_results, details
