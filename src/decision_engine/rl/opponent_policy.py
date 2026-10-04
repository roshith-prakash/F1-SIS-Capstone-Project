"""
Opponent Wall AI Policies for F1-SIS RL Environment.
Simulates reactive pit strategies (undercut covering, overcut defense, stint management)
with randomized driver/team personalities.
"""

from __future__ import annotations
import random
from typing import Any


class ReactiveOpponentPolicy:
    """
    Simulates reactive pit-wall decisions of opponent teams.
    Reacts to ego driver actions (undercut attempts) and tyre degradation.
    """

    def __init__(
        self,
        personality: str = "balanced",
        reaction_probability: float | None = None,
        nominal_pit_lap: int | None = None,
    ):
        self.personality = personality.lower()
        if reaction_probability is not None:
            self.reaction_prob = reaction_probability
        elif self.personality == "aggressive":
            self.reaction_prob = 0.85
        elif self.personality == "conservative":
            self.reaction_prob = 0.30
        else:  # balanced
            self.reaction_prob = 0.60

        self.nominal_pit_lap = nominal_pit_lap
        self.under_threat_of_undercut = False

    def decide_pit(
        self,
        opp_driver: str,
        opp_life: float,
        opp_compound: str,
        opp_gap_ahead: float,
        opp_gap_behind: float,
        current_lap: int,
        total_laps: int,
        ego_driver: str,
        ego_pitted_this_lap: bool,
        is_sc: bool = False,
        py_rng: random.Random | None = None,
    ) -> bool:
        """
        Decides whether this opponent car should pit on the current lap.
        """
        rng = py_rng or random.Random()
        comp = opp_compound.upper()

        # Near end of race: do not pit
        if current_lap >= total_laps - 4:
            return False

        # Safety car opportunism
        if is_sc:
            sc_pit_prob = 0.80 if self.personality == "aggressive" else 0.50 if self.personality == "balanced" else 0.35
            if opp_life >= 8.0 and rng.random() < sc_pit_prob:
                return True

        # Nominal tyre life threshold
        nominal_life = 32 if comp == "MEDIUM" else 20 if comp == "SOFT" else 42

        # 1. Undercut defense reaction:
        # If ego just pitted or attempted undercut, and opponent is nearby (within 3.0s)
        # with tyres that have done at least 10 laps
        if ego_pitted_this_lap and opp_gap_behind < 3.5 and opp_life >= 10.0:
            if rng.random() < self.reaction_prob:
                return True

        # 2. Tyre cliff enforcement
        if opp_life >= nominal_life:
            cliff_pit_prob = 0.90 if self.personality == "aggressive" else 0.75
            if rng.random() < cliff_pit_prob:
                return True

        # 3. Scheduled nominal pit window
        if self.nominal_pit_lap is not None:
            if abs(current_lap - self.nominal_pit_lap) <= 1 and opp_life >= 12.0:
                return True

        # 4. Routine wear probability
        if opp_life > nominal_life - 5:
            wear_risk = (opp_life - (nominal_life - 5)) / 10.0
            if rng.random() < wear_risk * self.reaction_prob:
                return True

        return False


def create_random_opponent_policies(
    drivers: list[str],
    rng: random.Random | None = None,
) -> dict[str, ReactiveOpponentPolicy]:
    """
    Creates a varied field of opponent personalities (aggressive, balanced, conservative).
    """
    r = rng or random.Random()
    profiles = ["aggressive", "balanced", "conservative"]
    policies = {}
    for d in drivers:
        prof = r.choice(profiles)
        policies[d] = ReactiveOpponentPolicy(personality=prof)
    return policies
