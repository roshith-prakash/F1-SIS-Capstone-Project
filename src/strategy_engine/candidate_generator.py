"""
src/strategy_engine/candidate_generator.py
==========================================
Component 1: Candidate Strategy Generator.

Generates feasible, legal, and tactically diverse candidate strategies from the
current race state. Prunes invalid strategies (violating F1 compound regulations,
exceeding tyre wear limits, or having physically infeasible pit windows).

Strict Architectural Boundary:
- Generates multi-lap strategy plans over the planning horizon.
- Identifies the immediate action implication ("PIT_<COMPOUND>" vs "STAY")
  for the future Decision Engine.
"""

from __future__ import annotations

import math
from typing import Any, Optional

from race_state.models import ParticipantState, RaceState, normalize_driver

from .types import StintPlan, Strategy, StrategyEngineConfig


class CandidateStrategyGenerator:
    """
    Generates feasible candidate strategy families (1-stop and 2-stop plans)
    tailored to the current race state.
    """

    def __init__(self, config: StrategyEngineConfig | None = None):
        self.config = config or StrategyEngineConfig()

    def _compute_deg_cliff_excess_loss(self, compound: str, stint_laps: int, start_age: int = 0) -> float:
        """
        Compute pace loss accumulated only in laps BEYOND the cliff reference age.
        Mirrors config.compute_stint_cliff_excess_loss() for use in candidate filtering.
        """
        return self.config.compute_stint_cliff_excess_loss(compound, stint_laps, start_age)

    def _is_stint_viable_hybrid(
        self,
        compound: str,
        stint_laps: int,
        start_age: int = 0,
    ) -> tuple[bool, str]:
        """
        Hybrid viability check: determine if a planned stint is viable by simulating
        tyre degradation BEYOND the compound's cliff reference age.

        A stint is viable if the pace loss accumulated in laps PAST the cliff reference
        does NOT exceed `config.deg_cliff_excess_threshold_seconds`.

        This correctly allows long-but-viable stints (e.g. HARD 39 laps stays within
        cliff window → 0s excess → valid) while rejecting truly cliff-prone ones
        (e.g. SOFT 30 laps → 6 laps beyond cliff → ~16.9s excess → invalid).

        If dynamic_tyre_viability_check is disabled, falls back to the static max-age gate.
        """
        if not self.config.dynamic_tyre_viability_check:
            # Legacy static-gate fallback
            max_age = self.config.get_max_tyre_age(compound)
            if start_age + stint_laps > max_age:
                return False, (
                    f"Static gate: {compound} stint of {stint_laps} laps (starting age {start_age}) "
                    f"would reach age {start_age + stint_laps}, exceeding cliff reference {max_age}."
                )
            return True, "OK"

        # Hybrid: cliff-excess degradation check
        cliff_excess = self._compute_deg_cliff_excess_loss(compound, stint_laps, start_age)
        threshold = self.config.deg_cliff_excess_threshold_seconds
        if cliff_excess > threshold:
            cliff_age = self.config.get_max_tyre_age(compound)
            laps_beyond = max(0, (start_age + stint_laps) - cliff_age)
            return False, (
                f"Hybrid check: {compound} stint of {stint_laps} laps (starting age {start_age}) "
                f"runs {laps_beyond} laps beyond cliff (age {cliff_age}), "
                f"accumulating {cliff_excess:.2f}s cliff-excess pace loss (threshold {threshold:.1f}s)."
            )
        return True, "OK"

    def validate_strategy(
        self,
        strategy: Strategy,
        state: RaceState,
        ego_driver: str,
    ) -> tuple[bool, str]:
        """
        Validate physical, regulatory, and temporal feasibility of a strategy plan.
        
        Returns:
            (is_valid, rejection_reason)
        """
        curr_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        driver_code = normalize_driver(ego_driver)
        participant = state.participants.get(driver_code) if driver_code else None

        if curr_lap >= total_laps:
            return False, "Race is already completed."

        current_compound = str(participant.compound if participant and participant.compound else "MEDIUM").upper()
        current_tyre_age = int(participant.tyre_life if participant and participant.tyre_life is not None else 1)
        prior_pit_count = int(participant.pit_count if participant and participant.pit_count is not None else 0)

        # 1. Validate pit laps are strictly chronological and in the future
        for i, pit_lap in enumerate(strategy.pit_laps):
            if pit_lap < curr_lap:
                return False, f"Pit lap {pit_lap} is in the past (current lap {curr_lap})."
            if pit_lap >= total_laps:
                return False, f"Pit lap {pit_lap} is at or beyond race finish {total_laps}."
            if i > 0 and (pit_lap - strategy.pit_laps[i - 1]) < self.config.minimum_stint_length:
                return False, f"Stint between stops ({pit_lap - strategy.pit_laps[i-1]} laps) is shorter than minimum {self.config.minimum_stint_length} laps."

        # 2. Validate current tyre can reach the first stop (hybrid or static)
        first_pit_lap = strategy.pit_laps[0] if strategy.pit_laps else total_laps
        laps_until_first_pit = first_pit_lap - curr_lap

        viable, reason = self._is_stint_viable_hybrid(
            compound=current_compound,
            stint_laps=laps_until_first_pit,
            start_age=current_tyre_age,
        )
        if not viable:
            return False, f"Current tyre ({current_compound}) cannot reach first stop: {reason}"

        # 3. Validate subsequent stint tyre wear (hybrid or static)
        if strategy.num_stops >= 1:
            for i, stint in enumerate(strategy.stints):
                # Subsequent stints always start on fresh (age=0) tyres after a pit stop
                if i > 0:
                    viable, reason = self._is_stint_viable_hybrid(
                        compound=stint.compound,
                        stint_laps=stint.target_laps,
                        start_age=0,  # Fresh tyres after pit stop
                    )
                    if not viable:
                        return False, f"Stint {stint.stint_number} ({stint.compound}): {reason}"

        # 4. F1 Two-Compound Regulation (Dry race)
        if self.config.enforce_f1_two_compound_rule:
            all_compounds_used = {current_compound}
            for comp in strategy.compounds:
                all_compounds_used.add(comp.upper())

            # If 0-stop is proposed, driver must have already completed a pit stop in the race
            if len(strategy.pit_laps) == 0:
                if prior_pit_count == 0:
                    return False, "F1 Regulation violation: Must use at least 2 distinct dry compounds during a dry race."
            else:
                # If driver hasn't pitted yet in the race, the forward plan must use at least 2 compounds
                if prior_pit_count == 0 and len(all_compounds_used) < 2:
                    return False, f"F1 Regulation violation: Strategy uses only 1 compound ({current_compound})."

        return True, "Valid"

    def generate_candidates(
        self,
        state: RaceState,
        ego_driver: str,
        horizon_laps: int | None = None,
        max_stops: int = 2,
    ) -> list[Strategy]:
        """
        Generate feasible candidate strategy families from current race state.
        
        Focuses on tactically meaningful strategy families:
        1. Undercut / Pit Now (Stop immediately on current lap)
        2. Nominal Pit Window (Stop in optimal degradation window)
        3. Stint Extension / Overcut (Extend stint for tyre offset)
        4. Alternative 2-Stop (Aggressive multi-stint pace)
        5. 0-Stop / Run to End (Only if already pitted and compound allows)
        """
        curr_lap = state.current_lap or 1
        total_laps = state.total_laps_expected or 57
        remaining_laps = max(0, total_laps - curr_lap)
        if remaining_laps <= 0:
            return []

        if horizon_laps is not None:
            h_laps = horizon_laps
        elif self.config.default_horizon_laps is not None:
            h_laps = self.config.default_horizon_laps
        else:
            h_laps = max(1, remaining_laps)

        driver_code = normalize_driver(ego_driver) or "EGO"
        participant = state.participants.get(driver_code)

        curr_compound = str(participant.compound if participant and participant.compound else "MEDIUM").upper()
        curr_tyre_age = int(participant.tyre_life if participant and participant.tyre_life is not None else 1)
        prior_pit_count = int(participant.pit_count if participant and participant.pit_count is not None else 0)

        # Usable dry compounds
        available_compounds = [c for c in self.config.dry_compounds if c != curr_compound]
        if not available_compounds:
            available_compounds = ["HARD", "MEDIUM"]

        candidates: list[Strategy] = []
        strat_counter = 1

        # ---------------------------------------------------------------------
        # 1. ONE-STOP STRATEGY FAMILY
        # ---------------------------------------------------------------------
        # Determine nominal pit window based on tyre wear
        max_life = self.config.get_max_tyre_age(curr_compound)
        remaining_life_on_curr = max(0, max_life - curr_tyre_age)

        # Pit lap offsets to test
        pit_lap_offsets: list[tuple[int, str, str]] = []

        # A. Undercut Now: Pit this exact lap
        pit_lap_offsets.append((0, "undercut", "Undercut Now"))

        # B. Nominal Window: Pit in 3-5 laps (or near half of remaining tyre life)
        if remaining_life_on_curr >= 3:
            nominal_offset = min(remaining_life_on_curr - 2, max(2, remaining_life_on_curr // 2))
            if nominal_offset > 0:
                pit_lap_offsets.append((nominal_offset, "nominal", f"Nominal Window (+{nominal_offset} laps)"))

        # C. Stint Extension / Overcut: Pit near the end of current tyre life
        if remaining_life_on_curr >= 6:
            ext_offset = max(1, remaining_life_on_curr - 2)
            if all(ext_offset != o[0] for o in pit_lap_offsets):
                pit_lap_offsets.append((ext_offset, "overcut", f"Overcut Extension (+{ext_offset} laps)"))

        for offset, intent, desc in pit_lap_offsets:
            pit_lap = curr_lap + offset
            if pit_lap >= total_laps:
                continue

            for next_comp in available_compounds:
                # Hybrid check: verify second stint is viable under degradation model
                stint2_length = total_laps - pit_lap
                stint2_viable, _ = self._is_stint_viable_hybrid(
                    compound=next_comp,
                    stint_laps=stint2_length,
                    start_age=0,
                )
                if not stint2_viable:
                    continue

                strat_id = f"S{strat_counter:02d}"

                stints = [
                    StintPlan(
                        stint_number=prior_pit_count + 1,
                        compound=curr_compound,
                        target_laps=curr_tyre_age + offset,
                        start_lap=curr_lap - curr_tyre_age,
                        end_lap=pit_lap,
                    ),
                    StintPlan(
                        stint_number=prior_pit_count + 2,
                        compound=next_comp,
                        target_laps=stint2_length,
                        start_lap=pit_lap,
                        end_lap=total_laps,
                    ),
                ]

                strat = Strategy(
                    strategy_id=strat_id,
                    name=f"1-Stop: {curr_compound[0]}->{next_comp[0]} ({desc})",
                    num_stops=1,
                    pit_laps=[pit_lap],
                    compounds=[next_comp],
                    stints=stints,
                    horizon_laps=h_laps,
                    tactical_intent=intent,
                    assumptions={
                        "pit_lap": pit_lap,
                        "compound_switch": f"{curr_compound} -> {next_comp}",
                        "stint_extension_laps": offset,
                    },
                )

                is_valid, _ = self.validate_strategy(strat, state, ego_driver)
                if is_valid:
                    candidates.append(strat)
                    strat_counter += 1

        # ---------------------------------------------------------------------
        # 2. TWO-STOP STRATEGY FAMILY (If race distance and laps permit)
        # ---------------------------------------------------------------------
        if max_stops >= 2 and remaining_laps >= (self.config.minimum_stint_length * 2 + 2):
            # Test 2-stop combinations: e.g. Pit soon, then sprint stint
            # Stop 1: curr_lap + 1 to 4 laps (or now)
            # Stop 2: mid-way through remaining distance
            stop1_options = [0, min(3, max(1, remaining_life_on_curr // 3))]
            seen_offsets: set[int] = set()
            for s1_off in stop1_options:
                if s1_off in seen_offsets:
                    continue
                seen_offsets.add(s1_off)

                p1 = curr_lap + s1_off
                p2 = p1 + max(self.config.minimum_stint_length + 2, (total_laps - p1) // 2)

                if p2 >= total_laps - 2:
                    continue

                dry_compounds = [c.upper() for c in (self.config.dry_compounds or ["SOFT", "MEDIUM", "HARD"])]
                for comp1 in dry_compounds:
                    for comp2 in dry_compounds:
                        # F1 Two-Compound Regulation (Dry race):
                        # Across the entire race, at least 2 distinct dry compounds must be used.
                        # If driver hasn't pitted yet (prior_pit_count == 0), the full compound sequence
                        # [curr_compound, comp1, comp2] must contain at least 2 unique compounds.
                        if self.config.enforce_f1_two_compound_rule and prior_pit_count == 0:
                            if len({curr_compound, comp1, comp2}) < 2:
                                continue

                        # Hybrid check: verify stint 2 and stint 3 are viable under degradation model
                        stint2_length = p2 - p1
                        stint2_viable, _ = self._is_stint_viable_hybrid(
                            compound=comp1,
                            stint_laps=stint2_length,
                            start_age=0,
                        )
                        if not stint2_viable:
                            continue

                        stint3_length = total_laps - p2
                        stint3_viable, _ = self._is_stint_viable_hybrid(
                            compound=comp2,
                            stint_laps=stint3_length,
                            start_age=0,
                        )
                        if not stint3_viable:
                            continue

                        strat_id = f"S{strat_counter:02d}"

                        stints = [
                            StintPlan(
                                stint_number=prior_pit_count + 1,
                                compound=curr_compound,
                                target_laps=curr_tyre_age + s1_off,
                                start_lap=curr_lap - curr_tyre_age,
                                end_lap=p1,
                            ),
                            StintPlan(
                                stint_number=prior_pit_count + 2,
                                compound=comp1,
                                target_laps=stint2_length,
                                start_lap=p1,
                                end_lap=p2,
                            ),
                            StintPlan(
                                stint_number=prior_pit_count + 3,
                                compound=comp2,
                                target_laps=stint3_length,
                                start_lap=p2,
                                end_lap=total_laps,
                            ),
                        ]

                        strat = Strategy(
                            strategy_id=strat_id,
                            name=f"2-Stop: {curr_compound[0]}->{comp1[0]}->{comp2[0]} (Laps {p1}, {p2})",
                            num_stops=2,
                            pit_laps=[p1, p2],
                            compounds=[comp1, comp2],
                            stints=stints,
                            horizon_laps=h_laps,
                            tactical_intent="aggressive_pace",
                            assumptions={
                                "pit_laps": [p1, p2],
                                "compounds": [comp1, comp2],
                            },
                        )

                        is_valid, _ = self.validate_strategy(strat, state, ego_driver)
                        if is_valid:
                            candidates.append(strat)
                            strat_counter += 1

        # ---------------------------------------------------------------------
        # 3. ZERO-STOP / NO MORE PITS (If driver already completed mandatory stop)
        # ---------------------------------------------------------------------
        if prior_pit_count >= 1:
            strat_id = f"S{strat_counter:02d}"
            strat = Strategy(
                strategy_id=strat_id,
                name=f"0-Stop: Stay Out to End ({curr_compound})",
                num_stops=0,
                pit_laps=[],
                compounds=[],
                stints=[
                    StintPlan(
                        stint_number=prior_pit_count,
                        compound=curr_compound,
                        target_laps=curr_tyre_age + remaining_laps,
                        start_lap=curr_lap - curr_tyre_age,
                        end_lap=total_laps,
                    )
                ],
                horizon_laps=h_laps,
                tactical_intent="track_position",
                assumptions={"remaining_laps": remaining_laps},
            )
            is_valid, _ = self.validate_strategy(strat, state, ego_driver)
            if is_valid:
                candidates.append(strat)
                strat_counter += 1

        # Fallback guarantee: if for any reason candidate list is empty, produce standard 1-stop
        if not candidates:
            strat_id = "S01"
            fallback_comp = "HARD" if curr_compound != "HARD" else "MEDIUM"
            pit_lap = min(curr_lap + max(1, remaining_life_on_curr // 2), total_laps - 1)
            strat = Strategy(
                strategy_id=strat_id,
                name=f"1-Stop Default: {curr_compound[0]}->{fallback_comp[0]} (Lap {pit_lap})",
                num_stops=1,
                pit_laps=[pit_lap],
                compounds=[fallback_comp],
                stints=[],
                horizon_laps=h_laps,
                tactical_intent="nominal",
            )
            candidates.append(strat)

        return candidates
