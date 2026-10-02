from __future__ import annotations

from .types import ImmediateAction


class ActionMapper:
    """
    Maps a multi-lap strategic plan (Strategy Engine candidate) to a single-lap 
    tactical action for the current lap.
    """

    @staticmethod
    def map_to_immediate_action(
        current_lap: int,
        candidate_strategy: dict
    ) -> tuple[ImmediateAction, int | None, str | None]:
        """
        Takes the current lap and the selected candidate strategy payload to determine
        the immediate tactical action.

        Returns:
            tuple[ImmediateAction, int | None, str | None]: 
                (The action, target pit lap if any, target compound if pitting)
        """
        pit_laps = candidate_strategy.get("pit_laps", [])
        compounds = candidate_strategy.get("compounds", [])

        # If we are not on a scheduled pit lap, stay out.
        if current_lap not in pit_laps:
            # Optionally find the next target pit lap for explainability
            next_pit_laps = [lap for lap in pit_laps if lap > current_lap]
            target_pit_lap = next_pit_laps[0] if next_pit_laps else None
            
            target_compound = None
            if target_pit_lap is not None:
                idx = pit_laps.index(target_pit_lap)
                if idx + 1 < len(compounds):
                    target_compound = compounds[idx + 1]

            return ImmediateAction.STAY_OUT, target_pit_lap, target_compound

        # We are on a pit lap
        idx = pit_laps.index(current_lap)
        target_compound = None
        
        # compounds array includes the starting compound at index 0. 
        # The new compound for the i-th pit stop is at index i + 1.
        if idx + 1 < len(compounds):
            target_compound = str(compounds[idx + 1]).strip().upper()
        else:
            target_compound = "MEDIUM" # Fallback if malformed
            
        action_name = f"PIT_{target_compound}"
        
        try:
            action = ImmediateAction(action_name)
        except ValueError:
            # Fallback if the compound is unknown or not mapped
            action = ImmediateAction.STAY_OUT
            
        return action, current_lap, target_compound
