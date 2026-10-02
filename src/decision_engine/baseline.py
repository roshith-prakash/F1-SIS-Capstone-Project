from __future__ import annotations
from typing import Any

from .types import DecisionExplanation, ImmediateAction
from .mapper import ActionMapper


class BaselineDecisionPolicy:
    """
    Deterministic utility-based decision policy.
    Uses configurable weights based on the active risk profile to rank candidate strategies
    and select the immediate tactical action.
    """

    def __init__(self, risk_profile: str = "balanced"):
        self.risk_profile = str(risk_profile).strip().lower()
        self.weights = self._get_weights_for_profile(self.risk_profile)

    def _get_weights_for_profile(self, profile: str) -> dict[str, float]:
        """Returns the objective weights for the utility function."""
        if profile == "aggressive":
            return {
                "p_win": 50.0,
                "p_podium": 30.0,
                "p_top5": 10.0,
                "position_norm": 20.0,
                "robustness": 5.0,
                "tyre_risk": -5.0,
                "traffic_risk": -5.0,
            }
        elif profile == "conservative":
            return {
                "p_win": 10.0,
                "p_podium": 10.0,
                "p_top5": 20.0,
                "position_norm": 30.0,
                "robustness": 30.0,
                "tyre_risk": -25.0,
                "traffic_risk": -15.0,
            }
        else:  # "balanced" or unknown
            return {
                "p_win": 25.0,
                "p_podium": 20.0,
                "p_top5": 15.0,
                "position_norm": 25.0,
                "robustness": 15.0,
                "tyre_risk": -15.0,
                "traffic_risk": -10.0,
            }

    def _calculate_utility(self, candidate: dict[str, Any]) -> float:
        """Calculates a deterministic utility score for a single candidate."""
        p_win = float(candidate.get("p_win", 0.0))
        p_pod = float(candidate.get("p_podium", 0.0))
        p_t5 = float(candidate.get("p_top5", 0.0))
        rob = float(candidate.get("robustness", 0.0))
        tr = float(candidate.get("tyre_risk", 0.0))
        tf = float(candidate.get("traffic_risk", 0.0))

        # Normalize expected position with progressive weighting for podium and top 5
        exp_pos = float(candidate.get("expected_position", 10.0))
        pos_norm = max(0.0, ((21.0 - exp_pos) / 20.0) ** 1.4)

        # Tyre risk: only penalize when tyre wear approaches the cliff zone (>0.75 life)
        tyre_cliff_penalty = max(0.0, (tr - 0.75) / 0.25) if tr > 0.75 else 0.0

        w = self.weights
        utility = (
            p_win * w["p_win"]
            + p_pod * w["p_podium"]
            + p_t5 * w["p_top5"]
            + pos_norm * w["position_norm"]
            + rob * w["robustness"]
            + tyre_cliff_penalty * w["tyre_risk"]
            + tf * w["traffic_risk"]
        )
        return float(utility)

    def _build_explainability_summary(
        self,
        best_candidate: dict[str, Any],
        best_utility: float,
        runner_up: dict[str, Any] | None,
        action: ImmediateAction,
        target_pit_lap: int | None,
        target_compound: str | None,
    ) -> str:
        """Constructs an intuitive natural language explanation for why the decision was taken."""
        strat_id = best_candidate.get("strategy_id", "UNKNOWN")
        stops = len(best_candidate.get("pit_laps", []))
        compounds = " -> ".join(best_candidate.get("compounds", []))

        action_desc = (
            f"box this lap for {target_compound}"
            if action != ImmediateAction.STAY_OUT
            else f"stay out (next planned stop: L{target_pit_lap} for {target_compound})"
            if target_pit_lap
            else "stay out to the flag"
        )

        base_msg = (
            f"Strategy {strat_id} ({stops}-stop: {compounds}) selected under {self.risk_profile} "
            f"profile (Utility: {best_utility:.2f}). Tactical command: {action_desc}."
        )

        if runner_up is not None:
            r_id = runner_up.get("strategy_id", "runner-up")
            reasons = []

            # Compare key dimensions
            best_pos = best_candidate.get("expected_position", 10.0)
            run_pos = runner_up.get("expected_position", 10.0)
            if best_pos < run_pos - 0.2:
                reasons.append(f"better expected finish (P{best_pos:.1f} vs P{run_pos:.1f})")

            best_win = best_candidate.get("p_win", 0.0)
            run_win = runner_up.get("p_win", 0.0)
            if best_win > run_win + 0.03:
                reasons.append(f"+{((best_win - run_win) * 100):.1f}% win probability")

            best_tyre = best_candidate.get("tyre_risk", 0.0)
            run_tyre = runner_up.get("tyre_risk", 0.0)
            if best_tyre < run_tyre - 0.05:
                reasons.append(f"significantly lower tyre risk ({best_tyre:.2f} vs {run_tyre:.2f})")

            best_rob = best_candidate.get("robustness", 0.0)
            run_rob = runner_up.get("robustness", 0.0)
            if best_rob > run_rob + 0.05:
                reasons.append(f"superior robustness ({best_rob:.2f} vs {run_rob:.2f})")

            if reasons:
                contrast_str = " Key advantages over " + r_id + ": " + ", ".join(reasons) + "."
            else:
                contrast_str = f" Balanced trade-off preferred over {r_id} based on risk weights."
            return base_msg + contrast_str

        return base_msg

    def select_action(
        self,
        payload_or_state: Any,
        candidates: list[dict[str, Any]] | None = None,
    ) -> DecisionExplanation:
        """
        Evaluates candidate strategies and returns the selected tactical action.

        Accepts:
            1. StrategyEngineResult instance
            2. Downstream payload dict (with 'strategies' and 'current_lap')
            3. RaceState + explicit candidates list
        """
        current_lap = 1
        strat_candidates: list[dict[str, Any]] = []

        # 1. StrategyEngineResult object
        if hasattr(payload_or_state, "to_decision_engine_payload"):
            payload = payload_or_state.to_decision_engine_payload()
            current_lap = int(payload.get("current_lap", 1))
            strat_candidates = payload.get("strategies", [])

        # 2. RaceState object with explicit candidates
        elif hasattr(payload_or_state, "current_lap") and candidates is not None:
            current_lap = int(payload_or_state.current_lap or 1)
            strat_candidates = candidates

        # 3. Dictionary payload
        elif isinstance(payload_or_state, dict):
            if "downstream_payload" in payload_or_state:
                dp = payload_or_state["downstream_payload"]
                current_lap = int(dp.get("current_lap", 1))
                strat_candidates = dp.get("strategies", [])
            else:
                current_lap = int(payload_or_state.get("current_lap", 1))
                strat_candidates = payload_or_state.get("strategies", candidates or [])
        else:
            strat_candidates = candidates or []

        if not strat_candidates:
            return DecisionExplanation(
                selected_strategy_id="FALLBACK",
                immediate_action=ImmediateAction.STAY_OUT,
                target_pit_lap=None,
                target_compound=None,
                top_candidates_utilities={},
                summary="No candidates provided. Falling back to STAY_OUT.",
            )

        # Evaluate utility for all candidates
        utilities: dict[str, float] = {}
        for c in strat_candidates:
            c_id = str(c.get("strategy_id", "UNKNOWN"))
            utilities[c_id] = self._calculate_utility(c)

        # Sort descending
        sorted_candidates = sorted(
            strat_candidates,
            key=lambda c: utilities[str(c.get("strategy_id", "UNKNOWN"))],
            reverse=True,
        )

        best_candidate = sorted_candidates[0]
        best_id = str(best_candidate.get("strategy_id", "UNKNOWN"))
        best_utility = utilities[best_id]

        runner_up = sorted_candidates[1] if len(sorted_candidates) > 1 else None

        # Map to immediate tactical action
        action, target_lap, target_compound = ActionMapper.map_to_immediate_action(
            current_lap, best_candidate
        )

        # Grab top 3 candidates for explainability payload
        top_3_utils = {
            str(c.get("strategy_id", "UNKNOWN")): round(utilities[str(c.get("strategy_id", "UNKNOWN"))], 3)
            for c in sorted_candidates[:3]
        }

        summary = self._build_explainability_summary(
            best_candidate=best_candidate,
            best_utility=best_utility,
            runner_up=runner_up,
            action=action,
            target_pit_lap=target_lap,
            target_compound=target_compound,
        )

        return DecisionExplanation(
            selected_strategy_id=best_id,
            immediate_action=action,
            target_pit_lap=target_lap,
            target_compound=target_compound,
            top_candidates_utilities=top_3_utils,
            summary=summary,
        )
