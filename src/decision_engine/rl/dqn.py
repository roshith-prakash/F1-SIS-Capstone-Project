"""
Candidate-Conditioned DQN Network for F1 Strategy Decision Selection.
Handles variable-size candidate strategy sets via parametrized candidate embeddings.
"""

from __future__ import annotations
from pathlib import Path
import random
from typing import Sequence, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..types import ImmediateAction, DecisionExplanation
from ..mapper import ActionMapper
from ..state_encoder import StateEncoder


class CandidateConditionedQNetwork(nn.Module):
    """
    Evaluates Q(S, C_i) for each candidate strategy C_i given race state S.
    Uses separate state and candidate encoders followed by a joint scoring MLP.
    """

    def __init__(
        self,
        state_dim: int = 13,
        candidate_dim: int = 9,
        embed_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.embed_dim = embed_dim

        # State embedding branch
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # Candidate strategy embedding branch
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # Joint scoring head: takes [state_embed, candidate_embed] -> scalar Q-value
        self.q_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            states: Tensor of shape [batch_size, state_dim]
            candidates: Tensor of shape [batch_size, max_candidates, candidate_dim]
            mask: Optional boolean Tensor of shape [batch_size, max_candidates]
                  where True/1 indicates a valid candidate, and False/0 indicates padding.

        Returns:
            q_values: Tensor of shape [batch_size, max_candidates]
        """
        batch_size, max_candidates, _ = candidates.shape

        # Encode state -> [batch_size, 1, embed_dim]
        s_embed = self.state_encoder(states).unsqueeze(1)
        s_embed = s_embed.expand(-1, max_candidates, -1)

        # Encode candidates -> [batch_size, max_candidates, embed_dim]
        c_embed = self.candidate_encoder(candidates)

        # Joint representation -> [batch_size, max_candidates, embed_dim * 2]
        joint = torch.cat([s_embed, c_embed], dim=-1)

        # Compute Q-values -> [batch_size, max_candidates]
        q_vals = self.q_head(joint).squeeze(-1)

        if mask is not None:
            # Mask out invalid/padded candidate slots with -1e9
            q_vals = q_vals.masked_fill(~mask.bool(), -1e9)

        return q_vals


class DuelingCandidateConditionedQNetwork(nn.Module):
    """
    Dueling variant of the Candidate-Conditioned Q-Network.
    Decomposes Q(S, C_i) = V(S) + A(S, C_i) - mean(A) where:
    - V(S): State value stream (how good is our overall race position).
    - A(S, C_i): Advantage stream (marginal benefit of picking candidate i).

    This separation accelerates learning during monotonous cruising laps where all
    candidates have near-identical value, allowing the value stream to learn
    independently from the advantage stream's candidate ranking.
    """

    def __init__(
        self,
        state_dim: int = 13,
        candidate_dim: int = 9,
        embed_dim: int = 64,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.candidate_dim = candidate_dim
        self.embed_dim = embed_dim

        # State embedding branch (shared)
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # Candidate strategy embedding branch
        self.candidate_encoder = nn.Sequential(
            nn.Linear(candidate_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
        )

        # Value stream: V(S) — operates only on state embedding
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Advantage stream: A(S, C_i) — operates on joint [state, candidate] representation
        self.advantage_head = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(
        self,
        states: torch.Tensor,
        candidates: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            states: Tensor of shape [batch_size, state_dim]
            candidates: Tensor of shape [batch_size, max_candidates, candidate_dim]
            mask: Optional boolean Tensor of shape [batch_size, max_candidates]
                  where True/1 indicates a valid candidate, and False/0 indicates padding.

        Returns:
            q_values: Tensor of shape [batch_size, max_candidates]
        """
        batch_size, max_candidates, _ = candidates.shape

        # Encode state -> [batch_size, embed_dim]
        s_embed = self.state_encoder(states)

        # Value stream: V(S) -> [batch_size, 1]
        value = self.value_head(s_embed)

        # Expand state for joint representation -> [batch_size, max_candidates, embed_dim]
        s_embed_expanded = s_embed.unsqueeze(1).expand(-1, max_candidates, -1)

        # Encode candidates -> [batch_size, max_candidates, embed_dim]
        c_embed = self.candidate_encoder(candidates)

        # Joint representation -> [batch_size, max_candidates, embed_dim * 2]
        joint = torch.cat([s_embed_expanded, c_embed], dim=-1)

        # Advantage stream: A(S, C_i) -> [batch_size, max_candidates]
        advantages = self.advantage_head(joint).squeeze(-1)

        # Mean-center advantages over valid candidates only
        if mask is not None:
            # Set invalid slots to 0 before computing mean, then restore
            masked_adv = advantages.masked_fill(~mask.bool(), 0.0)
            valid_counts = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0)
            adv_mean = masked_adv.sum(dim=1, keepdim=True) / valid_counts
        else:
            adv_mean = advantages.mean(dim=1, keepdim=True)

        # Q(S, C_i) = V(S) + A(S, C_i) - mean(A)
        q_vals = value + advantages - adv_mean

        if mask is not None:
            q_vals = q_vals.masked_fill(~mask.bool(), -1e9)

        return q_vals


class RLDecisionPolicy:
    """
    Inference and decision wrapper using the Candidate-Conditioned Q-Network.
    Supports both standard and dueling architectures.
    Provides the same select_action interface as BaselineDecisionPolicy.
    """

    def __init__(
        self,
        q_net: CandidateConditionedQNetwork | DuelingCandidateConditionedQNetwork | None = None,
        state_dim: int = 13,
        candidate_dim: int = 9,
        architecture: str = "standard",
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.architecture = architecture
        if q_net is not None:
            self.q_net = q_net
            if isinstance(q_net, DuelingCandidateConditionedQNetwork):
                self.architecture = "dueling"
            elif isinstance(q_net, CandidateConditionedQNetwork):
                self.architecture = "standard"
        elif architecture == "dueling":
            self.q_net = DuelingCandidateConditionedQNetwork(
                state_dim=state_dim, candidate_dim=candidate_dim
            )
        else:
            self.q_net = CandidateConditionedQNetwork(
                state_dim=state_dim, candidate_dim=candidate_dim
            )
        self.q_net.to(self.device)
        self.q_net.eval()
        self.encoder = StateEncoder()

    def select_candidate_index(
        self,
        state_vec: np.ndarray,
        candidate_vecs: np.ndarray,
        epsilon: float = 0.0,
    ) -> tuple[int, np.ndarray]:
        """
        Selects candidate strategy index via epsilon-greedy Q-evaluation.

        Returns:
            (selected_idx, q_values_array)
        """
        n_candidates = len(candidate_vecs)
        if n_candidates == 0:
            return 0, np.zeros((0,), dtype=np.float32)

        # Exploration
        if epsilon > 0.0 and random.random() < epsilon:
            random_idx = random.randrange(n_candidates)
            return random_idx, np.zeros((n_candidates,), dtype=np.float32)

        # Exploitation via Q-Network
        with torch.no_grad():
            s_t = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)
            c_t = torch.tensor(candidate_vecs, dtype=torch.float32, device=self.device).unsqueeze(0)
            mask_t = torch.ones((1, n_candidates), dtype=torch.bool, device=self.device)

            q_vals = self.q_net(s_t, c_t, mask_t).squeeze(0).cpu().numpy()
            best_idx = int(np.argmax(q_vals))

        return best_idx, q_vals

    def select_action(
        self,
        payload_or_state: Any,
        candidates: list[dict[str, Any]] | None = None,
        race_state: Any = None,
    ) -> DecisionExplanation:
        """
        Evaluates candidate strategies via the trained Q-Network and returns
        the selected tactical action and decision explanation.

        Accepts:
            1. StrategyEngineResult instance
            2. Downstream payload dict (with 'strategies' and 'current_lap')
            3. RaceState + explicit candidates list
        """
        current_lap = 1
        strat_candidates: list[dict[str, Any]] = []
        rs = race_state
        ego_driver = "VER"

        # 1. StrategyEngineResult object
        if hasattr(payload_or_state, "to_decision_engine_payload"):
            payload = payload_or_state.to_decision_engine_payload()
            current_lap = int(payload.get("current_lap", 1))
            strat_candidates = payload.get("strategies", [])
            ego_driver = payload.get("ego_driver", "VER")
            if rs is None:
                rs = getattr(payload_or_state, "race_state", None)

        # 2. RaceState object with explicit candidates
        elif hasattr(payload_or_state, "current_lap") and candidates is not None:
            current_lap = int(payload_or_state.current_lap or 1)
            strat_candidates = candidates
            rs = payload_or_state

        # 3. Dictionary payload
        elif isinstance(payload_or_state, dict):
            if "downstream_payload" in payload_or_state:
                dp = payload_or_state["downstream_payload"]
                current_lap = int(dp.get("current_lap", 1))
                strat_candidates = dp.get("strategies", [])
                ego_driver = dp.get("ego_driver", "VER")
            else:
                current_lap = int(payload_or_state.get("current_lap", 1))
                strat_candidates = payload_or_state.get("strategies", candidates or [])
                ego_driver = payload_or_state.get("ego_driver", "VER")
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

        # Encode state vector
        if rs is not None:
            state_vec = self.encoder.encode_state(rs, ego_driver=ego_driver)
        elif hasattr(payload_or_state, "current_lap"):
            state_vec = self.encoder.encode_state(payload_or_state, ego_driver=ego_driver)
        else:
            state_vec = np.zeros(self.encoder.state_dim, dtype=np.float32)

        # Encode candidates
        cand_vecs = np.array(
            [self.encoder.encode_candidate(c, current_lap=current_lap) for c in strat_candidates],
            dtype=np.float32,
        )

        # Q-Network inference
        best_idx, q_vals = self.select_candidate_index(state_vec, cand_vecs, epsilon=0.0)
        best_candidate = strat_candidates[best_idx]
        best_id = str(best_candidate.get("strategy_id", "UNKNOWN"))
        best_q = float(q_vals[best_idx]) if len(q_vals) > best_idx else 0.0

        # Map to immediate tactical action
        action, target_lap, target_compound = ActionMapper.map_to_immediate_action(
            current_lap, best_candidate
        )

        # Grab top Q-values for explainability payload
        top_candidates_q: dict[str, float] = {}
        sorted_indices = np.argsort(-q_vals) if len(q_vals) > 0 else [best_idx]
        for idx in sorted_indices[:3]:
            if idx < len(strat_candidates):
                cid = str(strat_candidates[idx].get("strategy_id", f"C{idx}"))
                top_candidates_q[cid] = round(float(q_vals[idx]), 3)

        summary = (
            f"RL Decision Policy ({self.architecture.capitalize()} DQN) selected strategy {best_id} "
            f"with Q-value {best_q:.3f}. Immediate action: {action.value}."
        )

        return DecisionExplanation(
            selected_strategy_id=best_id,
            immediate_action=action,
            target_pit_lap=target_lap,
            target_compound=target_compound,
            top_candidates_utilities=top_candidates_q,
            summary=summary,
        )

    def save_checkpoint(self, checkpoint_path: str | Path) -> None:
        """Saves policy network weights and architectural metadata."""
        save_path = Path(checkpoint_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.q_net.state_dict(),
                "state_dim": getattr(self.q_net, "state_dim", 13),
                "candidate_dim": getattr(self.q_net, "candidate_dim", 9),
                "embed_dim": getattr(self.q_net, "embed_dim", 64),
                "architecture": self.architecture,
            },
            save_path,
        )

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> RLDecisionPolicy:
        """
        Loads an RL policy from a saved checkpoint, auto-detecting the architecture.
        Maintains backward compatibility with older checkpoints.
        """
        checkpoint_path = Path(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)

        state_dim = checkpoint.get("state_dim", 13)
        candidate_dim = checkpoint.get("candidate_dim", 9)
        embed_dim = checkpoint.get("embed_dim", 64)

        arch = checkpoint.get("architecture")
        if arch is None:
            # Auto-detect from state_dict keys
            if any("value_head" in k for k in state_dict.keys()):
                arch = "dueling"
            else:
                arch = "standard"

        if arch == "dueling":
            net = DuelingCandidateConditionedQNetwork(
                state_dim=state_dim,
                candidate_dim=candidate_dim,
                embed_dim=embed_dim,
            )
        else:
            net = CandidateConditionedQNetwork(
                state_dim=state_dim,
                candidate_dim=candidate_dim,
                embed_dim=embed_dim,
            )

        net.load_state_dict(state_dict)
        return cls(
            q_net=net,
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            architecture=arch,
            device=device,
        )
