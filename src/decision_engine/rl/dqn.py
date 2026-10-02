"""
Candidate-Conditioned DQN Network for F1 Strategy Decision Selection.
Handles variable-size candidate strategy sets via parametrized candidate embeddings.
"""

from __future__ import annotations
import random
from typing import Sequence, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


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


class RLDecisionPolicy:
    """
    Inference and decision wrapper using the Candidate-Conditioned Q-Network.
    Provides the same select_action interface as BaselineDecisionPolicy.
    """

    def __init__(
        self,
        q_net: CandidateConditionedQNetwork | None = None,
        state_dim: int = 13,
        candidate_dim: int = 9,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.q_net = q_net or CandidateConditionedQNetwork(
            state_dim=state_dim, candidate_dim=candidate_dim
        )
        self.q_net.to(self.device)
        self.q_net.eval()

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
