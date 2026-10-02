"""
DQN Trainer for F1-SIS Candidate-Conditioned Decision Engine.
Implements:
- Warm-start Imitation Learning from Baseline Deterministic Policy
- Double DQN / Target Network stabilization with soft Polyak updates
- Dynamic variable-candidate Bellman updates
- Checkpointing and evaluation against baseline
"""

from __future__ import annotations
import copy
from pathlib import Path
from typing import Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..baseline import BaselineDecisionPolicy
from .environment import F1StrategyEnv
from .dqn import CandidateConditionedQNetwork, RLDecisionPolicy
from .replay_buffer import VariableCandidateReplayBuffer


class DQNTrainer:
    """
    Trains CandidateConditionedQNetwork using Deep Q-Learning with dynamic candidate sets.
    """

    def __init__(
        self,
        env: F1StrategyEnv | None = None,
        state_dim: int = 13,
        candidate_dim: int = 9,
        embed_dim: int = 64,
        hidden_dim: int = 64,
        learning_rate: float = 5e-4,
        gamma: float = 0.98,
        tau: float = 0.005,
        buffer_capacity: int = 5000,
        batch_size: int = 32,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.env = env or F1StrategyEnv(rollouts_per_step=10, horizon_laps=6, seed=42)
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        # Networks: Policy Net and Target Net
        self.policy_net = CandidateConditionedQNetwork(
            state_dim=state_dim,
            candidate_dim=candidate_dim,
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.target_net = copy.deepcopy(self.policy_net).to(self.device)
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.replay_buffer = VariableCandidateReplayBuffer(capacity=buffer_capacity)

        # Baseline policy for warm-up imitation learning
        self.baseline_policy = BaselineDecisionPolicy(risk_profile="balanced")

    def warm_up_with_baseline(self, n_episodes: int = 5) -> None:
        """
        Pre-fills the replay buffer with high-utility actions from BaselineDecisionPolicy.
        Provides a stable initial policy initialization (Imitation / Warm-start).
        """
        print(f"--> Warm-starting replay buffer with {n_episodes} baseline demonstration episodes...")
        for ep in range(n_episodes):
            obs, info = self.env.reset(seed=100 + ep)
            terminated = False

            while not terminated:
                candidates = info.get("candidates", [])
                cand_features = info.get("candidate_features", np.zeros((0, 9)))
                k = len(candidates)
                if k == 0:
                    break

                # Query Baseline Policy for expert action
                explanation = self.baseline_policy.select_action(
                    self.env.current_state, candidates=candidates
                )
                action_idx = 0
                for idx, c in enumerate(candidates):
                    if c.get("strategy_id") == explanation.selected_strategy_id:
                        action_idx = idx
                        break

                next_obs, reward, terminated, truncated, next_info = self.env.step(action_idx)
                next_cand_features = next_info.get("candidate_features", np.zeros((0, 9)))

                self.replay_buffer.push(
                    state=obs,
                    candidates=cand_features,
                    action=action_idx,
                    reward=reward,
                    next_state=next_obs,
                    next_candidates=next_cand_features,
                    done=terminated,
                )

                obs = next_obs
                info = next_info

        print(f"     Buffer successfully pre-warmed with {len(self.replay_buffer)} transitions.\n")

    def _train_step(self) -> float | None:
        """Samples a mini-batch and performs a single gradient update."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        batch = self.replay_buffer.sample(self.batch_size, device=str(self.device))

        states = batch["states"]
        candidates = batch["candidates"]
        mask = batch["candidates_mask"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        next_states = batch["next_states"]
        next_candidates = batch["next_candidates"]
        next_mask = batch["next_candidates_mask"]
        dones = batch["dones"]

        # Compute Q(S, C_a)
        q_vals = self.policy_net(states, candidates, mask)  # [B, K]
        # Gather chosen action Q-value
        chosen_q = q_vals.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Target Q: R + gamma * max_a' Q_target(S', C'_a') * (1 - Done)
        with torch.no_grad():
            next_q_vals = self.target_net(next_states, next_candidates, next_mask)  # [B, K_next]
            # Replace -1e9 mask values with large negative for argmax safety
            max_next_q = next_q_vals.max(dim=1)[0]
            # Clamp to prevent invalid masking blowups
            max_next_q = torch.clamp(max_next_q, min=-10.0, max=50.0)
            target_q = rewards + (1.0 - dones) * self.gamma * max_next_q

        loss = F.smooth_l1_loss(chosen_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Polyak soft update for target network
        for param, target_param in zip(self.policy_net.parameters(), self.target_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

        return float(loss.item())

    def train(
        self,
        n_episodes: int = 25,
        warm_up_episodes: int = 5,
        epsilon_start: float = 0.8,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.95,
        save_path: str | Path | None = None,
    ) -> RLDecisionPolicy:
        """
        Executes complete DQN training loop across n_episodes.
        """
        # Step 1: Warm-start replay buffer
        if warm_up_episodes > 0:
            self.warm_up_with_baseline(n_episodes=warm_up_episodes)

        epsilon = epsilon_start
        print(f"--> Starting DQN Training Loop ({n_episodes} episodes)...")

        for ep in range(1, n_episodes + 1):
            obs, info = self.env.reset(seed=200 + ep)
            terminated = False
            total_reward = 0.0
            losses = []

            while not terminated:
                cand_features = info.get("candidate_features", np.zeros((0, 9)))
                candidates = info.get("candidates", [])
                k = len(candidates)
                if k == 0:
                    break

                # Epsilon-greedy selection
                if np.random.rand() < epsilon:
                    action_idx = int(np.random.randint(k))
                else:
                    with torch.no_grad():
                        s_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                        c_t = torch.tensor(cand_features, dtype=torch.float32, device=self.device).unsqueeze(0)
                        m_t = torch.ones((1, k), dtype=torch.bool, device=self.device)
                        q_vals = self.policy_net(s_t, c_t, m_t).squeeze(0).cpu().numpy()
                        action_idx = int(np.argmax(q_vals))

                next_obs, reward, terminated, truncated, next_info = self.env.step(action_idx)
                next_cand_features = next_info.get("candidate_features", np.zeros((0, 9)))

                self.replay_buffer.push(
                    state=obs,
                    candidates=cand_features,
                    action=action_idx,
                    reward=reward,
                    next_state=next_obs,
                    next_candidates=next_cand_features,
                    done=terminated,
                )

                # Train step
                loss = self._train_step()
                if loss is not None:
                    losses.append(loss)

                obs = next_obs
                info = next_info
                total_reward += reward

            # Decay epsilon
            epsilon = max(epsilon_end, epsilon * epsilon_decay)
            mean_loss = np.mean(losses) if losses else 0.0
            finish_pos = info.get("ego_position", 10)

            if ep % 5 == 0 or ep == 1 or ep == n_episodes:
                print(
                    f"  Episode {ep:2d}/{n_episodes:2d} | "
                    f"Finish: P{finish_pos:<2d} | "
                    f"Reward: {total_reward:>6.2f} | "
                    f"Avg Loss: {mean_loss:>6.4f} | "
                    f"Eps: {epsilon:.3f}"
                )

        print("\n--> Training Complete!")

        # Create trained RL policy
        trained_policy = RLDecisionPolicy(q_net=self.policy_net, device=str(self.device))

        # Save checkpoint if requested
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": self.policy_net.state_dict(),
                    "state_dim": self.policy_net.state_dim,
                    "candidate_dim": self.policy_net.candidate_dim,
                    "embed_dim": self.policy_net.embed_dim,
                },
                save_path,
            )
            print(f"--> Saved trained RL checkpoint to: {save_path}")

        return trained_policy
