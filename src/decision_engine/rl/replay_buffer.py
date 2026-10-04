"""
Replay buffer for variable candidate set Q-learning in F1 Decision Engine.
Supports uniform experience replay and Prioritized Experience Replay (PER).
"""

from __future__ import annotations
import random
from dataclasses import dataclass
from typing import Sequence
import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    candidates: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    next_candidates: np.ndarray
    done: bool


def collate_candidate_batch(
    batch: list[Transition],
    indices: list[int] | None = None,
    weights: np.ndarray | None = None,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """
    Dynamically pads candidate strategy matrices across a batch of transitions.
    """
    states = np.stack([t.state for t in batch], axis=0)
    actions = np.array([t.action for t in batch], dtype=np.int64)
    rewards = np.array([t.reward for t in batch], dtype=np.float32)
    next_states = np.stack([t.next_state for t in batch], axis=0)
    dones = np.array([t.done for t in batch], dtype=np.float32)

    # Pad current candidates
    cand_dim = batch[0].candidates.shape[-1] if len(batch[0].candidates) > 0 else 9
    max_k = max(max(len(t.candidates) for t in batch), 1)

    padded_cands = np.zeros((len(batch), max_k, cand_dim), dtype=np.float32)
    cands_mask = np.zeros((len(batch), max_k), dtype=bool)

    for i, t in enumerate(batch):
        k = len(t.candidates)
        if k > 0:
            padded_cands[i, :k, :] = t.candidates
            cands_mask[i, :k] = True

    # Pad next candidates
    next_cand_dim = (
        batch[0].next_candidates.shape[-1] if len(batch[0].next_candidates) > 0 else cand_dim
    )
    max_next_k = max(max(len(t.next_candidates) for t in batch), 1)

    padded_next_cands = np.zeros((len(batch), max_next_k, next_cand_dim), dtype=np.float32)
    next_cands_mask = np.zeros((len(batch), max_next_k), dtype=bool)

    for i, t in enumerate(batch):
        k_next = len(t.next_candidates)
        if k_next > 0:
            padded_next_cands[i, :k_next, :] = t.next_candidates
            next_cands_mask[i, :k_next] = True

    dev = torch.device(device)
    batch_dict: dict[str, torch.Tensor] = {
        "states": torch.tensor(states, dtype=torch.float32, device=dev),
        "candidates": torch.tensor(padded_cands, dtype=torch.float32, device=dev),
        "candidates_mask": torch.tensor(cands_mask, dtype=torch.bool, device=dev),
        "actions": torch.tensor(actions, dtype=torch.int64, device=dev),
        "rewards": torch.tensor(rewards, dtype=torch.float32, device=dev),
        "next_states": torch.tensor(next_states, dtype=torch.float32, device=dev),
        "next_candidates": torch.tensor(padded_next_cands, dtype=torch.float32, device=dev),
        "next_candidates_mask": torch.tensor(next_cands_mask, dtype=torch.bool, device=dev),
        "dones": torch.tensor(dones, dtype=torch.float32, device=dev),
    }

    if indices is not None:
        batch_dict["indices"] = torch.tensor(indices, dtype=torch.int64, device=dev)
    if weights is not None:
        batch_dict["is_weights"] = torch.tensor(weights, dtype=torch.float32, device=dev)
    else:
        batch_dict["is_weights"] = torch.ones(len(batch), dtype=torch.float32, device=dev)

    return batch_dict


class VariableCandidateReplayBuffer:
    """
    Experience replay buffer that supports variable numbers of candidate strategies per step.
    Pads candidate matrices dynamically within sampled batches.
    """

    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer: list[Transition] = []
        self.position = 0

    def push(
        self,
        state: np.ndarray,
        candidates: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_candidates: np.ndarray,
        done: bool,
    ) -> None:
        """Stores a transition in the buffer."""
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            candidates=np.asarray(candidates, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            next_candidates=np.asarray(next_candidates, dtype=np.float32),
            done=bool(done),
        )

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        """
        Samples a mini-batch uniformly and dynamically pads candidates to the batch maximum.
        """
        sampled_batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        return collate_candidate_batch(sampled_batch, device=device)

    def update_priorities(self, indices: Any, td_errors: Any) -> None:
        """No-op for uniform buffer compatibility."""
        pass

    def anneal_beta(self, current_step: int, total_steps: int) -> None:
        """No-op for uniform buffer compatibility."""
        pass

    def __len__(self) -> int:
        return len(self.buffer)


class SumTree:
    """
    Binary sum-tree data structure for O(log N) prioritized sampling and priority updates.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        # tree size: 2 * capacity. Leaves start at index capacity.
        self.tree = np.zeros(2 * capacity, dtype=np.float64)
        self.data: list[Transition | None] = [None] * capacity
        self.write_idx = 0
        self.size = 0

    def total(self) -> float:
        """Returns the sum of all priorities (root node value)."""
        return float(self.tree[1])

    def update(self, tree_idx: int, priority: float) -> None:
        """Updates priority of a leaf and propagates delta up to root."""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        idx = tree_idx // 2
        while idx >= 1:
            self.tree[idx] += change
            idx //= 2

    def add(self, priority: float, data: Transition) -> int:
        """Inserts a new transition with priority. Returns tree_idx."""
        tree_idx = self.write_idx + self.capacity
        self.data[self.write_idx] = data
        self.update(tree_idx, priority)

        self.write_idx = (self.write_idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return tree_idx

    def get(self, value: float) -> tuple[int, float, Transition]:
        """
        Samples a leaf node based on cumulative priority prefix sum.
        Traverses tree from root to leaf in O(log N).
        """
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            right = left + 1
            if left >= len(self.tree):
                break
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                if right < len(self.tree):
                    idx = right
                else:
                    idx = left

        data_idx = idx - self.capacity
        data_idx = max(0, min(data_idx, max(0, self.size - 1)))
        real_tree_idx = data_idx + self.capacity
        transition = self.data[data_idx]
        if transition is None:
            # Fallback if slot uninitialized
            for i in range(self.size):
                if self.data[i] is not None:
                    data_idx = i
                    real_tree_idx = data_idx + self.capacity
                    transition = self.data[data_idx]
                    break
        assert transition is not None, "SumTree sampled empty slot"
        return real_tree_idx, float(self.tree[real_tree_idx]), transition


class PrioritizedVariableCandidateReplayBuffer:
    """
    Prioritized Experience Replay (PER) buffer for variable candidate set Q-learning.
    Uses proportional sampling with a SumTree and importance-sampling (IS) bias correction.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_end: float = 1.0,
        epsilon: float = 1e-5,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta_start
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.epsilon = epsilon
        self.tree = SumTree(capacity)
        self.max_priority = 1.0

    def push(
        self,
        state: np.ndarray,
        candidates: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        next_candidates: np.ndarray,
        done: bool,
    ) -> None:
        """Stores a transition with initial max priority."""
        transition = Transition(
            state=np.asarray(state, dtype=np.float32),
            candidates=np.asarray(candidates, dtype=np.float32),
            action=int(action),
            reward=float(reward),
            next_state=np.asarray(next_state, dtype=np.float32),
            next_candidates=np.asarray(next_candidates, dtype=np.float32),
            done=bool(done),
        )
        self.tree.add(self.max_priority, transition)

    def sample(self, batch_size: int, device: str = "cpu") -> dict[str, torch.Tensor]:
        """
        Samples a mini-batch stratified across priority segments and calculates IS weights.
        """
        n = len(self)
        if n == 0:
            raise ValueError("Cannot sample from empty replay buffer")

        batch_size = min(batch_size, n)
        total_p = max(self.tree.total(), self.epsilon)
        segment = total_p / batch_size

        batch_transitions: list[Transition] = []
        tree_indices: list[int] = []
        priorities: list[float] = []

        for i in range(batch_size):
            low = i * segment
            high = (i + 1) * segment
            v = random.uniform(low, high)
            t_idx, p, trans = self.tree.get(v)
            batch_transitions.append(trans)
            tree_indices.append(t_idx)
            priorities.append(max(p, self.epsilon))

        # Importance-sampling weights: w_i = (N * P(i))^(-beta) / max_w
        probs = np.array(priorities, dtype=np.float64) / total_p
        weights = (n * probs) ** (-self.beta)
        weights = weights / (weights.max() + 1e-8)  # normalize to max 1.0

        return collate_candidate_batch(
            batch=batch_transitions,
            indices=tree_indices,
            weights=weights,
            device=device,
        )

    def update_priorities(
        self,
        tree_indices: Sequence[int] | torch.Tensor,
        td_errors: Sequence[float] | np.ndarray | torch.Tensor,
    ) -> None:
        """Updates priorities in SumTree based on TD errors."""
        if isinstance(tree_indices, torch.Tensor):
            tree_indices = tree_indices.cpu().numpy().tolist()
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().abs().cpu().numpy().tolist()
        elif isinstance(td_errors, np.ndarray):
            td_errors = np.abs(td_errors).tolist()

        for idx, td in zip(tree_indices, td_errors):
            p = float((abs(td) + self.epsilon) ** self.alpha)
            p = min(max(p, self.epsilon), 100.0)
            self.tree.update(int(idx), p)
            self.max_priority = max(self.max_priority, p)

    def anneal_beta(self, current_step: int, total_steps: int) -> None:
        """Linearly anneals beta parameter from beta_start to beta_end."""
        frac = min(1.0, max(0.0, current_step / max(1, total_steps)))
        self.beta = self.beta_start + frac * (self.beta_end - self.beta_start)

    def __len__(self) -> int:
        return self.tree.size
