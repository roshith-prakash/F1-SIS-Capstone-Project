"""
Replay buffer for variable candidate set Q-learning in F1 Decision Engine.
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
        Samples a mini-batch and dynamically pads candidates to the batch maximum.
        """
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

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
        return {
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

    def __len__(self) -> int:
        return len(self.buffer)
