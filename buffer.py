import torch
import numpy as np
import random
from collections import namedtuple

from segment_tree import SumSegmentTree, MinSegmentTree

class ReplayBuffer(object):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device="cpu"):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.device = device
        self.action_size = action_size
        self.buffer_size = buffer_size
        self.memory = []
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self._next_idx = 0
        random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)

        if self._next_idx >= len(self.memory):
            self.memory.append(e)
        else:
            self.memory[self._next_idx] = e

        self._next_idx = (self._next_idx + 1) % self.buffer_size

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class PrioritizedReplayBuffer(ReplayBuffer):
    """Fixed-size prioritized buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, alpha=0.6, beta=0.5, device="cpu"):
        """Initialize a PrioritizedReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            alpha (float): how much prioritization is used (0 - no prioritization, 1 - full prioritization)
            beta (float): To what degree to use importance weights (0 - no corrections, 1 - full correction)
        """
        super(PrioritizedReplayBuffer, self).__init__(action_size, buffer_size, batch_size, seed, device=device)

        self.alpha = alpha
        self.beta = beta
        self._eps = 0.00000001

        it_capacity = 1
        while it_capacity < buffer_size:
            it_capacity *= 2

        self._it_sum = SumSegmentTree(it_capacity)
        self._it_min = MinSegmentTree(it_capacity)
        self._max_priority = 1.0

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        idx = self._next_idx
        super().add(state, action, reward, next_state, done)

        self._it_sum[idx] = self._max_priority ** self.alpha
        self._it_min[idx] = self._max_priority ** self.alpha

    def _sample_proportional(self):
        res = []
        p_total = self._it_sum.sum(0, len(self.memory) - 1)
        every_range_len = p_total / self.batch_size
        for i in range(self.batch_size):
            mass = random.random() * every_range_len + i * every_range_len
            idx = self._it_sum.find_prefixsum_idx(mass)
            res.append(idx)
        return res

    def sample(self):
        idxes = self._sample_proportional()

        weights = []
        p_min = self._it_min.min() / self._it_sum.sum()
        max_weight = (p_min * len(self.memory) + self._eps) ** (-self.beta)

        for idx in idxes:
            p_sample = self._it_sum[idx] / self._it_sum.sum()
            weight = (p_sample * len(self.memory) + self._eps) ** (-self.beta)
            weights.append(weight / max_weight)

        weights = torch.tensor(weights, device=self.device, dtype=torch.float)

        states = torch.from_numpy(np.vstack([self.memory[i].state for i in idxes])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([self.memory[i].action for i in idxes])).long().to(self.device)
        rewards = torch.from_numpy(np.vstack([self.memory[i].reward for i in idxes])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([self.memory[i].next_state for i in idxes])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([self.memory[i].done for i in idxes]).astype(np.uint8)).float().to(self.device)

        return (states, actions, rewards, next_states, dones, idxes, weights)

    def update_priorities(self, indexes, priorities):
        """Update priorities of sampled transitions.
        sets priority of transition at index indexes[i] in buffer
        to priorities[i].
        Parameters
        ----------
        indexes: [int]
            List of idxes of sampled transitions
        priorities: [float]
            List of updated priorities corresponding to
            transitions at the sampled idxes denoted by
            variable `idxes`.
        """
        for idx, priority in zip(indexes, priorities):
            self._it_sum[idx] = priority ** self.alpha
            self._it_min[idx] = priority ** self.alpha

            self._max_priority = max(self._max_priority, priority)
