import torch
from torch.utils.data import Dataset


class RolloutBuffer(Dataset):
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, state_size, frames, num_agents, gamma=0.99, gae=0.95, device="cpu"):
        """Initialize a RolloutBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            state_size (int): dimension of state
            frames (int): maximum frames
            num_agents (int): number of agents
            gamma (float) gamma discount
            gae (float) gae coefficient
        """
        super(RolloutBuffer).__init__()

        self.device = device
        self.action_size = action_size
        self.state_size = state_size
        self.num_agents = num_agents
        self.frames = frames
        frames += 1
        self.gamma = gamma
        self.gae = gae

        # inputs
        self.states = torch.zeros((frames, num_agents, state_size), dtype=torch.float64, device=self.device)
        self.actions = torch.zeros((frames, num_agents, action_size), dtype=torch.float64, device=self.device)
        self.rewards = torch.zeros((frames, num_agents), dtype=torch.float64, device=self.device)
        self.log_probs = torch.zeros((frames, num_agents), dtype=torch.float64, device=self.device)
        self.values = torch.zeros((frames, num_agents), dtype=torch.float64, device=self.device)
        self.dones = torch.zeros((frames, num_agents), dtype=torch.int8, device=self.device)

        # computed
        self.returns = torch.zeros((frames, num_agents), dtype=torch.float64, device=self.device)
        self.advantages = torch.zeros((frames, num_agents), dtype=torch.float64, device=self.device)

        self.reset()

    def reset(self):
        self.idx = 0

        self.states.fill_(0.0)
        self.actions.fill_(0.0)
        self.rewards.fill_(0.0)
        self.log_probs.fill_(0.0)
        self.values.fill_(0.0)
        self.dones.fill_(0.0)

        # computed
        self.returns.fill_(0.0)
        self.advantages.fill_(0.0)

    def add(self, actions, rewards, next_states, log_probs, values, dones):
        """Add a new experience to memory."""

        self.states[self.idx].copy_(next_states)
        self.actions[self.idx].copy_(actions)
        self.rewards[self.idx].copy_(rewards)
        self.log_probs[self.idx].copy_(log_probs)
        self.values[self.idx].copy_(values)
        self.dones[self.idx].copy_(dones)

        self.idx += 1

    def compute_rollout(self):
        self.returns[self.frames] = torch.sum(self.rewards, dim=0)

        for i in reversed(range(self.frames)):
            gd = self.gamma * (1 - self.dones[i])

            self.returns[i] = self.rewards[i] + gd * self.returns[i + 1]
            td = self.rewards[i] + gd * self.values[i + 1] - self.values[i]
            self.advantages[i] = self.advantages[i + 1] * self.gae * gd + td

        # Normalize advantages
        adv_norm = self.advantages[0:self.frames]
        std, mean = torch.std_mean(adv_norm, unbiased=False)
        self.advantages[0:self.frames] = (adv_norm - mean) / std

    def __len__(self):
        """Return the current size of internal memory."""
        return self.frames

    def __getitem__(self, index):
        return (self.states[index],
                self.actions[index],
                self.log_probs[index],
                self.returns[index],
                self.advantages[index])
