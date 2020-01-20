import numpy as np
import random
import copy

from model import Actor, Critic
from buffer import PrioritizedReplayBuffer, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_agents, training, args):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            num_agents (int): number of agents
            training (bool): Prepare for training
            args (object): Command line arguments
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        random.seed(seed)
        self.seed = seed

        self._update_buffer_priorities = False

        if args.cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        # Noise process
        self.noise = OUNoise(action_size, self.seed, device=self.device)

        # NN
        if training:
            self.batch_size = args.batch_size
            self.gamma = args.gamma
            self.tau = args.tau

            # Actor Network (w/ Target Network)
            self.actor_local = Actor(state_size, action_size, self.seed).to(self.device)
            self.actor_target = Actor(state_size, action_size, self.seed).to(self.device)
            self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=args.actor_learning_rate)

            # Critic Network (w/ Target Network)
            self.critic_local = Critic(state_size, action_size, self.seed).to(self.device)
            self.critic_target = Critic(state_size, action_size, self.seed).to(self.device)
            self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=args.critic_learning_rate)

            # Replay memory
            self.memory = self._create_buffer(args.buffer.lower(), action_size, args.buffer_size,
                                              self.batch_size, args.alpha, args.beta, self.seed, self.device)
        else:
            self.actor_local = Actor(state_size, action_size, self.seed).to(self.device)

    def _create_buffer(self, buffer_type, action_size, buffer_size, batch_size, alpha, beta, seed, device):
        if buffer_type == 'prioritized':
            self._update_buffer_priorities = True
            return PrioritizedReplayBuffer(action_size, buffer_size, batch_size, seed, alpha=alpha, beta=beta, device=device)
        elif buffer_type == 'sample':
            return ReplayBuffer(action_size, buffer_size, batch_size, seed, device=device)
        else:
            raise Exception('Unknown buffer type - must be one of prioritized or sample')

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences, self.gamma)

    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().to(self.device)
        self.actor_local.eval()
        with torch.no_grad():
            action_values = self.actor_local(state)
        self.actor_local.train()

        if add_noise:
            action_values += self.noise.sample()
        return torch.clamp(action_values, -1, 1).cpu().numpy().tolist()

    def reset(self):
        self.noise.reset()

    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        if self._update_buffer_priorities:
            states, actions, rewards, next_states, dones, indexes, weights = experiences
        else:
            states, actions, rewards, next_states, dones = experiences

        # Critic
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        if self._update_buffer_priorities:
            critic_loss = (F.mse_loss(Q_expected, Q_targets) * weights).mean()
            # Update memory priorities
            self.memory.update_priorities(indexes, (Q_expected - Q_targets).detach().squeeze().abs().cpu().numpy().tolist())
        else:
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        for param in self.critic_local.parameters():
            param.grad.data.clamp_(-1, 1)

        self.critic_optimizer.step()

        # Actor
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()

        for param in self.actor_local.parameters():
            param.grad.data.clamp_(-1, 1)

        self.actor_optimizer.step()

        # ------------------- update target networks ------------------- #
        self.soft_update(self.actor_local, self.actor_target, self.tau)
        self.soft_update(self.critic_local, self.critic_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.05, device='cpu'):
        """Initialize parameters and noise process."""
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.device = device
        torch.manual_seed(seed)
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = self.mu * torch.ones(self.size).to(self.device)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu * torch.ones(self.size).to(self.device) - x) + self.sigma * torch.rand(len(x)).to(self.device)
        self.state = x + dx
        return self.state
