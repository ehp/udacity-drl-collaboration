import random

from model import Policy
from buffer import PrioritizedReplayBuffer, ReplayBuffer

import torch
import torch.nn.functional as F
import torch.optim as optim


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_agents, training, args, writer=None):
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
        self.writer = writer
        self.writer_counter = 0

        self._update_buffer_priorities = False

        if args.cuda:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        # Noise process
        self.noise = OUNoise(action_size, self.seed, device=self.device)

        self.policy = Policy(state_size, action_size, seed).to(self.device)

        # NN
        if training:
            self.batch_size = args.batch_size
            self.gamma = args.gamma
            self.tau = args.tau
            self.clip = 0.2

            self.policy_target = Policy(state_size, action_size, seed).to(self.device)

            self.optimizer = optim.Adam(self.policy.parameters(), lr=args.learning_rate)

            # Replay memory
            self.memory = self._create_buffer(args.buffer.lower(), action_size, args.buffer_size,
                                              self.batch_size, args.alpha, args.beta, self.seed, self.device)

    def _create_buffer(self, buffer_type, action_size, buffer_size, batch_size, alpha, beta, seed, device):
        if buffer_type == 'prioritized':
            self._update_buffer_priorities = True
            return PrioritizedReplayBuffer(action_size, buffer_size, batch_size, seed, alpha=alpha, beta=beta, device=device)
        elif buffer_type == 'sample':
            return ReplayBuffer(action_size, buffer_size, batch_size, seed, device=device)
        else:
            raise Exception('Unknown buffer type - must be one of prioritized or sample')

    def step(self, states, actions, rewards, next_states, log_probs, dones):
        # Save experience in replay memory
        for i in range(len(states)):
            self.memory.add(states[i], actions[i], rewards[i], next_states[i], log_probs[i], dones[i])

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
        self.policy.eval()
        with torch.no_grad():
            _, action, action_log_probs = self.policy.act(state)
            action_values = action.squeeze(1)
        self.policy.train()

        if add_noise:
            action_values += self.noise.sample()
        return torch.clamp(action_values, -1, 1).cpu().numpy().tolist(), action_log_probs

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
            states, actions, rewards, next_states, log_probs, dones, indexes, weights = experiences
        else:
            states, actions, rewards, next_states, log_probs, dones = experiences

        values, action_log_probs, dist_entropy = self.policy.evaluate_actions(states, actions)

        ratio = torch.exp(action_log_probs - log_probs)

        value_target = self.policy_target.get_value(next_states)
        adv_target = rewards + (gamma * value_target * (1 - dones)) - value_target

        surr1 = ratio * adv_target
        surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv_target

        action_loss = -torch.min(surr1, surr2).mean()

        value_loss = F.mse_loss(rewards, values)
        self.optimizer.zero_grad()

        loss = (value_loss * 0.5 + action_loss - dist_entropy * 0.01)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        self.writer.add_scalar('value_loss', value_loss.item(), self.writer_counter)
        self.writer.add_scalar('action_loss', action_loss.item(), self.writer_counter)
        self.writer.add_scalar('entropy_loss', dist_entropy.item(), self.writer_counter)
        self.writer.add_scalar('overall_loss', loss.item(), self.writer_counter)
        self.writer_counter += 1

        # ------------------- update target networks ------------------- #
        self.soft_update(self.policy, self.policy_target, self.tau)

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
