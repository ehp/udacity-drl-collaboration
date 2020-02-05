import random

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from buffer import RolloutBuffer
from model import Policy
from normalizer import MeanStdNormalizer


class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, training, args, num_agents=2, writer=None, device='cpu'):
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
        random.seed(seed)
        self.seed = seed
        self.writer = writer
        self.writer_counter = 0
        self.device = device

        self.normalizer = MeanStdNormalizer(device=device)
        self.policy = Policy(state_size, action_size, seed).to(self.device)

        # NN
        if training:
            self.batch_size = args.batch_size
            self.clip = args.eps_clip
            self.c1 = args.c1
            self.c2 = args.c2
            self.training_epochs = args.training_epochs

            self.buffer = RolloutBuffer(action_size, state_size, args.frames, num_agents,
                                         gamma=args.gamma, gae=args.gae, device=device)

            self.optimizer = optim.Adam(self.policy.parameters(), lr=args.learning_rate,
                                        weight_decay=args.weight_decay, eps=1e-5)

    def act(self, state):
        self.policy.eval()
        with torch.no_grad():
            norm_states = self.normalizer(state)
            values, action_distribution = self.policy(norm_states)
            actions = action_distribution.sample()
            log_prob = self.get_log_prob(action_distribution, actions)

        self.policy.train()

        return actions, log_prob.squeeze(), values.squeeze()

    def save_step(self, actions, rewards, states, log_probs, values, dones):
        norm_states = self.normalizer(states)
        self.buffer.add(actions, rewards, norm_states, log_probs, values, dones)

    def finish_rollout(self, states):
        self.buffer.compute_rollout()
        self.learn()
        self.buffer.reset()


    def get_log_prob(self, action_distribution, actions):
        return action_distribution.log_prob(actions).sum(-1).unsqueeze(-1)

    def learn(self):
        loader = DataLoader(self.buffer, batch_size=self.batch_size, shuffle=True, num_workers=0)

        for i in range(self.training_epochs):
            for bs in loader:
                sampled_states, sampled_actions, sampled_log_probs, sampled_returns, sampled_advantages = bs
                sampled_states, sampled_actions, sampled_log_probs, \
                sampled_returns, sampled_advantages = sampled_states.detach(), \
                                                      sampled_actions.detach(), \
                                                      sampled_log_probs.detach(), \
                                                      sampled_returns.detach(), \
                                                      sampled_advantages.detach()

                values, action_distribution = self.policy(sampled_states)
                log_prob_actions = self.get_log_prob(action_distribution, sampled_actions).squeeze()

                # Action loss
                ratio = torch.exp(log_prob_actions - sampled_log_probs)

                surr1 = ratio * sampled_advantages
                surr2 = torch.clamp(ratio, min=1.0 - self.clip,
                                    max=1.0 + self.clip) * sampled_advantages
                action_loss = -torch.min(surr1, surr2).mean()

                # Value loss - mean squared error
                value_loss = 0.5 * (sampled_returns - values.squeeze(2)).pow(2).mean()

                # Entropy loss
                dist_entropy = action_distribution.entropy().sum(-1).mean()

                # Combine losses
                loss = (value_loss * self.c1 + action_loss - dist_entropy * self.c2)

                # Do the actual optimization
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.75)
                self.optimizer.step()

                self.writer.add_scalar('value_loss', value_loss.item(), self.writer_counter)
                self.writer.add_scalar('action_loss', action_loss.item(), self.writer_counter)
                self.writer.add_scalar('entropy_loss', dist_entropy.item(), self.writer_counter)
                self.writer.add_scalar('overall_loss', loss.item(), self.writer_counter)
                self.writer_counter += 1

                if torch.isnan(loss).any():
                    raise Exception('NaN loss !')
