import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Init layer to have the proper weight initializations.
def init_layer(layer):
    weight = layer.weight.data
    weight.normal_(0, 1)
    weight *= 1.0 / torch.sqrt(weight.pow(2).sum(1, keepdim=True))
    nn.init.constant_(layer.bias.data, 0)
    return layer


class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=128):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = init_layer(nn.Linear(state_size, fc1_units))
        self.fc2 = init_layer(nn.Linear(fc1_units, action_size))

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        x = F.leaky_relu(self.fc1(state))
        return torch.tanh(self.fc2(x))


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, seed, fc1_units=128, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            seed (int): Random seed
            fcs1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = init_layer(nn.Linear(state_size, fc1_units))
        self.fc2 = init_layer(nn.Linear(fc1_units, fc2_units))
        self.fc3 = init_layer(nn.Linear(fc2_units, 1))

    def forward(self, state):
        """Build a critic (value) network that maps state -> Q-values."""
        x = F.leaky_relu(self.fc1(state))
        x = F.leaky_relu(self.fc2(x))
        return self.fc3(x)


class Policy(nn.Module):
    """Policy model"""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.actor = Actor(state_size, action_size, seed)
        self.critic = Critic(state_size, seed)

        # How we will define our normal distribution to sample action from
        self.action_mean = init_layer(nn.Linear(action_size, action_size))
        # TODO self.action_mean = init_layer(nn.Linear(64, action_size))
        self.action_log_std = nn.Parameter(torch.zeros(1, action_size))

    def __get_dist(self, actor_features):
        action_mean = self.action_mean(actor_features)
        action_log_std = self.action_log_std

        return Normal(action_mean, action_log_std.exp())

    def act(self, state):
        actor_features = self.actor(state)
        value = self.critic(state)

        dist = self.__get_dist(actor_features)
        action = dist.sample()
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)

        return value, action, action_log_probs

    def get_value(self, state):
        return self.critic(state)

    def evaluate_actions(self, state, action):
        actor_features = self.actor(state)
        value = self.critic(state)

        dist = self.__get_dist(actor_features)
        action_log_probs = dist.log_prob(action).sum(-1, keepdim=True)
        dist_entropy = dist.entropy().sum(-1).mean()

        return value, action_log_probs, dist_entropy
