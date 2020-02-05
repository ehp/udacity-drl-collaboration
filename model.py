import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


# Init layer to have the proper weight initializations.
def init_layer(layer, w_scale=1.0):
    nn.init.orthogonal_(layer.weight.data)
    layer.weight.data.mul_(w_scale)
    nn.init.constant_(layer.bias.data, 0)
    return layer


class Policy(nn.Module):
    """Policy model"""

    def __init__(self, state_size, action_size, seed, actor_units=256, critic_units=256):
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

        self.actor = nn.Sequential(
            init_layer(nn.Linear(state_size, actor_units)),
            nn.Tanh(),
            init_layer(nn.Linear(actor_units, actor_units)),
            nn.Tanh(),
            init_layer(nn.Linear(actor_units, actor_units)),
            nn.Tanh(),
            init_layer(nn.Linear(actor_units, action_size), 1e-3),
            nn.Tanh()
        ).double()
        self.critic = nn.Sequential(
            init_layer(nn.Linear(state_size, critic_units)),
            nn.Tanh(),
            init_layer(nn.Linear(critic_units, critic_units)),
            nn.Tanh(),
            init_layer(nn.Linear(critic_units, critic_units)),
            nn.Tanh(),
            init_layer(nn.Linear(critic_units, 1), 1e-3)
        ).double()

        self.std = nn.Parameter(torch.zeros(action_size, dtype=torch.float64))

    def forward(self, state):
        mean = self.actor(state)
        value = self.critic(state)
        dist = Normal(mean, F.softplus(self.std))
        return value, dist
