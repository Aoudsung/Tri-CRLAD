import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

def mlp(input_dim, hidden_dims, output_dim, output_mod=None):

    mods = [nn.Linear(input_dim, hidden_dims[0]), nn.ReLU(inplace=True)]

    # Append subsequent hidden layers based on the provided hidden_dims list
    for i in range(1, len(hidden_dims)):
        mods += [nn.Linear(hidden_dims[i - 1], hidden_dims[i]), nn.ReLU(inplace=True)]

    mods.append(nn.Linear(hidden_dims[-1], output_dim))

    if output_mod is not None:
        mods.append(output_mod)

    trunk = nn.Sequential(*mods)
    return trunk

class DoubleQCritic(nn.Module):
    """Critic network, employes double Q-learning."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.Q1 = mlp(obs_dim + action_dim, hidden_dim, 1)
        self.Q2 = mlp(obs_dim + action_dim, hidden_dim, 1)

        self.outputs = dict()
        self._initialize_weights()

    def _initialize_weights(self, ):
        with torch.no_grad():
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=-1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

