import numpy as np
import torch
import math
from torch import nn
import torch.nn.functional as F
from torch import distributions as distribution

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


class TanhTransform(distribution.transforms.Transform):
    domain = distribution.constraints.real
    codomain = distribution.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(distribution.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = distribution.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class DiagGaussianActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim,
                 log_std_bounds, device):
        super().__init__()

        self.log_std_bounds = log_std_bounds

        self.trunk = mlp(obs_dim, hidden_dim, 20)
        # 使用双网络结构
        self.mu_net = mlp(obs_dim, hidden_dim, action_dim)
        self.log_std_net = mlp(obs_dim, hidden_dim, action_dim)

        self.device = device
        self.outputs = dict()

    #     self._initialize_weights()
    #
    # def _initialize_weights(self):
    #     with torch.no_grad():
    #         for m in self.modules():
    #             if isinstance(m, nn.Linear):
    #                 nn.init.xavier_uniform_(m.weight)
    #                 nn.init.constant_(m.bias, 0.0)

    def forward(self, obs):
        mu = self.mu_net(obs)
        log_std = self.log_std_net(obs)

        log_std = torch.clamp(log_std, self.log_std_bounds[0], self.log_std_bounds[1])
        std = log_std.exp()

        self.outputs['mu'] = mu
        self.outputs['std'] = std

        dist = distribution.Normal(mu, std)
        return dist

    def sample_action(self, obs, deterministic=False):
        dist = self.forward(obs)
        action = dist.sample() if not deterministic else dist.mean
        action = torch.sigmoid(action)  # 仅使用sigmoid转换到[0, 1]范围
        return action

