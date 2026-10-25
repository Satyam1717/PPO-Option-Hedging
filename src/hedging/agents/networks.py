"""PyTorch actor/critic with bounded Beta actions and an optional hedge band.

For direct PPO the sampled action is a stock position. For band PPO two sampled
actions set lower/upper distances from observable BS delta. PPO likelihoods are
always calculated on these sampled controls, including when multiple controls
produce the same held position. No supervised delta targets are used.
"""
import numpy as np
import torch
from torch import nn
from torch.distributions import Beta
from hedging.pricing.black_scholes import call_hedge_ratio


def make_dense_network(input_dim, hidden_sizes, output_dim, output_gain):
    layers = []
    for left, right in zip([input_dim, *hidden_sizes], hidden_sizes):
        linear = nn.Linear(left, right)
        nn.init.orthogonal_(linear.weight, np.sqrt(2))
        nn.init.zeros_(linear.bias)
        layers.extend((linear, nn.Tanh()))
    output = nn.Linear(hidden_sizes[-1], output_dim)
    nn.init.orthogonal_(output.weight, output_gain)
    nn.init.zeros_(output.bias)
    return nn.Sequential(*layers, output)


class HedgeActorCritic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, observe_cost=True):
        super().__init__()
        self.actor = make_dense_network(obs_dim, hidden_sizes, 2 * action_dim, 0.01)
        self.critic = make_dense_network(obs_dim, hidden_sizes, 1, 1.0)
        self.action_dim, self.observe_cost = action_dim, observe_cost
        self.register_buffer("feature_scale", torch.tensor([10., 1., 1., 2.5, 1., 500., 20.]))

    def features(self, obs):
        x = obs * self.feature_scale
        if not self.observe_cost:
            x = x.clone()
            x[..., 5] = 0
        return x

    def distribution(self, obs):
        raw = self.actor(self.features(obs))
        alpha, beta = torch.nn.functional.softplus(raw).chunk(2, dim=-1)
        return Beta(alpha + 1.001, beta + 1.001)

    def value(self, obs):
        return self.critic(self.features(obs)).squeeze(-1)


def hedge_from_controls(controls, obs, policy_kind, market, band_max):
    controls = np.asarray(controls)
    if policy_kind == "direct":
        return controls[..., 0].astype(np.float32)
    obs = np.asarray(obs)
    spot = market.K * np.exp(obs[..., 0].astype(float))
    tau = obs[..., 1].astype(float) * market.maturity
    delta = call_hedge_ratio(spot, market.K, tau, market.r, obs[..., 3])
    lower = np.maximum(0, delta - band_max * controls[..., 0])
    upper = np.minimum(1, delta + band_max * controls[..., 1])
    return np.clip(obs[..., 2], lower, upper).astype(np.float32)


ActorCritic = HedgeActorCritic
controls_to_positions = hedge_from_controls
mlp = make_dense_network
