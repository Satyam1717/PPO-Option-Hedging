"""Transparent, non-learned controls for the RL experiment."""
import numpy as np
from hedging.agents.delta_hedge import EstimatedVolatilityDelta


class ConstantHedge:
    def __init__(self, holding=0.5):
        if not np.isfinite(holding) or not 0 <= holding <= 1:
            raise ValueError("constant holding must be in [0,1]")
        self.holding = holding

    def reset(self):
        pass

    def act_batch(self, obs):
        return np.full(len(obs), self.holding, dtype=np.float32)

    def act(self, obs):
        return np.array([self.holding], dtype=np.float32)


class PeriodicDelta(EstimatedVolatilityDelta):
    def __init__(self, market, interval=5):
        super().__init__(market)
        if interval < 1 or int(interval) != interval:
            raise ValueError("rebalancing interval must be a positive integer")
        self.interval, self.t = interval, 0

    def reset(self):
        self.t = 0

    def act_batch(self, obs):
        action = super().act_batch(obs) if self.t % self.interval == 0 else obs[:, 2].copy()
        self.t += 1
        return action

    def act(self, obs):
        return self.act_batch(np.asarray(obs)[None]).reshape(1)


class DeltaBand(EstimatedVolatilityDelta):
    def __init__(self, market, width=0.05):
        super().__init__(market)
        if not 0 <= width <= 1:
            raise ValueError("band width must be in [0,1]")
        self.width = width

    def act_batch(self, obs):
        delta = super().act_batch(obs)
        return np.clip(obs[:, 2], np.maximum(0, delta-self.width), np.minimum(1, delta+self.width))

    def act(self, obs):
        return self.act_batch(np.asarray(obs)[None]).reshape(1)
