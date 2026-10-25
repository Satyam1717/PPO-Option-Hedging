"""Daily Black-Scholes strategies with a shared scalar/batch implementation."""
from __future__ import annotations
import numpy as np
from hedging.pricing.black_scholes import call_hedge_ratio


class DeltaStrategy:
    def __init__(self, market):
        self.market = market

    def volatility_for(self, observations):
        raise NotImplementedError

    def act_batch(self, obs):
        observations = np.asarray(obs)
        spot = self.market.K * np.exp(observations[:,0].astype(float))
        time_left = observations[:,1].astype(float) * self.market.maturity
        positions = call_hedge_ratio(spot,self.market.K,time_left,self.market.r,
                                  self.volatility_for(observations))
        return np.asarray(positions,dtype=np.float32)

    def act(self, obs):
        return self.act_batch(np.asarray(obs)[None]).reshape(1)

    def reset(self):
        """There is no private trading history in a daily delta strategy."""


class FixedVolatilityDelta(DeltaStrategy):
    def __init__(self, market, static_vol):
        super().__init__(market)
        if not np.isfinite(static_vol) or static_vol <= 0:
            raise ValueError("fixed volatility must be finite and positive")
        self.static_vol = float(static_vol)

    def volatility_for(self, observations):
        return self.static_vol


class EstimatedVolatilityDelta(DeltaStrategy):
    def volatility_for(self, observations):
        return observations[:,3]


StaticBSBaseline = FixedVolatilityDelta
AdaptiveBSBaseline = EstimatedVolatilityDelta
