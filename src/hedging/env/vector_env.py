"""NumPy batch simulation sharing accounting with the reference Gym environment.

Fixed-length episodes reset together. Checkpoints include paths, partial cash
balances and RNG state. Policies see only prices up to the current time index.
"""
from __future__ import annotations
import copy
import numpy as np
from hedging.env.accounting import (
    OBS_DIM, advance_portfolio, make_observation, estimate_trailing_volatility, quadratic_utility,
)
from hedging.pricing.premium import quote_initial_option


class VectorHedgingEnv:
    def __init__(self, market, hedging, regimes, simulators, n_envs, seed,
                 paths=None, fixed_regime=None, cost_rate=None, auto_reset=True):
        self.market, self.hedging, self.regimes, self.simulators = market, hedging, regimes, simulators
        self.n_envs, self.obs_dim = n_envs, OBS_DIM
        self.rng = np.random.default_rng(seed)
        self.fixed_paths, self.fixed_regime, self.fixed_cost = paths, fixed_regime, cost_rate
        self.auto_reset = auto_reset
        self.quotes = {name: quote_initial_option(market, (hedging.premium_mc_paths, hedging.premium_seed), r)
                       for name, r in regimes.items()}

    def reset(self):
        names = list(self.regimes)
        self.names = (np.full(self.n_envs, self.fixed_regime, dtype=object) if self.fixed_regime is not None
                      else self.rng.choice(names, self.n_envs))
        self.paths = np.zeros((self.n_envs, self.market.n_steps + 1))
        self.premium, self.prior = np.zeros(self.n_envs), np.zeros(self.n_envs)
        for name in names:
            ids = np.flatnonzero(self.names == name)
            if len(ids):
                self.paths[ids] = (np.asarray(self.fixed_paths)[ids] if self.fixed_paths is not None
                                   else self.simulators[name].generate_paths(len(ids), self.rng))
                self.premium[ids], self.prior[ids] = self.quotes[name][:2]
        if not np.isfinite(self.paths).all() or np.any(self.paths <= 0):
            raise ValueError("invalid batch paths")
        if self.fixed_cost is not None:
            self.costs = np.full(self.n_envs, self.fixed_cost)
        elif self.hedging.training_cost_rates:
            self.costs = self.rng.choice(self.hedging.training_cost_rates, self.n_envs)
        else:
            self.costs = np.full(self.n_envs, self.hedging.transaction_cost_rate)
        self.cash = self.premium.copy()
        self.previous = np.zeros(self.n_envs)
        self.potential = np.zeros(self.n_envs)
        self.t = 0
        return self._obs()

    def _obs(self):
        vol = estimate_trailing_volatility(self.paths, self.t, self.market, self.prior, self.hedging.vol_prior_strength)
        return make_observation(self.paths[:, self.t], (self.market.n_steps - self.t) / self.market.n_steps,
                           self.previous, vol, self.cash, self.costs, self.premium, self.market)

    def step(self, actions):
        if self.t >= self.market.n_steps:
            raise ValueError("reset a finished batch before stepping")
        h = np.asarray(actions).reshape(self.n_envs).clip(0, 1)
        if not np.isfinite(h).all():
            raise ValueError("non-finite actions")
        spot, nxt = self.paths[:, self.t], self.paths[:, self.t + 1]
        self.t += 1
        done = self.t == self.market.n_steps
        vol = estimate_trailing_volatility(self.paths,self.t,self.market,self.prior,
                                           self.hedging.vol_prior_strength)
        transition = advance_portfolio(self.cash,self.previous,h,spot,nxt,self.costs,self.market,
                                        (self.market.n_steps-self.t)*self.market.dt,vol)
        self.cash, self.previous = transition.cash, h.copy()
        info = {"cost":transition.trading_fee,"hedge_pnl":transition.stock_gain}
        if done:
            info.update(total_pnl=transition.hedge_error.copy(),
                        final_unwind_cost=transition.liquidation_fee,regimes=self.names.copy())
        potential = quadratic_utility(transition.hedge_error,self.hedging.variance_penalty_lambda)
        rewards = (potential - self.potential).astype(np.float32)
        self.potential = potential
        obs = self.reset() if done and self.auto_reset else self._obs()
        return obs, rewards, np.full(self.n_envs, done, dtype=np.float32), info

    def state_dict(self):
        return copy.deepcopy({k: v for k, v in self.__dict__.items()
                              if k not in ("market", "hedging", "regimes", "simulators", "quotes", "rng")}
                             | {"rng_state": self.rng.bit_generator.state})

    def load_state_dict(self, state):
        state = copy.deepcopy(state)
        self.rng.bit_generator.state = state.pop("rng_state")
        self.__dict__.update(state)
