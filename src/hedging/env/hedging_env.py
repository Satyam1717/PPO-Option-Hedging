"""Financed short-call hedge with seven observations and utility-difference rewards.

Cash pays for trades, then accrues at r. Intermediate BS liability marks use
observable returns only. Their utility differences telescope to terminal utility;
the mark is reward shaping, not an oracle Heston price. Use gamma=1.
"""
from __future__ import annotations
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from hedging.env.accounting import (
    OBS_DIM, advance_portfolio, make_observation, estimate_trailing_volatility, quadratic_utility,
)
from hedging.exceptions import HedgingEnvError
from hedging.pricing.premium import quote_initial_option


class HedgingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, market, hedging, regimes, simulators, rng=None, fixed_regime=None):
        super().__init__()
        if fixed_regime is not None and fixed_regime not in regimes:
            raise HedgingEnvError("unknown fixed_regime")
        self.market, self.hedging = market, hedging
        self.regimes, self.simulators = regimes, simulators
        self.fixed_regime = fixed_regime
        self._rng = rng if rng is not None else np.random.default_rng()
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
        self.action_space = spaces.Box(0.0, 1.0, (1,), np.float32)
        self._initialized = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        options = options or {}
        name = options.get("regime", self.fixed_regime)
        if name is None:
            name = self._rng.choice(list(self.regimes))
        if name not in self.regimes:
            raise HedgingEnvError(f"unknown regime {name!r}")
        path = options.get("path")
        if path is None:
            path = self.simulators[name].generate_path(self._rng)
        self._path = np.asarray(path, dtype=float)
        if (self._path.shape != (self.market.n_steps + 1,) or not np.isfinite(self._path).all()
                or np.any(self._path <= 0) or not np.isclose(self._path[0], self.market.S0)):
            raise HedgingEnvError("path must start at S0 and contain n_steps+1 finite positive prices")
        self._regime_name = name
        self._C0, self._prior, self._premium_se = quote_initial_option(
            self.market, (self.hedging.premium_mc_paths, self.hedging.premium_seed), self.regimes[name])
        self._cost_rate = float(options.get("cost_rate", self.hedging.transaction_cost_rate))
        if options.get("randomize_cost", False) and self.hedging.training_cost_rates:
            self._cost_rate = float(self._rng.choice(self.hedging.training_cost_rates))
        if self._cost_rate < 0 or not np.isfinite(self._cost_rate):
            raise HedgingEnvError("cost_rate must be finite and nonnegative")
        self._t, self._prev_hedge, self._cash = 0, 0.0, self._C0
        self._potential = 0.0
        self._initialized = True
        return self._build_obs(), {"regime": name, "C0": self._C0, "premium_se": self._premium_se,
                                   "cost_rate": self._cost_rate}

    def step(self, action):
        if not self._initialized or self._t >= self.market.n_steps:
            raise HedgingEnvError("reset before stepping a new or finished episode")
        a = np.asarray(action).reshape(-1)
        if a.size != 1 or not np.isfinite(a).all():
            raise HedgingEnvError("action must contain one finite number")
        h = float(np.clip(a[0], 0, 1))
        spot, nxt = self._path[self._t:self._t + 2]
        self._t += 1
        done = self._t == self.market.n_steps
        remaining_time = (self.market.n_steps-self._t)*self.market.dt
        vol = estimate_trailing_volatility(self._path,self._t,self.market,self._prior,
                                           self.hedging.vol_prior_strength)
        transition = advance_portfolio(self._cash,self._prev_hedge,h,spot,nxt,self._cost_rate,
                                        self.market,remaining_time,vol)
        self._cash, self._prev_hedge = transition.cash, h
        info = {"regime": self._regime_name,"cost":float(transition.trading_fee),
                "hedge_pnl":float(transition.stock_gain)}
        if done:
            info.update(total_pnl=float(transition.hedge_error),
                        final_unwind_cost=float(transition.liquidation_fee), payoff=float(transition.payoff),
                        C0=self._C0, terminal_cash=float(self._cash))
        potential = float(quadratic_utility(transition.hedge_error,self.hedging.variance_penalty_lambda))
        reward = potential - self._potential
        self._potential = potential
        return self._build_obs(), reward, done, False, info

    def _build_obs(self):
        vol = estimate_trailing_volatility(self._path, self._t, self.market, self._prior, self.hedging.vol_prior_strength)
        return make_observation(self._path[self._t], (self.market.n_steps - self._t) / self.market.n_steps,
                           self._prev_hedge, vol, self._cash, self._cost_rate, self._C0, self.market)
