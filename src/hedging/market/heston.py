"""A stochastic-volatility price model using projected Euler variance.

The variance tends toward theta and receives shocks correlated with stock
returns. Each raw variance update is floored at zero. This is projected Euler,
not the full-truncation scheme that retains a separate raw variance state.
"""
from __future__ import annotations
import numpy as np
from hedging.exceptions import SimulationError
from hedging.market.path_tools import check_price_paths, require_positive_count


class HestonSimulator:
    def __init__(self, S0, mu, v0, kappa, theta, xi, rho, dt, n_steps):
        values = dict(S0=S0, mu=mu, v0=v0, kappa=kappa, theta=theta, xi=xi, rho=rho, dt=dt)
        if not np.isfinite(list(values.values())).all():
            raise SimulationError("Heston parameters must be finite")
        if min(S0, v0, kappa, theta, xi, dt) <= 0 or not -1 <= rho <= 1:
            raise SimulationError("Heston requires positive scale parameters and rho in [-1,1]")
        for name, value in values.items():
            setattr(self, name, float(value))
        self.n_steps = require_positive_count(n_steps, "time steps")

    def _correlated_shocks(self, count, rng):
        stock_shock = rng.standard_normal(count)
        independent_shock = rng.standard_normal(count)
        variance_shock = self.rho * stock_shock + np.sqrt(max(1-self.rho**2, 0.0)) * independent_shock
        return stock_shock, variance_shock

    def _advance_market(self, spot, variance, stock_shock, variance_shock):
        usable_variance = np.maximum(variance, 0.0)
        diffusion_scale = np.sqrt(usable_variance * self.dt)
        next_spot = spot * np.exp((self.mu-0.5*usable_variance)*self.dt + diffusion_scale*stock_shock)
        next_variance = (variance + self.kappa*(self.theta-usable_variance)*self.dt
                         + self.xi*diffusion_scale*variance_shock)
        return next_spot, np.maximum(next_variance, 0.0)

    def generate_with_variance(self, count, rng):
        count = require_positive_count(count, "path count")
        prices = np.empty((count, self.n_steps+1))
        variances = np.empty_like(prices)
        prices[:, 0], variances[:, 0] = self.S0, self.v0
        for day in range(self.n_steps):
            stock_shock, variance_shock = self._correlated_shocks(count, rng)
            prices[:, day+1], variances[:, day+1] = self._advance_market(
                prices[:, day], variances[:, day], stock_shock, variance_shock)
        if not np.isfinite(variances).all():
            raise SimulationError("simulated variance is not finite")
        return check_price_paths(prices), variances

    def generate_paths(self, count, rng):
        return self.generate_with_variance(count, rng)[0]

    def generate_path(self, rng):
        return self.generate_paths(1, rng)[0]

    def simulate_paths(self, n_paths, rng):
        return self.generate_paths(n_paths, rng)

    simulate_path = generate_path

    def simulate_paths_with_variance(self, n_paths, rng):
        return self.generate_with_variance(n_paths, rng)

    _simulate = simulate_paths_with_variance
