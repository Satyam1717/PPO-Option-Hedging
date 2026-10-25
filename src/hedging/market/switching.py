"""Geometric price paths whose volatility changes once during the episode."""
import numpy as np
from hedging.exceptions import SimulationError
from hedging.market.path_tools import draw_geometric_paths, require_positive_count


class SwitchingSimulator:
    def __init__(self, S0, mu, sigma, sigma_after, dt, n_steps, switch_fraction=0.5):
        if (not np.isfinite([S0,mu,sigma,sigma_after,dt,switch_fraction]).all()
                or min(S0,sigma,sigma_after,dt) <= 0 or not 0 < switch_fraction < 1):
            raise SimulationError("invalid volatility-switch parameters")
        self.n_steps = require_positive_count(n_steps, "time steps")
        if self.n_steps < 2:
            raise SimulationError("a volatility switch requires at least two time steps")
        self.S0, self.mu, self.dt = S0, mu, dt
        self.switch_day = max(1, min(self.n_steps-1, int(self.n_steps*switch_fraction)))
        self.sigmas = np.where(np.arange(self.n_steps) < self.switch_day, sigma, sigma_after)

    def generate_paths(self, count, rng):
        return draw_geometric_paths(self.S0,self.mu,self.sigmas,self.dt,count,self.n_steps,rng)

    def generate_path(self, rng):
        return self.generate_paths(1,rng)[0]

    def simulate_paths(self, n_paths, rng):
        return self.generate_paths(n_paths, rng)

    simulate_path = generate_path
