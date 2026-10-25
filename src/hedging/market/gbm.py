"""A constant-volatility geometric price model with exact daily transitions."""
from __future__ import annotations
import numpy as np
from hedging.exceptions import SimulationError
from hedging.market.path_tools import draw_geometric_paths, require_positive_count


class GBMSimulator:
    def __init__(self, S0: float, mu: float, sigma: float, dt: float, n_steps: int):
        parameters = np.asarray([S0, mu, sigma, dt], dtype=float)
        if not np.isfinite(parameters).all() or min(S0, sigma, dt) <= 0:
            raise SimulationError("GBM requires finite inputs and positive spot, volatility and time step")
        self.S0, self.mu, self.sigma, self.dt = map(float, parameters)
        self.n_steps = require_positive_count(n_steps, "time steps")

    def generate_paths(self, count: int, rng: np.random.Generator) -> np.ndarray:
        return draw_geometric_paths(self.S0, self.mu, self.sigma, self.dt, count, self.n_steps, rng)

    def generate_path(self, rng: np.random.Generator) -> np.ndarray:
        return self.generate_paths(1, rng)[0]

    # Existing notebooks and recorded experiments used these method names.
    def simulate_paths(self, n_paths, rng):
        return self.generate_paths(n_paths, rng)

    simulate_path = generate_path
