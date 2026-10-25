"""Small shared operations for generating and checking simulated stock paths."""
from __future__ import annotations

from typing import Protocol
import numpy as np
from hedging.exceptions import SimulationError


class PriceModel(Protocol):
    def generate_paths(self, count: int, rng: np.random.Generator) -> np.ndarray: ...
    def generate_path(self, rng: np.random.Generator) -> np.ndarray: ...


def require_positive_count(value: int, name: str) -> int:
    if not np.isfinite(value) or value < 1 or int(value) != value:
        raise SimulationError(f"{name} must be a positive integer")
    return int(value)


def check_price_paths(prices: np.ndarray) -> np.ndarray:
    if not np.isfinite(prices).all() or np.any(prices <= 0):
        raise SimulationError("simulated prices must be finite and strictly positive")
    return prices


def accumulate_log_returns(initial_price: float, changes: np.ndarray) -> np.ndarray:
    """Turn a paths-by-time log-return matrix into prices including time zero."""
    prices = np.empty((len(changes), changes.shape[1] + 1), dtype=float)
    prices[:, 0] = initial_price
    prices[:, 1:] = initial_price * np.exp(np.cumsum(changes, axis=1))
    return check_price_paths(prices)


def draw_geometric_paths(initial_price, drift_rate, volatility, time_step, count, steps, rng):
    count = require_positive_count(count, "path count")
    shocks = rng.standard_normal((count, steps))
    deterministic_change = (drift_rate - 0.5 * np.square(volatility)) * time_step
    random_change = volatility * np.sqrt(time_step) * shocks
    return accumulate_log_returns(initial_price, deterministic_change + random_change)
