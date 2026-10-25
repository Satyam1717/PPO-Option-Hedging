"""Risk and P&L metrics for comparing hedging methods.

All metrics operate on arrays collected across many evaluation episodes
(one `total_pnl` / `total_cost` / `turnover` value per episode).
"""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from hedging.exceptions import EvaluationError
from hedging.env.accounting import quadratic_utility


@dataclass(frozen=True)
class Metrics:
    n_episodes: int
    mean_pnl: float
    std_pnl: float
    var_5: float
    cvar_5: float
    mean_total_cost: float
    mean_turnover: float
    sharpe_like: float
    rmse: float
    mean_utility: float
    loss_probability: float

    def as_dict(self) -> dict:
        return asdict(self)


def _cvar(pnls: np.ndarray, alpha: float = 0.05) -> float:
    """Expected Shortfall at the alpha level: mean of the worst alpha-fraction
    of outcomes (most negative P&L). A coherent tail-risk measure, unlike VaR."""
    sorted_pnls = np.sort(pnls)
    cutoff = max(1, int(np.ceil(alpha * len(sorted_pnls))))
    return float(np.mean(sorted_pnls[:cutoff]))


def _var(pnls: np.ndarray, alpha: float = 0.05) -> float:
    return float(np.percentile(pnls, alpha * 100))


def summarize_hedging(
    total_pnls: np.ndarray, total_costs: np.ndarray, turnovers: np.ndarray, risk_aversion: float = 0.3
) -> Metrics:
    total_pnls = np.asarray(total_pnls, dtype=float)
    total_costs = np.asarray(total_costs, dtype=float)
    turnovers = np.asarray(turnovers, dtype=float)
    if any(x.ndim != 1 for x in (total_pnls,total_costs,turnovers)):
        raise EvaluationError("metrics require one-dimensional outcome arrays")
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise EvaluationError("risk aversion must be finite and nonnegative")
    if not all(np.isfinite(x).all() for x in (total_pnls, total_costs, turnovers)):
        raise EvaluationError("metrics require finite values")

    if len(total_pnls) == 0:
        raise EvaluationError("cannot compute metrics over zero episodes")
    if not (len(total_pnls) == len(total_costs) == len(turnovers)):
        raise EvaluationError("total_pnls/total_costs/turnovers length mismatch")

    std_pnl = float(np.std(total_pnls))
    mean_pnl = float(np.mean(total_pnls))
    sharpe_like = mean_pnl / std_pnl if std_pnl > 1e-12 else float("nan")

    return Metrics(
        n_episodes=len(total_pnls),
        mean_pnl=mean_pnl,
        std_pnl=std_pnl,
        var_5=_var(total_pnls, 0.05),
        cvar_5=_cvar(total_pnls, 0.05),
        mean_total_cost=float(np.mean(total_costs)),
        mean_turnover=float(np.mean(turnovers)),
        sharpe_like=sharpe_like,
        rmse=float(np.sqrt(np.mean(total_pnls**2))),
        mean_utility=float(np.mean(quadratic_utility(total_pnls,risk_aversion))),
        loss_probability=float(np.mean(total_pnls < 0)),
    )


def compare_paired_utilities(first, second, risk_aversion=0.3, n_bootstrap=1000, seed=90210):
    """Path-paired bootstrap for one trained policy; excludes training-seed uncertainty."""
    first, second = np.asarray(first), np.asarray(second)
    if first.shape != second.shape or first.ndim != 1 or len(first) < 2:
        raise EvaluationError("paired bootstrap needs equally-sized 1D samples, n >= 2")
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise EvaluationError("paired samples must be finite")
    if n_bootstrap < 1 or int(n_bootstrap) != n_bootstrap:
        raise EvaluationError("bootstrap count must be a positive integer")
    if not np.isfinite(risk_aversion) or risk_aversion < 0:
        raise EvaluationError("risk aversion must be finite and nonnegative")
    delta = quadratic_utility(first,risk_aversion)-quadratic_utility(second,risk_aversion)
    rng = np.random.default_rng(seed)
    # Chunk to bound peak allocation independently of the requested bootstrap size.
    means = []
    for start in range(0, n_bootstrap, 100):
        ids = rng.integers(0, len(delta), (min(100, n_bootstrap-start), len(delta)))
        means.extend(delta[ids].mean(axis=1))
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(delta.mean()), float(lo), float(hi)


compute_metrics = summarize_hedging
paired_utility_interval = compare_paired_utilities
