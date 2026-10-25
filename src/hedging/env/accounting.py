"""Shared scalar/batch accounting for a self-financing short-call hedge."""
from dataclasses import dataclass
import numpy as np
from hedging.pricing.black_scholes import call_value


OBS_DIM = 7
OBSERVATION_NAMES = ("log_moneyness", "time_remaining", "stock_position", "volatility_estimate",
                     "cash_over_initial_spot", "transaction_cost_rate", "premium_over_initial_spot")


def execute_rebalance(cash, previous, target, spot, rate, dt, cost_rate):
    """Pay for the stock trade and its fee, then accrue the resulting cash."""
    cost = cost_rate * spot * np.abs(target - previous)
    after_trade = cash - (target - previous) * spot - cost
    return after_trade * np.exp(rate * dt), cost


def quadratic_utility(pnl, risk_aversion):
    return pnl - risk_aversion * np.square(pnl)


def make_observation(spot, tau_norm, previous, vol, cash, cost, premium, market):
    values = np.broadcast_arrays(np.log(spot / market.K), tau_norm, previous, vol,
                                 cash / market.S0, cost, premium / market.S0)
    return np.stack(values, axis=-1).astype(np.float32)


def estimate_trailing_volatility(paths, t, market, prior, strength):
    if t == 0:
        return np.asarray(prior)
    start = max(0, t - market.vol_window)
    returns = np.log(paths[..., start + 1:t + 1] / paths[..., start:t])
    return np.sqrt((np.sum(returns**2, axis=-1) / market.dt + strength * np.square(prior))
                   / (t - start + strength))


@dataclass(frozen=True)
class PortfolioTransition:
    cash: object
    trading_fee: object
    stock_gain: object
    hedge_error: object
    payoff: object = 0.0
    liquidation_fee: object = 0.0


def advance_portfolio(cash, previous, target, spot, next_spot, cost_rate, market,
                      remaining_time, volatility):
    """One shared cash/valuation rule for both scalar and batched environments."""
    next_cash, fee = execute_rebalance(cash,previous,target,spot,market.r,market.dt,cost_rate)
    stock_gain = target * (next_spot-spot)
    if remaining_time == 0:
        liquidation_fee = cost_rate * next_spot * target
        payoff = np.maximum(next_spot-market.K,0.0)
        error = next_cash + target*next_spot - liquidation_fee - payoff
        return PortfolioTransition(next_cash,fee,stock_gain,error,payoff,liquidation_fee)
    option_mark = call_value(next_spot,market.K,remaining_time,market.r,volatility)
    error = next_cash + target*next_spot - option_mark
    return PortfolioTransition(next_cash,fee,stock_gain,error)


# Compatibility with earlier scripts and saved experiment descriptions.
rebalance_and_accrue = execute_rebalance
utility = quadratic_utility
observation = make_observation
trailing_vol = estimate_trailing_volatility
