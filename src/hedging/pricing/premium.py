"""Inception quotes, calculated independently of training/evaluation paths."""
from dataclasses import replace
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq

from hedging.market.registry import create_price_model
from hedging.pricing.black_scholes import call_value


@lru_cache(maxsize=256)
def quote_initial_option(market, hedging, regime):
    # RegimeConfig/MarketConfig are immutable. HedgingConfig contains a list, so
    # callers pass only the numerical pricing settings, not the entire config.
    n_mc, seed = hedging
    if regime.type == "gbm":
        vol = regime.sigma
        return float(call_value(market.S0, market.K, market.maturity, market.r, vol)), vol, 0.0
    sim = create_price_model(replace(regime, mu=market.r), market)
    if regime.type == "switching":
        vol = float(np.sqrt(np.mean(sim.sigmas**2)))
        return float(call_value(market.S0, market.K, market.maturity, market.r, vol)), vol, 0.0
    paths = sim.generate_paths(n_mc, np.random.default_rng(seed))
    discounted = np.exp(-market.r * market.maturity) * np.maximum(paths[:, -1] - market.K, 0)
    quote = float(discounted.mean())
    se = float(discounted.std(ddof=1) / np.sqrt(n_mc))
    implied = brentq(lambda v: float(call_value(market.S0, market.K, market.maturity,
                                                 market.r, v)) - quote, 1e-5, 5.0)
    return quote, implied, se


inception_quote = quote_initial_option
