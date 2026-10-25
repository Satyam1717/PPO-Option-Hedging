"""European-call pricing and sensitivities, sharing one set of calculation terms."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.stats import norm

_MIN_TAU = 1e-8
_MIN_SIGMA = 1e-8


@dataclass(frozen=True)
class Greeks:
    price: np.ndarray
    delta: np.ndarray
    gamma: np.ndarray
    vega: np.ndarray
    theta: np.ndarray


@dataclass(frozen=True)
class _CallTerms:
    spot: np.ndarray
    strike: np.ndarray
    time: np.ndarray
    volatility: np.ndarray
    root_time: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    discount: np.ndarray
    expired: np.ndarray


def _prepare_call(spot, strike, time_left, rate, volatility):
    spot = np.asarray(spot,dtype=float)
    strike = np.asarray(strike,dtype=float)
    time_left = np.asarray(time_left,dtype=float)
    safe_time = np.maximum(time_left,_MIN_TAU)
    safe_volatility = np.maximum(np.asarray(volatility,dtype=float),_MIN_SIGMA)
    root_time = np.sqrt(safe_time)
    d1 = (np.log(spot/strike)+(rate+0.5*safe_volatility**2)*safe_time)/(safe_volatility*root_time)
    d2 = d1-safe_volatility*root_time
    return _CallTerms(spot,strike,safe_time,safe_volatility,root_time,d1,d2,
                      np.exp(-rate*safe_time),time_left<=_MIN_TAU)


def _price_from_terms(terms, first_probability, second_probability):
    present_value = (terms.spot*first_probability
                     - terms.strike*terms.discount*second_probability)
    return np.where(terms.expired,np.maximum(terms.spot-terms.strike,0.0),present_value)


def _delta_from_terms(terms, first_probability):
    expiry_position = (terms.spot>terms.strike).astype(float)
    return np.where(terms.expired,expiry_position,first_probability)


def call_value(spot, strike, time_left, rate, volatility):
    terms = _prepare_call(spot,strike,time_left,rate,volatility)
    return _price_from_terms(terms,norm.cdf(terms.d1),norm.cdf(terms.d2))


def call_hedge_ratio(spot, strike, time_left, rate, volatility):
    terms = _prepare_call(spot,strike,time_left,rate,volatility)
    return _delta_from_terms(terms,norm.cdf(terms.d1))


def call_sensitivities(spot, strike, time_left, rate, volatility):
    terms = _prepare_call(spot,strike,time_left,rate,volatility)
    first_probability, second_probability = norm.cdf(terms.d1), norm.cdf(terms.d2)
    density = norm.pdf(terms.d1)
    gamma = density/(terms.spot*terms.volatility*terms.root_time)
    vega = terms.spot*density*terms.root_time
    theta = (-(terms.spot*density*terms.volatility)/(2*terms.root_time)
             - rate*terms.strike*terms.discount*second_probability)
    return Greeks(
        _price_from_terms(terms,first_probability,second_probability),
        _delta_from_terms(terms,first_probability),
        np.where(terms.expired,0.0,gamma),
        np.where(terms.expired,0.0,vega),
        np.where(terms.expired,0.0,theta),
    )


# Preserve established formula-style keywords for older callers.
def bs_call_price(S, K, tau, r, sigma):
    return call_value(S,K,tau,r,sigma)


def bs_call_delta(S, K, tau, r, sigma):
    return call_hedge_ratio(S,K,tau,r,sigma)


def bs_call_greeks(S, K, tau, r, sigma):
    return call_sensitivities(S,K,tau,r,sigma)
