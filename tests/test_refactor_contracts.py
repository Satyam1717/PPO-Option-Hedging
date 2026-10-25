"""Public API compatibility and mathematical checks for the reorganized code."""
import numpy as np
import pytest
import torch
from hedging.agents.delta_hedge import FixedVolatilityDelta, StaticBSBaseline
from hedging.agents.heuristics import ConstantHedge, PeriodicDelta
from hedging.agents.optimization import clipped_policy_loss
from hedging.config import read_experiment_config, load_config
from hedging.env.accounting import advance_portfolio
from hedging.evaluation.metrics import compare_paired_utilities, summarize_hedging
from hedging.exceptions import EvaluationError, SimulationError
from hedging.market.gbm import GBMSimulator
from hedging.market.switching import SwitchingSimulator
from hedging.pricing.black_scholes import call_value, call_hedge_ratio, call_sensitivities


def test_compatibility_aliases_use_the_current_implementation():
    assert read_experiment_config is load_config
    assert StaticBSBaseline is FixedVolatilityDelta
    model = GBMSimulator(100.,.02,.2,1/252,21)
    assert np.array_equal(model.generate_paths(4,np.random.default_rng(9)),
                          model.simulate_paths(4,np.random.default_rng(9)))


def test_legacy_keyword_arguments_are_still_supported():
    model = GBMSimulator(100.,.02,.2,1/252,21)
    assert model.simulate_paths(n_paths=2,rng=np.random.default_rng(1)).shape == (2,22)
    market = read_experiment_config('config/config.yaml').market
    baseline = StaticBSBaseline(market,static_vol=.2)
    obs = np.array([0.,1.,0.,.2,.025,.001,.025],dtype=np.float32)
    assert baseline.act(obs=obs).shape == (1,)


@pytest.mark.parametrize('invalid',[0,-1,1.5])
def test_path_counts_must_be_positive_integers(invalid):
    with pytest.raises(SimulationError):
        GBMSimulator(100.,.02,.2,1/252,21).generate_paths(invalid,np.random.default_rng(1))


def test_switching_model_rejects_an_impossible_one_step_switch():
    with pytest.raises(SimulationError):
        SwitchingSimulator(100.,.02,.2,.5,1/252,1)


def test_one_step_cash_settlement_matches_independent_arithmetic():
    config = read_experiment_config('config/config.yaml')
    step = advance_portfolio(3.,0.,.5,100.,104.,.001,config.market,0.,.2)
    expected_cash = (3.-50.-.05)*np.exp(config.market.r*config.market.dt)
    expected_payoff = 4.
    expected_unwind = .052
    expected_pnl = expected_cash+52.-expected_payoff-expected_unwind
    assert step.cash == pytest.approx(expected_cash)
    assert step.hedge_error == pytest.approx(expected_pnl)
    assert step.liquidation_fee == pytest.approx(expected_unwind)


@pytest.mark.parametrize('time_left',[0.,1/252,.5])
def test_sensitivities_share_consistent_price_and_delta(time_left):
    spot = np.array([80.,100.,120.])
    greeks = call_sensitivities(spot,100.,time_left,.02,.25)
    assert np.array_equal(greeks.price,call_value(spot,100.,time_left,.02,.25))
    assert np.array_equal(greeks.delta,call_hedge_ratio(spot,100.,time_left,.02,.25))


def test_ppo_clipping_handles_positive_and_negative_advantages():
    ratio = torch.tensor([3.,3.])
    advantage = torch.tensor([1.,-1.])
    # Positive advantage is clipped to 1.2; negative advantage keeps -3.
    assert clipped_policy_loss(ratio,advantage,.2).item() == pytest.approx(.9)


def test_evaluation_rejects_non_vector_or_invalid_bootstrap_inputs():
    with pytest.raises(EvaluationError):
        summarize_hedging(np.zeros((2,2)),np.zeros(2),np.zeros(2))
    with pytest.raises(EvaluationError):
        compare_paired_utilities([0.,1.],[0.,1.],n_bootstrap=0)


def test_heuristic_parameters_must_define_valid_trades():
    market = read_experiment_config('config/config.yaml').market
    with pytest.raises(ValueError): ConstantHedge(1.5)
    with pytest.raises(ValueError): PeriodicDelta(market,0)
