from dataclasses import replace
import numpy as np
import pytest
from hedging.agents.heuristics import ConstantHedge, DeltaBand, PeriodicDelta
from hedging.config import load_config
from hedging.evaluation.batch_backtest import batch_backtest, paths_for, select_band_width
from hedging.evaluation.metrics import compute_metrics, paired_utility_interval
from hedging.exceptions import EvaluationError, ConfigError


def test_utility_and_rmse_use_terminal_outcomes():
    p=np.array([-2.,0.,1.])
    m=compute_metrics(p,np.zeros(3),np.zeros(3),.4)
    assert m.rmse == pytest.approx(np.sqrt(5/3))
    assert m.mean_utility == pytest.approx(np.mean(p-.4*p*p))
    with pytest.raises(EvaluationError): compute_metrics([np.nan],[0],[0])


def test_identical_paired_samples_have_zero_interval():
    p=np.array([-2.,0.,1.,3.])
    assert paired_utility_interval(p,p,n_bootstrap=100) == (0.,0.,0.)


def test_band_and_periodic_policies_can_hold_existing_position():
    cfg=load_config('config/config.yaml')
    obs=np.array([[0.,.5,.5,.2,-.45,.002,.025]],dtype=np.float32)
    assert DeltaBand(cfg.market,.1).act_batch(obs)[0] == pytest.approx(.5)
    periodic=PeriodicDelta(cfg.market,5)
    periodic.act_batch(obs)
    obs[0,0]=.15
    assert periodic.act_batch(obs)[0] == pytest.approx(.5)


def test_validation_tuning_is_independent_of_test_seed_and_test_regimes():
    cfg=load_config('config/config.yaml')
    cfg=replace(cfg,evaluation=replace(cfg.evaluation,n_validation_episodes=16,band_candidates=[.0,.1]))
    a=select_band_width(cfg,.002)
    altered=replace(cfg,test_regimes={},evaluation=replace(cfg.evaluation,seed=9999))
    b=select_band_width(altered,.002)
    assert a == b


def test_path_seed_does_not_depend_on_other_regime_order():
    cfg=load_config('config/config.yaml'); r=cfg.regimes['gbm_low_vol']
    assert np.array_equal(paths_for(cfg,r,10,123),paths_for(cfg,r,10,123))
    assert not np.array_equal(paths_for(cfg,r,10,123),paths_for(cfg,r,10,456))


def test_discounted_objective_is_rejected():
    cfg=load_config('config/config.yaml')
    with pytest.raises(ConfigError): replace(cfg.training,gamma=.99)
    with pytest.raises(ConfigError): replace(cfg.training,n_minibatches=0)
