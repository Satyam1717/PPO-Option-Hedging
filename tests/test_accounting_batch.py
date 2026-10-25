from dataclasses import replace
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from hedging.config import load_config
from hedging.env.accounting import rebalance_and_accrue
from hedging.env.hedging_env import HedgingEnv
from hedging.env.vector_env import VectorHedgingEnv
from hedging.exceptions import HedgingEnvError
from hedging.market.registry import build_all_simulators


def test_one_step_financing_matches_hand_calculation():
    cash, fee = rebalance_and_accrue(3., 0.2, 0.7, 100., .05, .25, .002)
    assert fee == pytest.approx(.1)
    assert cash == pytest.approx((3.-50.-.1)*np.exp(.05*.25))


def test_batch_and_gym_match_every_step_including_terminal_cost_and_rewards():
    cfg = load_config('config/config.yaml')
    sims = build_all_simulators(cfg)
    name = 'gbm_high_vol'
    paths = sims[name].simulate_paths(4, np.random.default_rng(17))
    batch = VectorHedgingEnv(cfg.market, cfg.hedging, cfg.regimes, sims, 4, 3, paths, name, .002, False)
    refs = [HedgingEnv(cfg.market, cfg.hedging, cfg.regimes, sims, fixed_regime=name) for _ in range(4)]
    expected = [e.reset(options={'path': p, 'cost_rate': .002})[0] for e,p in zip(refs,paths)]
    assert np.allclose(batch.reset(), expected)
    rewards = np.zeros(4)
    rng = np.random.default_rng(23)
    for _ in range(cfg.market.n_steps):
        h = rng.uniform(0,1,4)
        obs, r, done, info = batch.step(h)
        rewards += r
        for i,e in enumerate(refs):
            o, rr, dd, _, ii = e.step([h[i]])
            assert np.allclose(o,obs[i],atol=1e-6)
            assert r[i] == pytest.approx(rr, abs=1e-6)
            assert bool(done[i]) == dd
            if dd:
                assert info['total_pnl'][i] == pytest.approx(ii['total_pnl'],abs=1e-9)
    x = info['total_pnl']
    assert np.allclose(rewards, x-cfg.hedging.variance_penalty_lambda*x*x, atol=1e-5)


def test_observation_does_not_reveal_future_prices():
    cfg = load_config('config/config.yaml'); sims = build_all_simulators(cfg)
    name='gbm_low_vol'; p=sims[name].simulate_path(np.random.default_rng(5)); q=p.copy(); q[8:]*=1.3
    envs=[HedgingEnv(cfg.market,cfg.hedging,cfg.regimes,sims,fixed_regime=name) for _ in range(2)]
    obs=[e.reset(options={'path':x})[0] for e,x in zip(envs,[p,q])]
    assert np.array_equal(*obs)
    for _ in range(7):
        obs=[e.step([.5])[0] for e in envs]
        assert np.array_equal(*obs)


def test_cost_observation_and_seed_are_reproducible():
    cfg=load_config('config/config.yaml'); sims=build_all_simulators(cfg)
    env=HedgingEnv(cfg.market,cfg.hedging,cfg.regimes,sims)
    first,info=env.reset(seed=20,options={'cost_rate':.003})
    second,_=env.reset(seed=20,options={'cost_rate':.003})
    assert np.array_equal(first,second)
    assert first[5] == pytest.approx(.003)
    assert 'path' not in info
    with pytest.raises(HedgingEnvError): env.step([np.nan])


def test_gymnasium_api_contract():
    cfg=load_config('config/config.yaml'); sims=build_all_simulators(cfg)
    check_env(HedgingEnv(cfg.market,cfg.hedging,cfg.regimes,sims),skip_render_check=True)
