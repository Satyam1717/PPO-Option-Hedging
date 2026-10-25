from dataclasses import replace
import numpy as np
import pytest
import torch
from hedging.agents.ppo_agent import PPOAgent, compute_gae
from hedging.config import load_config
from hedging.env.vector_env import VectorHedgingEnv
from hedging.market.registry import build_all_simulators


def setup(kind='band', seed=7):
    c=load_config('config/config.yaml')
    tc=replace(c.training,total_timesteps=256,rollout_len=64,n_envs=4,n_epochs=2,
               n_minibatches=2,hidden_sizes=[16,16],seed=seed,policy_kind=kind)
    env=VectorHedgingEnv(c.market,c.hedging,c.regimes,build_all_simulators(c),4,seed)
    agent=PPOAgent(7,tc.hidden_sizes,seed=seed,market=c.market,policy_kind=kind)
    return agent,env,tc


@pytest.mark.parametrize('kind',['band','direct'])
def test_training_updates_weights_and_actions_are_bounded(kind):
    a,e,c=setup(kind)
    before=a.network.actor[0].weight.detach().clone()
    history=a.train(e,c)
    assert len(history)==4
    assert not torch.equal(before,a.network.actor[0].weight)
    obs=e.reset()
    assert np.all((a.act_batch(obs)>=0)&(a.act_batch(obs)<=1))
    assert np.isfinite([r['value_loss'] for r in history]).all()


def test_save_load_round_trip(tmp_path):
    a,e,c=setup(); a.train(e,c)
    path=tmp_path/'agent.pt'; a.save(path)
    b=PPOAgent.load(path); obs=e.reset()
    assert np.array_equal(a.act_batch(obs),b.act_batch(obs))


def test_resume_matches_uninterrupted_training_exactly(tmp_path):
    full,ef,c=setup(); full.train(ef,c)
    expected={k:v.clone() for k,v in full.network.state_dict().items()}
    half,eh,c=setup(); checkpoint=tmp_path/'checkpoint.pt'
    half.train(eh,c,checkpoint_path=checkpoint,max_updates=2)
    resumed,er,c=setup(seed=7)
    h=resumed.train(er,c,checkpoint_path=checkpoint,resume=True)
    assert h[-1]['total_steps']==256
    for k,v in resumed.network.state_dict().items():
        assert torch.equal(v,expected[k]),k
    assert np.array_equal(er.paths,ef.paths)
    assert np.array_equal(er.cash,ef.cash)


def test_gae_never_bootstraps_across_episode_termination():
    r=np.array([[1.],[2.]],dtype=np.float32); v=np.zeros_like(r)
    adv,_=compute_gae(r,v,np.array([[1.],[1.]]),np.array([999.]),1.,1.)
    assert np.array_equal(adv,r)


def test_cost_blind_ablation_masks_only_cost_feature():
    a,e,c=setup()
    a.network.observe_cost=False
    x=torch.as_tensor(e.reset()); y=x.clone(); y[:,5]=.03
    assert torch.equal(a.network.distribution(x).mean,a.network.distribution(y).mean)
