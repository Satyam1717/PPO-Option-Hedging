"""Check financed BS replication: shrinking risk and no persistent pricing bias."""
from _bootstrap import ROOT
from dataclasses import replace
import numpy as np
import pandas as pd
from hedging.config import read_experiment_config
from hedging.env.accounting import execute_rebalance
from hedging.market.gbm import GBMSimulator
from hedging.pricing.black_scholes import call_value, call_hedge_ratio


def main():
    cfg=read_experiment_config(ROOT/'config/config.yaml')
    m=cfg.market; maturity=m.maturity; rows=[]
    for n in [3,7,21,63,126,252]:
        dt=maturity/n
        paths=GBMSimulator(m.S0,m.r,.2,dt,n).generate_paths(20000,np.random.default_rng(2024+n))
        cash=np.full(len(paths),float(call_value(m.S0,m.K,maturity,m.r,.2)))
        previous=np.zeros(len(paths))
        for t in range(n):
            target=call_hedge_ratio(paths[:,t],m.K,(n-t)*dt,m.r,.2)
            cash,_=execute_rebalance(cash,previous,target,paths[:,t],m.r,dt,0.)
            previous=target
        pnl=cash+previous*paths[:,-1]-np.maximum(paths[:,-1]-m.K,0)
        mean,sd,se=pnl.mean(),pnl.std(),pnl.std()/np.sqrt(len(pnl))
        rows.append({'n_steps':n,'mean_pnl':mean,'std_pnl':sd,'mean_standard_error':se})
        print(f'n={n}: mean={mean:+.5f}, std={sd:.5f}, mean SE={se:.5f}')
    exponent=float(np.polyfit(np.log([x['n_steps'] for x in rows]),np.log([x['std_pnl'] for x in rows]),1)[0])
    assert -.65<exponent<-.35, f'unexpected convergence rate {exponent}'
    assert abs(rows[-1]['mean_pnl'])<5*rows[-1]['mean_standard_error'], 'persistent replication bias'
    output=ROOT/'results/v2'; output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'sanity_check.csv',index=False)
    print(f'PASS: financed hedge converges with exponent {exponent:.4f}, terminal mean compatible with zero.')


if __name__=='__main__': main()
