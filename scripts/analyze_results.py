"""Produce training/behavior diagnostics and an honest measured-results note."""
from _bootstrap import ROOT
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hedging.agents.ppo_agent import PPOAgent
from hedging.pricing.black_scholes import call_value
from hedging.viz.research_plots import results_summary_plots


def main():
    output=ROOT/'results/v2/evaluation'; figures=output/'figures'
    figures.mkdir(parents=True,exist_ok=True)
    metrics=pd.read_csv(output/'metrics.csv')
    intervals=pd.read_csv(output/'paired_intervals.csv')
    results_summary_plots(metrics, figures)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for kind in ['band','direct','band_blind_cost']:
        tables=[pd.read_csv(p) for p in sorted((ROOT/'results/v2/training').glob(kind+'_seed*.csv'))]
        if not tables: continue
        common=min(map(len,tables))
        curves=np.array([t.mean_utility.iloc[:common].rolling(10,min_periods=1).mean() for t in tables])
        x=tables[0].total_steps.iloc[:common]
        mean=curves.mean(0); sd=curves.std(0,ddof=1) if len(tables)>1 else np.zeros(common)
        axes[0].plot(x,mean,label=kind); axes[0].fill_between(x,mean-sd,mean+sd,alpha=.12)
        curves=np.array([t.std_recent_total_pnl.iloc[:common].rolling(10,min_periods=1).mean() for t in tables])
        axes[1].plot(x,curves.mean(0),label=kind)
    axes[0].set(xlabel='Training transitions',ylabel='Mean episode utility',title='Training utility (10-update smoothing)')
    axes[1].set(xlabel='Training transitions',ylabel='Terminal P&L standard deviation',title='Training hedge error')
    for ax in axes: ax.legend(fontsize=8); ax.ticklabel_format(axis='x',style='sci',scilimits=(0,0))
    fig.tight_layout(); fig.savefig(figures/'training_curves.png',dpi=160); plt.close(fig)
    models=[PPOAgent.load(p) for p in sorted((ROOT/'models/v2').glob('band_seed*.pt'))]
    if models:
        market=models[0].market; costs=np.array([0,5,10,15,20,30,40])
        premium=float(call_value(market.S0,market.K,market.maturity,market.r,.25))
        fig,ax=plt.subplots(figsize=(7,4))
        for previous in [.2,.5,.8]:
            obs=np.tile([0.,.5,previous,.25,premium/market.S0-previous,0.,premium/market.S0],(len(costs),1)).astype(np.float32)
            obs[:,5]=costs/10000
            actions=np.array([a.act_batch(obs) for a in models])
            mean,sd=actions.mean(0),actions.std(0,ddof=1)
            ax.plot(costs,mean,'o-',label=f'Previous holding {previous:g}')
            ax.fill_between(costs,mean-sd,mean+sd,alpha=.15)
        ax.set(xlabel='Transaction cost (bps)',ylabel='Chosen stock position',
               title='Cost response at fixed synthetic state (mean +/- seed SD)')
        ax.legend(); fig.tight_layout(); fig.savefig(figures/'cost_response.png',dpi=160); plt.close(fig)
    # Average training seeds within each setting, then weight settings equally.
    groups=metrics.groupby(['regime','domain','cost_bps','group'],as_index=False).mean(numeric_only=True)
    main=groups[groups.group=='ppo_band']
    baseline=groups[groups.group=='delta_band']
    paired=main.merge(baseline,on=['regime','domain','cost_bps'],suffixes=('_ppo','_band'))
    wins=int((paired.mean_utility_ppo>paired.mean_utility_band).sum())
    masked=groups[groups.group=='ppo_band_blind_cost']
    ablation=main.merge(masked,on=['regime','domain','cost_bps'],suffixes=('_aware','_masked'))
    cost_wins=int((ablation.mean_utility_aware>ablation.mean_utility_masked).sum())
    ci=intervals[intervals.method.str.match(r'ppo_band_seed\d+$')]
    lines=['# Measured v2 findings','',
           'These results use corrected financing. They are not directly comparable with legacy results.',
           'Policies are frozen on the final test set; no hyperparameters were selected using that set.','',
           f'- Band PPO has higher seed-averaged utility than the tuned delta band in {wins}/{len(paired)} settings.',
           f'- Explicit cost input improves seed-averaged utility over masking that input in {cost_wins}/{len(ablation)} settings.',
           f'- Individual-agent paired path intervals are positive in {int((ci.ci_low>0).sum())}/{len(ci)} comparisons.',
           '- Those intervals exclude training randomness; three training seeds are reported separately. No multiple-comparison correction is applied.',
           '- Cost-feature ablation masks the explicit cost input. Cash and observed outcomes can still indirectly reveal costs.','',
           '## Graphs', '',
           'Training seeds are averaged within each setting, then all 28 settings receive equal weight.', '',
           '![Strategy comparison: utility, hedging error and transaction costs](results/v2/evaluation/figures/results_comparison.png)', '',
           '![Utility differences by market and transaction cost](results/v2/evaluation/figures/results_by_setting.png)', '',
           'Export figures: [comparison PDF](results/v2/evaluation/figures/results_comparison.pdf) and '
           '[settings PDF](results/v2/evaluation/figures/results_by_setting.pdf).', '',
           'Regenerate these graphs from saved metrics with `python scripts/plot_results.py`.', '',
           '## Average results', '',
           '| group | average utility | average RMSE | average cost |', '|---|---:|---:|---:|']
    for group,g in groups.groupby('group'):
        lines.append(f'| {group} | {g.mean_utility.mean():.4f} | {g.rmse.mean():.4f} | {g.mean_total_cost.mean():.4f} |')
    lines+=['','## Failures and successes by setting','',
            '| regime | cost bps | PPO utility | tuned-band utility | difference |','|---|---:|---:|---:|---:|']
    for r in paired.itertuples():
        lines.append(f'| {r.regime} | {r.cost_bps:g} | {r.mean_utility_ppo:.4f} | {r.mean_utility_band:.4f} | {r.mean_utility_ppo-r.mean_utility_band:+.4f} |')
    lines+=['','## Interpretation limits','',
            'The band architecture includes an observable BS-delta prior; direct PPO does not. Differences are not attributable solely to optimizer quality.',
            'The test scenarios are synthetic. The seven-feature observation is not a complete Markov state for latent stochastic volatility.',
            'Heston premium estimates have Monte Carlo uncertainty. Projected Euler simulation has discretization bias.',
            'A switch in future volatility affects the inception quote, available equally to all policies. The policy is not told the switch time.',
            'The cash/volatility state response plot is a fixed-state diagnostic, not an out-of-sample trading result.',
            'No real-market profitability or universal superiority is claimed.']
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(f'Wrote RESULTS.md and diagnostic figures; band wins {wins}/{len(paired)} settings.')


if __name__=='__main__': main()
