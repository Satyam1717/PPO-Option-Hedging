"""Frozen-policy evaluation, validation-tuned heuristics, paired intervals and plots."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from _bootstrap import ROOT
import numpy as np
import pandas as pd
from hedging.agents.delta_hedge import EstimatedVolatilityDelta, FixedVolatilityDelta
from hedging.agents.heuristics import ConstantHedge, DeltaBand, PeriodicDelta
from hedging.agents.ppo_agent import PPOAgent
from hedging.config import read_experiment_config
from hedging.evaluation.batch_backtest import evaluate_strategy, make_scenario_paths, tune_delta_band
from hedging.evaluation.metrics import compare_paired_utilities
from hedging.pricing.premium import quote_initial_option
from hedging.viz.research_plots import research_plots


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    p.add_argument("--models", nargs="+")
    p.add_argument("--episodes", type=int)
    p.add_argument("--output-dir", default=str(ROOT / "results/v2/evaluation"))
    args = p.parse_args()
    cfg = read_experiment_config(args.config)
    if args.episodes is not None:
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, n_eval_episodes=args.episodes))
    model_paths = ([Path(x) for x in args.models] if args.models else
                   sorted(x for x in (ROOT / "models/v2").glob("*.pt")
                          if x.name.startswith(("band_seed", "band_blind_cost_seed", "direct_seed"))))
    if not model_paths:
        p.error("no trained v2 agents; run scripts/train.py first")
    learned = {"ppo_"+path.stem: PPOAgent.load(path) for path in model_paths}
    for model in learned.values():
        if model.market != cfg.market:
            p.error("evaluation market geometry/rate must match the trained model")
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    all_regimes = cfg.regimes | cfg.test_regimes
    rows, comparisons, arrays, examples, tuning = [], [], {}, {}, {}
    selected = {}
    for cost in cfg.evaluation.cost_rates:
        width, scores = tune_delta_band(cfg, cost)
        selected[cost] = width
        tuning[str(cost)] = {"chosen_width": width, "validation_scores": scores}
    for name, regime in all_regimes.items():
        paths = make_scenario_paths(cfg, regime, cfg.evaluation.n_eval_episodes, cfg.evaluation.seed)
        quote, quote_vol, quote_se = quote_initial_option(cfg.market,
            (cfg.hedging.premium_mc_paths, cfg.hedging.premium_seed), regime)
        for cost in cfg.evaluation.cost_rates:
            policies = {"no_hedge": ConstantHedge(0), "constant_half": ConstantHedge(0.5),
                        "static_bs": FixedVolatilityDelta(cfg.market, cfg.hedging.static_baseline_vol),
                        "quote_bs": FixedVolatilityDelta(cfg.market, quote_vol),
                        "adaptive_bs": EstimatedVolatilityDelta(cfg.market),
                        "periodic_delta": PeriodicDelta(cfg.market, 5),
                        "delta_band": DeltaBand(cfg.market, selected[cost]), **learned}
            results = {}
            for method, policy in policies.items():
                result = evaluate_strategy(cfg, regime, paths, policy, cost)
                results[method] = result
                agent = learned.get(method)
                group = ("ppo_"+agent.policy_kind+("_blind_cost" if not agent.observe_cost else "")
                         if agent else method)
                rows.append({"regime": name, "domain": "training" if name in cfg.regimes else "held_out",
                             "cost_bps": cost*10000, "cost_seen_in_training": cost in cfg.hedging.training_cost_rates,
                             "method": method, "group": group, "training_seed": agent.seed if agent else None,
                             "premium": quote, "premium_mc_se": quote_se, **result["metrics"].as_dict()})
                key = f"{name}__{cost*10000:g}bps__{method}"
                arrays[key+"__pnl"] = result["total_pnls"]
                arrays[key+"__cost"] = result["total_costs"]
                arrays[key+"__turnover"] = result["turnovers"]
            for method in learned:
                diff, low, high = compare_paired_utilities(results[method]["total_pnls"],
                    results["delta_band"]["total_pnls"], cfg.hedging.variance_penalty_lambda,
                    cfg.evaluation.bootstrap_samples)
                comparisons.append({"regime": name, "cost_bps": cost*10000, "method": method,
                                    "reference": "delta_band", "utility_difference": diff,
                                    "ci_low": low, "ci_high": high,
                                    "interval_scope": "paired_paths_for_one_trained_agent"})
            if cost == cfg.evaluation.cost_rates[-1]:
                main_method = next((k for k in learned if k.startswith("ppo_band_seed")), next(iter(learned)))
                show = ["adaptive_bs", "delta_band", main_method]
                examples[name] = {m: {"prices": paths[0], "hedge_ratios": results[m]["hedge_ratios"][0]}
                                  for m in show}
            print(f"evaluated {name}, cost={cost*10000:g}bps, {len(paths)} paired paths", flush=True)
    metrics = pd.DataFrame(rows)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(comparisons).to_csv(out / "paired_intervals.csv", index=False)
    np.savez_compressed(out / "episode_results.npz", **arrays)
    aggregate = metrics.groupby(["regime", "domain", "cost_bps", "group"])[
        ["mean_utility", "std_pnl", "rmse", "cvar_5", "mean_total_cost", "mean_turnover"]].agg(["mean", "std", "count"])
    aggregate.to_csv(out / "across_seed_summary.csv")
    manifest = {"evaluation": asdict(cfg.evaluation), "hedging": asdict(cfg.hedging),
                "market": asdict(cfg.market), "training_regimes": list(cfg.regimes),
                "test_regimes": {k: asdict(v) for k,v in cfg.test_regimes.items()},
                "models": {str(x): hashlib.sha256(x.read_bytes()).hexdigest() for x in model_paths},
                "band_tuning": tuning,
                "interval_note": "Path bootstrap is conditional on each trained agent; across-seed SD is separate."}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    research_plots(metrics, pd.DataFrame(comparisons), examples, out / "figures")
    lines = ["# Corrected-accounting evaluation", "", "All agents are frozen; all methods share paths per scenario.",
             "Bands are selected on independent validation paths from training-domain regimes.",
             "Path bootstrap intervals exclude training-seed uncertainty. See across_seed_summary.csv.", "",
             "| group | mean utility across scenarios | mean RMSE | mean cost |", "|---|---:|---:|---:|"]
    for group, g in metrics.groupby("group"):
        lines.append(f"| {group} | {g.mean_utility.mean():.4f} | {g.rmse.mean():.4f} | {g.mean_total_cost.mean():.4f} |")
    lines.extend(["", "Scenario averages are descriptive, with equal weight per regime and cost setting.",
                  "Inspect metrics.csv for losses and failures hidden by an aggregate."])
    (out / "SUMMARY.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
