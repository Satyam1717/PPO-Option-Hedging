"""Fast paired evaluation over complete path batches and separate band tuning."""
import hashlib
import numpy as np
from hedging.agents.heuristics import DeltaBand
from hedging.env.vector_env import VectorHedgingEnv
from hedging.evaluation.metrics import summarize_hedging
from hedging.market.registry import create_price_model


def derive_scenario_seed(seed, name):
    return int.from_bytes(hashlib.sha256(f"{seed}:{name}".encode()).digest()[:4], "little")


def make_scenario_paths(config, regime, count, seed):
    return create_price_model(regime, config.market).generate_paths(
        count, np.random.default_rng(derive_scenario_seed(seed, regime.name)))


def evaluate_strategy(config, regime, paths, policy, cost_rate):
    env = VectorHedgingEnv(config.market, config.hedging, {regime.name: regime},
                           {regime.name: create_price_model(regime, config.market)}, len(paths), 0,
                           paths=paths, fixed_regime=regime.name, cost_rate=cost_rate, auto_reset=False)
    obs = env.reset()
    policy.reset()
    costs, turnover, previous = np.zeros(len(paths)), np.zeros(len(paths)), np.zeros(len(paths))
    hedges = []
    for _ in range(config.market.n_steps):
        h = np.asarray(policy.act_batch(obs)).reshape(-1)
        obs, _, _, info = env.step(h)
        costs += info["cost"]
        turnover += np.abs(h - previous)
        previous = h.copy()
        hedges.append(h)
    costs += info["final_unwind_cost"]
    turnover += previous
    pnls = info["total_pnl"]
    return {"metrics": summarize_hedging(pnls, costs, turnover, config.hedging.variance_penalty_lambda),
            "total_pnls": pnls, "total_costs": costs, "turnovers": turnover,
            "hedge_ratios": np.asarray(hedges).T, "prices": paths}


def tune_delta_band(config, cost_rate):
    # Only training-domain regimes and the dedicated validation seed are used.
    paths = {name: make_scenario_paths(config, r, config.evaluation.n_validation_episodes,
                            config.evaluation.validation_seed) for name, r in config.regimes.items()}
    scores = {}
    for width in config.evaluation.band_candidates:
        values = [evaluate_strategy(config, r, paths[name], DeltaBand(config.market, width), cost_rate)
                  ["metrics"].mean_utility for name, r in config.regimes.items()]
        scores[width] = float(np.mean(values))
    return max(scores, key=scores.get), scores


scenario_seed = derive_scenario_seed
paths_for = make_scenario_paths
batch_backtest = evaluate_strategy
select_band_width = tune_delta_band
