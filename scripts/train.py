"""Train fresh v2 PPO models; e.g. python scripts/train.py --seeds 42 43 44."""
from __future__ import annotations
import argparse
from dataclasses import asdict, replace
import hashlib
import json
import platform
import time
from pathlib import Path
from _bootstrap import ROOT
import numpy as np
import pandas as pd
import torch
from hedging.agents.ppo_agent import PPOAgent
from hedging.config import read_experiment_config
from hedging.env.accounting import OBS_DIM
from hedging.env.vector_env import VectorHedgingEnv
from hedging.market.registry import create_training_markets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT / "config/config.yaml"))
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument("--policy", choices=["band", "direct"], default="band")
    p.add_argument("--blind-cost", action="store_true")
    p.add_argument("--steps", type=int)
    p.add_argument("--run-name", default="")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-updates", type=int, help="stop at an update boundary, saving exact resume state")
    args = p.parse_args()
    if any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in args.run_name):
        p.error("run-name may contain letters, numbers, underscores and hyphens")
    cfg = read_experiment_config(args.config)
    train_cfg = replace(cfg.training, policy_kind=args.policy, observe_cost=not args.blind_cost)
    if args.steps is not None:
        train_cfg = replace(train_cfg, total_timesteps=args.steps)
    for seed in args.seeds:
        tc = replace(train_cfg, seed=seed)
        name = (args.run_name + "_" if args.run_name else "") + args.policy
        name += "_blind_cost" if args.blind_cost else ""
        name += f"_seed{seed}"
        model_path = ROOT / "models/v2" / f"{name}.pt"
        checkpoint = model_path.with_name("checkpoint_" + model_path.name)
        env = VectorHedgingEnv(cfg.market, cfg.hedging, cfg.regimes, create_training_markets(cfg), tc.n_envs, seed)
        agent = PPOAgent(OBS_DIM, tc.hidden_sizes, seed=seed, policy_kind=tc.policy_kind,
                         observe_cost=tc.observe_cost, market=cfg.market, band_max=tc.band_max)
        start = time.perf_counter()
        history = agent.train(env, tc, checkpoint_path=checkpoint, resume=args.resume,
                              max_updates=args.max_updates)
        elapsed = time.perf_counter() - start
        steps = history[-1]["total_steps"] if history else 0
        completed = steps >= tc.total_timesteps
        # Partial runs never replace a model labelled as trained to the full budget.
        if completed:
            agent.save(model_path)
        output = ROOT / "results/v2/training"
        output.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(history).to_csv(output / f"{name}.csv", index=False)
        sources = sorted((ROOT / "src/hedging").rglob("*.py"))
        source_hash = hashlib.sha256(b"".join(str(x.relative_to(ROOT)).encode()+x.read_bytes()
                                              for x in sources)).hexdigest()
        manifest = {"name": name, "completed": completed, "total_steps": steps,
                    "elapsed_seconds_this_invocation": elapsed, "training": asdict(tc),
                    "market": asdict(cfg.market), "hedging": asdict(cfg.hedging),
                    "training_regimes": {k: asdict(v) for k,v in cfg.regimes.items()},
                    "source_sha256": source_hash, "python": platform.python_version(),
                    "torch": torch.__version__, "numpy": np.__version__,
                    "model": str(model_path.relative_to(ROOT)) if completed else None,
                    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest() if completed else None}
        (output / f"{name}.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"{name}: completed={completed}, {steps} steps, {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()
