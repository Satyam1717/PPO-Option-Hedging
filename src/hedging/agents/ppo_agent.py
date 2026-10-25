"""Clipped PPO with Beta controls, GAE, KL stopping and exact training resume."""
from __future__ import annotations
import math
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from hedging.agents.networks import HedgeActorCritic, hedge_from_controls
from hedging.agents.experience import collect_experience, estimate_advantages, prepare_training_batch
from hedging.agents.optimization import optimize_rollout
from hedging.config import MarketConfig
from hedging.env.accounting import OBS_DIM
from hedging.exceptions import ModelNotTrainedError
from hedging.logging_setup import get_logger

logger = get_logger(__name__)
FORMAT_VERSION = 2


def compute_gae(rewards, values, dones, last_values, gamma=1.0, gae_lambda=0.95):
    """Compatibility entry point; new code uses estimate_advantages."""
    return estimate_advantages(rewards,values,dones,last_values,gamma,gae_lambda)


class PPOAgent:
    def __init__(self, obs_dim=OBS_DIM, hidden_sizes=(64, 64), log_std_init=None, seed=42,
                 policy_kind="band", observe_cost=True, market=None, band_max=0.30):
        if obs_dim != OBS_DIM:
            raise ValueError("v2 uses seven observations; legacy five-input weights are incompatible")
        torch.set_num_threads(1)
        torch.manual_seed(seed)
        self.obs_dim, self.hidden_sizes, self.seed = obs_dim, list(hidden_sizes), seed
        self.policy_kind, self.observe_cost, self.band_max = policy_kind, observe_cost, band_max
        self.market = market or MarketConfig(100., 100., 0.02, 1 / 252, 21, 0.2, 5)
        self.network = HedgeActorCritic(obs_dim, hidden_sizes, 2 if policy_kind == "band" else 1, observe_cost)
        self.rng = np.random.default_rng(seed)
        self._trained = False

    def act_batch(self, obs, deterministic=True):
        with torch.no_grad():
            dist = self.network.distribution(torch.as_tensor(obs, dtype=torch.float32))
            controls = dist.mean if deterministic else dist.sample()
        return hedge_from_controls(controls.numpy(), obs, self.policy_kind, self.market, self.band_max)

    def act(self, obs, deterministic=True):
        return self.act_batch(np.asarray(obs)[None], deterministic).reshape(1)

    def reset(self):
        pass

    def _payload(self):
        return {"format_version": FORMAT_VERSION, "state_dict": self.network.state_dict(),
                "obs_dim": self.obs_dim, "hidden_sizes": self.hidden_sizes, "seed": self.seed,
                "policy_kind": self.policy_kind, "observe_cost": self.observe_cost,
                "market": asdict(self.market), "band_max": self.band_max}

    @staticmethod
    def _write(payload, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, tmp)
        tmp.replace(path)

    def save(self, path):
        self._write(self._payload(), path)

    @classmethod
    def load(cls, path):
        if not Path(path).exists():
            raise ModelNotTrainedError(f"no v2 model at {path}; run scripts/train.py")
        if Path(path).suffix != ".pt":
            raise ValueError("legacy .pkl agents use incompatible accounting and observations")
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload.get("format_version") != FORMAT_VERSION:
            raise ValueError("unsupported model format")
        model = cls(payload["obs_dim"], payload["hidden_sizes"], seed=payload["seed"],
                    policy_kind=payload["policy_kind"], observe_cost=payload["observe_cost"],
                    market=MarketConfig(**payload["market"]), band_max=payload["band_max"])
        model.network.load_state_dict(payload["state_dict"])
        model._trained = True
        return model

    def train(self, vec_env, cfg, log_fn=None, checkpoint_path=None, checkpoint_every=20,
              resume=False, max_updates=None):
        torch.set_num_threads(cfg.torch_threads)
        if vec_env.n_envs != cfg.n_envs or self.policy_kind != cfg.policy_kind or self.observe_cost != cfg.observe_cost:
            raise ValueError("agent/environment settings do not match training config")
        optimizer = torch.optim.Adam(self.network.parameters(), lr=cfg.learning_rate, eps=1e-5)
        history, start, total_steps = [], 0, 0
        obs = vec_env.reset()
        # Trusted local training checkpoints contain NumPy RNG/environment objects.
        if resume:
            if checkpoint_path is None or not Path(checkpoint_path).exists():
                raise FileNotFoundError("--resume requires an existing v2 checkpoint")
            ck = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            expected = asdict(cfg)
            for key, value in ck["training_config"].items():
                if key not in ("total_timesteps", "log_every_updates") and expected[key] != value:
                    raise ValueError(f"resume changed training setting: {key}")
            if ck["market"] != asdict(vec_env.market) or ck["hedging"] != asdict(vec_env.hedging):
                raise ValueError("resume changed market/accounting settings")
            if ck["regimes"] != {k: asdict(v) for k, v in vec_env.regimes.items()}:
                raise ValueError("resume changed market regimes")
            self.network.load_state_dict(ck["model"]["state_dict"])
            optimizer.load_state_dict(ck["optimizer"])
            torch.set_rng_state(ck["torch_rng"])
            self.rng.bit_generator.state = ck["numpy_rng"]
            vec_env.load_state_dict(ck["env"])
            obs, history, start, total_steps = ck["obs"], ck["history"], ck["update"], ck["total_steps"]
        steps_per_rollout = cfg.rollout_len // cfg.n_envs
        total_updates = math.ceil(cfg.total_timesteps / cfg.rollout_len)
        stop = min(total_updates, start + max_updates) if max_updates is not None else total_updates
        for update in range(start + 1, stop + 1):
            rate = cfg.learning_rate * (1 - (update - 1) / total_updates) if cfg.anneal_lr else cfg.learning_rate
            for group in optimizer.param_groups:
                group["lr"] = rate
            experience, obs = collect_experience(self,vec_env,steps_per_rollout,obs)
            training_batch = prepare_training_batch(experience,cfg)
            statistics, early_stop = optimize_rollout(self.network,optimizer,training_batch,cfg,self.rng)
            pnls = experience.completed_pnls
            total_steps += cfg.rollout_len
            row = dict(update=update, total_steps=total_steps, learning_rate=rate,
                       mean_recent_total_pnl=float(np.mean(pnls)) if pnls else float("nan"),
                       std_recent_total_pnl=float(np.std(pnls)) if pnls else float("nan"),
                       mean_utility=float(np.mean(np.asarray(pnls)-vec_env.hedging.variance_penalty_lambda*np.square(pnls))) if pnls else float("nan"),
                       early_stopped=early_stop,
                       **statistics)
            history.append(row)
            if update == 1 or update % cfg.log_every_updates == 0 or update == stop:
                logger.info(f"{self.policy_kind} seed={self.seed} update={update}/{total_updates} "
                            f"steps={total_steps} utility={row['mean_utility']:.4f} "
                            f"std={row['std_recent_total_pnl']:.4f}")
                if log_fn:
                    log_fn(row)
            if checkpoint_path and (update % checkpoint_every == 0 or update == stop):
                self._write({"model": self._payload(), "optimizer": optimizer.state_dict(),
                             "torch_rng": torch.get_rng_state(), "numpy_rng": self.rng.bit_generator.state,
                             "env": vec_env.state_dict(), "obs": obs, "history": history,
                             "update": update, "total_steps": total_steps, "training_config": asdict(cfg),
                             "market": asdict(vec_env.market), "hedging": asdict(vec_env.hedging),
                             "regimes": {k: asdict(v) for k,v in vec_env.regimes.items()}}, checkpoint_path)
        self._trained = True
        return history
