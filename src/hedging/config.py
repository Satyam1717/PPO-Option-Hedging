"""Typed configuration schema + YAML loader.

Centralizing config here means every numeric constant in the pipeline (vol
levels, cost rates, PPO hyperparameters, ...) lives in one YAML file and one
validated schema, instead of being scattered as magic numbers through the
codebase.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import yaml

from hedging.exceptions import ConfigError


@dataclass(frozen=True)
class MarketConfig:
    S0: float
    K: float
    r: float
    dt: float
    n_steps: int
    prior_vol: float
    vol_window: int

    def __post_init__(self) -> None:
        if self.S0 <= 0:
            raise ConfigError(f"market.S0 must be > 0, got {self.S0}")
        if self.K <= 0:
            raise ConfigError(f"market.K must be > 0, got {self.K}")
        if self.dt <= 0:
            raise ConfigError(f"market.dt must be > 0, got {self.dt}")
        if self.n_steps < 1:
            raise ConfigError(f"market.n_steps must be >= 1, got {self.n_steps}")
        if self.prior_vol <= 0:
            raise ConfigError(f"market.prior_vol must be > 0, got {self.prior_vol}")
        if self.vol_window < 1:
            raise ConfigError(f"market.vol_window must be >= 1, got {self.vol_window}")

    @property
    def maturity(self) -> float:
        """Time to maturity in years, T = n_steps * dt."""
        return self.n_steps * self.dt


@dataclass(frozen=True)
class RegimeConfig:
    name: str
    type: str  # "gbm" or "heston"
    mu: float
    pricing_vol: float
    # GBM-only
    sigma: float | None = None
    # Heston-only
    v0: float | None = None
    kappa: float | None = None
    theta: float | None = None
    xi: float | None = None
    rho: float | None = None
    sigma_after: float | None = None
    switch_fraction: float = 0.5

    def __post_init__(self) -> None:
        if self.type not in ("gbm", "heston", "switching"):
            raise ConfigError(
                f"regimes.{self.name}.type must be 'gbm' or 'heston', got {self.type!r}"
            )
        if self.pricing_vol <= 0:
            raise ConfigError(f"regimes.{self.name}.pricing_vol must be > 0")
        if self.type in ("gbm", "switching"):
            if self.sigma is None or self.sigma <= 0:
                raise ConfigError(f"regimes.{self.name}: gbm regime requires sigma > 0")
        if self.type == "switching" and (
            self.sigma_after is None or self.sigma_after <= 0 or not 0 < self.switch_fraction < 1
        ):
            raise ConfigError("switching requires positive sigma_after and switch_fraction in (0,1)")
        if self.type == "heston":
            missing = [
                p
                for p in ("v0", "kappa", "theta", "xi", "rho")
                if getattr(self, p) is None
            ]
            if missing:
                raise ConfigError(
                    f"regimes.{self.name}: heston regime missing params {missing}"
                )
            if self.v0 <= 0 or self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
                raise ConfigError(
                    f"regimes.{self.name}: heston v0/kappa/theta/xi must be > 0"
                )
            if not (-1.0 <= self.rho <= 1.0):
                raise ConfigError(f"regimes.{self.name}: heston rho must be in [-1, 1]")


@dataclass(frozen=True)
class HedgingConfig:
    transaction_cost_rate: float
    variance_penalty_lambda: float
    static_baseline_vol: float
    training_cost_rates: List[float] = field(default_factory=list)
    vol_prior_strength: float = 5.0
    premium_mc_paths: int = 50000
    premium_seed: int = 91021

    def __post_init__(self) -> None:
        if self.transaction_cost_rate < 0:
            raise ConfigError("hedging.transaction_cost_rate must be >= 0")
        if self.variance_penalty_lambda < 0:
            raise ConfigError("hedging.variance_penalty_lambda must be >= 0")
        if self.static_baseline_vol <= 0:
            raise ConfigError("hedging.static_baseline_vol must be > 0")
        if any(x < 0 for x in self.training_cost_rates) or self.vol_prior_strength < 0:
            raise ConfigError("cost rates and vol_prior_strength must be nonnegative")
        if self.premium_mc_paths < 100:
            raise ConfigError("premium_mc_paths must be >= 100")


@dataclass(frozen=True)
class TrainingConfig:
    total_timesteps: int
    rollout_len: int
    n_envs: int
    n_epochs: int
    n_minibatches: int
    learning_rate: float
    gamma: float
    gae_lambda: float
    clip_range: float
    entropy_coef: float
    vf_coef: float
    max_grad_norm: float
    hidden_sizes: List[int]
    log_std_init: float
    seed: int
    log_every_updates: int
    policy_kind: str = "band"
    observe_cost: bool = True
    target_kl: float = 0.025
    anneal_lr: bool = True
    band_max: float = 0.30
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if self.gamma != 1.0:
            raise ConfigError("gamma must be 1.0 to optimize terminal utility without discounting")
        if self.n_minibatches < 1:
            raise ConfigError("training.n_minibatches must be positive")
        if self.total_timesteps < 1:
            raise ConfigError("training.total_timesteps must be >= 1")
        if self.rollout_len < 1:
            raise ConfigError("training.rollout_len must be >= 1")
        if self.n_envs < 1:
            raise ConfigError("training.n_envs must be >= 1")
        if self.rollout_len % self.n_envs != 0:
            raise ConfigError(
                "training.rollout_len must be divisible by training.n_envs "
                f"(got rollout_len={self.rollout_len}, n_envs={self.n_envs})"
            )
        if self.rollout_len % self.n_minibatches != 0:
            raise ConfigError(
                "training.rollout_len must be divisible by training.n_minibatches "
                f"(got rollout_len={self.rollout_len}, n_minibatches={self.n_minibatches})"
            )
        if not (0.0 < self.clip_range < 1.0):
            raise ConfigError("training.clip_range must be in (0, 1)")
        if self.n_epochs < 1 or self.n_minibatches < 1 or self.learning_rate <= 0:
            raise ConfigError("epochs, minibatches and learning rate must be positive")
        if not 0 <= self.gae_lambda <= 1 or self.target_kl <= 0:
            raise ConfigError("invalid GAE lambda or target KL")
        if self.policy_kind not in ("direct", "band") or not 0 < self.band_max <= 1:
            raise ConfigError("policy_kind must be direct/band and band_max in (0,1]")


@dataclass(frozen=True)
class EvaluationConfig:
    n_eval_episodes: int
    seed: int
    validation_seed: int = 456
    n_validation_episodes: int = 512
    cost_rates: List[float] = field(default_factory=lambda: [0.0005, 0.001, 0.002, 0.003])
    band_candidates: List[float] = field(default_factory=lambda: [0.0, 0.025, 0.05, 0.10, 0.20])
    bootstrap_samples: int = 1000

    def __post_init__(self) -> None:
        if self.n_eval_episodes < 1:
            raise ConfigError("evaluation.n_eval_episodes must be >= 1")
        if self.n_validation_episodes < 1 or self.bootstrap_samples < 10:
            raise ConfigError("validation episodes must be positive and bootstrap_samples >= 10")
        if not self.cost_rates or any(x < 0 for x in self.cost_rates):
            raise ConfigError("evaluation.cost_rates must be nonempty and nonnegative")
        if not self.band_candidates or any(not 0 <= x <= 1 for x in self.band_candidates):
            raise ConfigError("band_candidates must be nonempty and in [0,1]")


@dataclass(frozen=True)
class Config:
    market: MarketConfig
    regimes: Dict[str, RegimeConfig]
    hedging: HedgingConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    test_regimes: Dict[str, RegimeConfig] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.regimes:
            raise ConfigError("config must define at least one regime under 'regimes'")
        if set(self.regimes) & set(self.test_regimes):
            raise ConfigError("training and test regime names must be disjoint")


def _require(d: dict, key: str, section: str) -> object:
    if key not in d:
        raise ConfigError(f"missing required key '{key}' in section '{section}'")
    return d[key]


def read_experiment_config(path: str | Path) -> Config:
    """Load and validate the pipeline config from a YAML file.

    Raises `ConfigError` (never a raw KeyError/TypeError) on any structural
    or semantic problem, so failures are legible instead of a stack trace
    into dict-indexing code.
    """
    path = Path(path)
    if not path.exists():
        raise ConfigError(f"config file not found: {path}")

    try:
        with open(path, "r") as f:
            raw = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"failed to parse YAML in {path}: {e}") from e

    if not isinstance(raw, dict):
        raise ConfigError(f"top-level config in {path} must be a mapping")

    try:
        market_raw = _require(raw, "market", "root")
        market = MarketConfig(**market_raw)

        regimes_raw = _require(raw, "regimes", "root")
        if not isinstance(regimes_raw, dict) or not regimes_raw:
            raise ConfigError("'regimes' must be a non-empty mapping")
        regimes = {
            name: RegimeConfig(name=name, **params) for name, params in regimes_raw.items()
        }

        hedging = HedgingConfig(**_require(raw, "hedging", "root"))
        training = TrainingConfig(**_require(raw, "training", "root"))
        evaluation = EvaluationConfig(**_require(raw, "evaluation", "root"))
        test_regimes = {
            name: RegimeConfig(name=name, **params)
            for name, params in raw.get("test_regimes", {}).items()
        }
    except TypeError as e:
        # dataclass raises TypeError on unexpected/missing kwargs - translate
        # into our own error type so callers only ever see ConfigError.
        raise ConfigError(f"invalid config structure in {path}: {e}") from e

    return Config(
        market=market,
        regimes=regimes,
        hedging=hedging,
        training=training,
        evaluation=evaluation,
        test_regimes=test_regimes,
    )


load_config = read_experiment_config
