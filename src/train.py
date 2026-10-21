"""PPO training with risk-metric logging (TensorBoard / W&B) and checkpointing.

Usage:
    python -m src.train --config config/hyperparameters.yaml --set ppo.total_timesteps=100000
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv, VecMonitor, VecNormalize

from src.data import MarketData, load_market_data
from src.environment import make_portfolio_env
from src.features import FeatureSet, compute_features
from src.models import build_policy_kwargs
from src.utils import (
    configure_logging,
    get_logger,
    load_config,
    parse_overrides,
    resolve_path,
    save_config,
    set_global_seed,
)

logger = get_logger(__name__)

Range = tuple[int, int]


# ---------------------------------------------------------------------------
# Data splits
# ---------------------------------------------------------------------------
def first_valid_index(featureset: FeatureSet, cfg: Mapping[str, Any]) -> int:
    """First period at which a full feature window is available.

    Args:
        featureset: Computed features.
        cfg: Full configuration.

    Returns:
        ``max(warmup + lookback - 1, lookback - 1)``.
    """
    lookback = int(cfg["environment"]["lookback"])
    return featureset.warmup + lookback - 1


def split_indices(
    n_periods: int, first_valid: int, train_fraction: float, val_fraction: float, embargo: int = 0
) -> dict[str, Range]:
    """Chronological train / validation / test split with embargo gaps.

    Ranges are half-open ``[start, end)`` period indices.

    Args:
        n_periods: Total number of periods ``T``.
        first_valid: First usable period.
        train_fraction: Fraction of usable periods for training.
        val_fraction: Fraction for validation.
        embargo: Periods skipped between consecutive splits.

    Returns:
        ``{"train": (s, e), "val": (s, e), "test": (s, e)}``.

    Raises:
        ValueError: If a split would be too short.
    """
    usable = n_periods - first_valid
    n_train = int(usable * train_fraction)
    n_val = int(usable * val_fraction)
    train = (first_valid, first_valid + n_train)
    val = (train[1] + embargo, train[1] + embargo + n_val)
    test = (val[1] + embargo, n_periods)
    for name, (s, e) in {"train": train, "val": val, "test": test}.items():
        if e - s < 2:
            raise ValueError(f"{name} split [{s}, {e}) is too short; adjust fractions or data length")
    return {"train": train, "val": val, "test": test}


# ---------------------------------------------------------------------------
# Environments
# ---------------------------------------------------------------------------
def make_env_factory(
    market: MarketData,
    featureset: FeatureSet,
    cfg: Mapping[str, Any],
    period_range: Range,
    seed: int,
    env_overrides: Mapping[str, Any] | None = None,
    reward_overrides: Mapping[str, Any] | None = None,
) -> Callable[[], gym.Env]:
    """Return a picklable thunk that builds one :class:`PortfolioEnv`.

    Args:
        market: Market panel.
        featureset: Features aligned with ``market``.
        cfg: Full configuration.
        period_range: ``[start, end)`` period range.
        seed: Seed for the env's RNG.
        env_overrides: Overrides for the ``environment`` section.
        reward_overrides: Overrides for the ``reward`` section.

    Returns:
        Environment constructor.
    """

    def _init() -> gym.Env:
        env = make_portfolio_env(
            featureset.values,
            market.close,
            market.volume,
            cfg,
            start_index=period_range[0],
            end_index=period_range[1],
            overrides=env_overrides,
            reward_overrides=reward_overrides,
            asset_names=market.assets,
            dates=list(market.dates),
        )
        env.reset(seed=seed)
        return env

    return _init


def build_vec_env(
    market: MarketData,
    featureset: FeatureSet,
    cfg: Mapping[str, Any],
    period_range: Range,
    n_envs: int,
    seed: int,
    training: bool,
    reward_overrides: Mapping[str, Any] | None = None,
    env_overrides: Mapping[str, Any] | None = None,
) -> VecEnv:
    """Build a (possibly normalised) vectorised environment.

    Training envs sample random episode starts; evaluation envs run the whole range
    deterministically from its start.

    Args:
        market: Market panel.
        featureset: Features.
        cfg: Full configuration.
        period_range: ``[start, end)`` period range.
        n_envs: Number of parallel environments.
        seed: Base seed (env ``i`` uses ``seed + i``).
        training: Training (random starts, stats update) vs evaluation mode.
        reward_overrides: Overrides for the ``reward`` section.
        env_overrides: Variant-specific overrides for the ``environment`` section.

    Returns:
        The vectorised environment.
    """
    env_overrides = {**(env_overrides or {}),
                     **({} if training else {"random_start": False, "episode_length": None})}
    fns = [
        make_env_factory(market, featureset, cfg, period_range, seed + i, env_overrides, reward_overrides)
        for i in range(n_envs)
    ]
    ppo_cfg = cfg["ppo"]
    use_subproc = ppo_cfg.get("vec_env", "dummy") == "subproc" and n_envs > 1
    venv: VecEnv = SubprocVecEnv(fns) if use_subproc else DummyVecEnv(fns)
    venv = VecMonitor(venv)
    norm_obs = bool(ppo_cfg.get("normalize_obs", False))
    norm_rew = bool(ppo_cfg.get("normalize_reward", False))
    if norm_obs or norm_rew:
        venv = VecNormalize(
            venv,
            training=training,
            norm_obs=norm_obs,
            norm_reward=norm_rew and training,
            gamma=float(ppo_cfg.get("gamma", 0.99)),
            clip_obs=10.0,
        )
    return venv


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
class RiskMetricsCallback(BaseCallback):
    """Log portfolio risk / trading diagnostics from env ``info`` dicts.

    Per rollout it records realised portfolio volatility, mean / max rolling
    drawdown, CVaR, penalty magnitudes, turnover and cost; per finished episode it
    records the env's ``episode_stats`` (total return, Sharpe, max drawdown, ...).
    """

    _STEP_KEYS = (
        "portfolio_return",
        "drawdown",
        "turnover",
        "cost",
        "cvar",
        "cvar_penalty",
        "drawdown_penalty",
        "gross_exposure",
        "cash_weight",
    )

    def __init__(self, periods_per_year: int = 252, verbose: int = 0) -> None:
        """Initialise the callback.

        Args:
            periods_per_year: Annualisation factor for volatility.
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        self.periods_per_year = periods_per_year
        self._buf: dict[str, list[float]] = defaultdict(list)

    def _on_rollout_start(self) -> None:
        self._buf = defaultdict(list)

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "portfolio_return" not in info:
                continue
            for key in self._STEP_KEYS:
                val = info.get(key)
                if val is not None and np.isfinite(val):
                    self._buf[key].append(float(val))
            stats = info.get("episode_stats")
            if stats:
                for key, val in stats.items():
                    self.logger.record_mean(f"episode/{key}", float(val))
        return True

    def _on_rollout_end(self) -> None:
        b = self._buf
        if not b["portfolio_return"]:
            return
        r = np.asarray(b["portfolio_return"])
        dd = np.asarray(b["drawdown"])
        self.logger.record("risk/portfolio_vol_ann", float(r.std() * np.sqrt(self.periods_per_year)))
        self.logger.record("risk/mean_return_ann", float(r.mean() * self.periods_per_year))
        self.logger.record("risk/rolling_drawdown_mean", float(dd.mean()))
        self.logger.record("risk/rolling_drawdown_max", float(dd.max()))
        self.logger.record("risk/cvar_mean", float(np.mean(b["cvar"])))
        self.logger.record("reward/cvar_penalty_mean", float(np.mean(b["cvar_penalty"])))
        self.logger.record("reward/drawdown_penalty_mean", float(np.mean(b["drawdown_penalty"])))
        self.logger.record("trading/turnover_mean", float(np.mean(b["turnover"])))
        self.logger.record("trading/cost_bps_mean", float(np.mean(b["cost"]) * 1e4))
        self.logger.record("portfolio/gross_exposure", float(np.mean(b["gross_exposure"])))
        self.logger.record("portfolio/cash_weight", float(np.mean(b["cash_weight"])))


class LagrangianCallback(BaseCallback):
    """RCPO-style adaptive penalty multipliers (Tessler et al., 2019).

    After every rollout the average constraint costs observed in the env ``info``
    dicts (rolling CVaR and squared drawdown) are compared with their targets and
    the multipliers are updated by projected gradient ascent on the Lagrangian,
    on a slower timescale than the policy::

        lambda_k <- clip(lambda_k + lr * (cost_k - target_k) / target_k, 0, lambda_max)

    The new multipliers are pushed into every environment's reward function.
    """

    def __init__(
        self,
        cvar_target: float,
        drawdown_target: float,
        lr: float = 0.2,
        lambda_max: float = 5.0,
        init: tuple[float, float] = (0.0, 0.0),
        state_path: str | Path | None = None,
        verbose: int = 0,
    ) -> None:
        """Initialise the callback.

        Args:
            cvar_target: Target for the average daily CVaR (as a positive loss, e.g. 0.015).
            drawdown_target: Target for the average squared drawdown ``D_t^2``.
            lr: Relative step size of the multiplier update.
            lambda_max: Upper bound on each multiplier.
            init: Initial ``(lambda_cvar, lambda_drawdown)``.
            state_path: JSON file where the final multipliers are stored (for warm starts).
            verbose: Verbosity level.
        """
        super().__init__(verbose)
        if cvar_target <= 0 or drawdown_target <= 0:
            raise ValueError("constraint targets must be positive")
        self.targets = (float(cvar_target), float(drawdown_target))
        self.lr, self.lambda_max = float(lr), float(lambda_max)
        self.lambdas = [float(init[0]), float(init[1])]
        self.state_path = Path(state_path) if state_path else None
        self._costs: list[list[float]] = [[], []]

    def _push(self) -> None:
        self.training_env.env_method("set_reward_lambdas", self.lambdas[0], self.lambdas[1])

    def _on_training_start(self) -> None:
        self._push()

    def _on_rollout_start(self) -> None:
        self._costs = [[], []]

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "cvar" in info and "drawdown" in info:
                self._costs[0].append(float(info["cvar"]))
                self._costs[1].append(float(info["drawdown"]) ** 2)
        return True

    def update(self, mean_costs: Sequence[float | None]) -> list[float]:
        """Apply one projected multiplier step given the rollout's average constraint costs.

        Args:
            mean_costs: ``(avg CVaR, avg squared drawdown)``; ``None`` skips that constraint.

        Returns:
            The updated ``[lambda_cvar, lambda_drawdown]``.
        """
        for k, c in enumerate(mean_costs):
            if c is None:
                continue
            step = self.lr * (float(c) - self.targets[k]) / self.targets[k]
            self.lambdas[k] = float(np.clip(self.lambdas[k] + step, 0.0, self.lambda_max))
        return list(self.lambdas)

    def _on_rollout_end(self) -> None:
        costs = [float(np.mean(c)) if c else None for c in self._costs]
        self.update(costs)
        for name, c in zip(("cvar", "dd2"), costs):
            if c is not None:
                self.logger.record(f"lagrangian/cost_{name}", c)
        self.logger.record("lagrangian/lambda_cvar", self.lambdas[0])
        self.logger.record("lagrangian/lambda_drawdown", self.lambdas[1])
        self._push()

    def _on_training_end(self) -> None:
        if self.state_path is not None:
            self.state_path.write_text(json.dumps({"lambda_cvar": self.lambdas[0],
                                                   "lambda_drawdown": self.lambdas[1]}))


def load_lagrangian_state(path: str | Path | None) -> tuple[float, float] | None:
    """Read multipliers saved by :class:`LagrangianCallback` (``None`` if absent)."""
    if path is None or not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    return float(d["lambda_cvar"]), float(d["lambda_drawdown"])


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def linear_schedule(initial: float) -> Callable[[float], float]:
    """Linearly decay a value from ``initial`` to 0 over training.

    Args:
        initial: Starting value.

    Returns:
        Schedule mapping SB3's ``progress_remaining`` (1 -> 0) to a value.
    """

    def schedule(progress_remaining: float) -> float:
        return progress_remaining * initial

    return schedule


def _ppo_kwargs(cfg: Mapping[str, Any]) -> dict[str, Any]:
    p = cfg["ppo"]
    lr = float(p.get("learning_rate", 3e-4))
    return {
        "learning_rate": linear_schedule(lr) if p.get("lr_schedule", "constant") == "linear" else lr,
        "n_steps": int(p.get("n_steps", 512)),
        "batch_size": int(p.get("batch_size", 256)),
        "n_epochs": int(p.get("n_epochs", 10)),
        "gamma": float(p.get("gamma", 0.99)),
        "gae_lambda": float(p.get("gae_lambda", 0.95)),
        "clip_range": float(p.get("clip_range", 0.2)),
        "ent_coef": float(p.get("ent_coef", 0.0)),
        "vf_coef": float(p.get("vf_coef", 0.5)),
        "max_grad_norm": float(p.get("max_grad_norm", 0.5)),
        "target_kl": p.get("target_kl"),
        "device": p.get("device", "auto"),
    }


def build_ppo(cfg: Mapping[str, Any], env: VecEnv, layout: Mapping[str, int], seed: int) -> PPO:
    """Construct a fresh PPO model with the custom feature extractor.

    Args:
        cfg: Full configuration.
        env: Training vectorised environment.
        layout: Observation layout from the environment.
        seed: Random seed.

    Returns:
        The PPO model.
    """
    return PPO(
        "MlpPolicy",
        env,
        policy_kwargs=build_policy_kwargs(cfg["model"], layout),
        seed=seed,
        verbose=0,
        **_ppo_kwargs(cfg),
    )


def _logger_formats(backend: str, stdout: bool = True) -> list[str]:
    formats = ["stdout", "csv"] if stdout else ["csv"]
    if backend in ("tensorboard", "wandb", "both"):
        if importlib.util.find_spec("tensorboard") is None:
            logger.warning("tensorboard not installed; TensorBoard logging disabled")
        else:
            formats.append("tensorboard")
    return formats


def _maybe_init_wandb(cfg: Mapping[str, Any], run_dir: Path, run_name: str) -> Any:
    tcfg = cfg.get("training", {})
    if tcfg.get("logger", "tensorboard") not in ("wandb", "both"):
        return None
    try:
        import wandb
    except ImportError:
        logger.warning("wandb not installed; continuing with TensorBoard only")
        return None
    wcfg = tcfg.get("wandb", {})
    return wandb.init(
        project=wcfg.get("project", "risk-constrained-ppo"),
        entity=wcfg.get("entity"),
        mode=wcfg.get("mode", "online"),
        tags=wcfg.get("tags") or None,
        name=run_name,
        dir=str(run_dir),
        config=dict(cfg),
        sync_tensorboard=True,
        reinit=True,
    )


@dataclass(frozen=True)
class TrainResult:
    """Paths produced by :func:`train_agent`.

    Attributes:
        run_dir: Run directory.
        model_path: Final model checkpoint.
        best_model_path: Best validation model (falls back to the final model).
        vecnormalize_path: VecNormalize statistics, if normalisation is enabled.
    """

    run_dir: Path
    model_path: Path
    best_model_path: Path
    vecnormalize_path: Path | None


def train_agent(
    cfg: Mapping[str, Any],
    market: MarketData,
    featureset: FeatureSet,
    train_range: Range,
    run_dir: str | Path,
    val_range: Range | None = None,
    total_timesteps: int | None = None,
    reward_overrides: Mapping[str, Any] | None = None,
    init_model_path: str | Path | None = None,
    run_name: str | None = None,
    seed: int | None = None,
    log_stdout: bool = True,
    lagrangian: Mapping[str, Any] | None = None,
    save_checkpoints: bool = True,
    env_overrides: Mapping[str, Any] | None = None,
) -> TrainResult:
    """Train a PPO agent on ``train_range`` with optional validation-based selection.

    Args:
        cfg: Full configuration.
        market: Market panel.
        featureset: Features.
        train_range: ``[start, end)`` training periods.
        run_dir: Output directory (logs, checkpoints, models).
        val_range: Optional validation periods for :class:`EvalCallback`.
        total_timesteps: Overrides ``ppo.total_timesteps``.
        reward_overrides: Overrides for the ``reward`` section (e.g. ablation lambdas).
        init_model_path: Warm-start from an existing model.
        run_name: Name used for W&B.
        seed: Overrides the global seed.
        log_stdout: Print SB3 training tables (disable in parallel workers).
        lagrangian: RCPO settings ``{cvar_target, drawdown_target, lr, lambda_max}``; the
            multipliers start from ``init_model_path``'s saved state when warm-starting.
        save_checkpoints: Write periodic checkpoints.
        env_overrides: Variant-specific ``environment`` overrides (e.g. trade smoothing).

    Returns:
        :class:`TrainResult` with artifact paths.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.get("seed", 0) if seed is None else seed)
    set_global_seed(seed)
    ppo_cfg, tcfg = cfg["ppo"], cfg.get("training", {})
    n_envs = int(ppo_cfg.get("n_envs", 1))
    steps = int(total_timesteps if total_timesteps is not None else ppo_cfg["total_timesteps"])
    run_name = run_name or run_dir.name
    save_config(cfg, run_dir / "config.yaml")

    train_env = build_vec_env(market, featureset, cfg, train_range, n_envs, seed, True, reward_overrides, env_overrides)
    layout = train_env.get_attr("observation_layout", indices=[0])[0]

    if init_model_path is not None and Path(init_model_path).exists():
        logger.info("Warm-starting from %s", init_model_path)
        overrides = {k: v for k, v in _ppo_kwargs(cfg).items() if k != "device"}
        model = PPO.load(
            str(init_model_path), env=train_env, device=ppo_cfg.get("device", "auto"),
            custom_objects=overrides,
        )
        model.set_random_seed(seed)
    else:
        model = build_ppo(cfg, train_env, layout, seed)

    wandb_run = _maybe_init_wandb(cfg, run_dir, run_name)
    formats = _logger_formats(tcfg.get("logger", "tensorboard"), log_stdout)
    model.set_logger(configure(str(run_dir / "logs"), formats))

    callbacks: list[BaseCallback] = [RiskMetricsCallback(int(cfg["environment"].get("periods_per_year", 252)))]
    if save_checkpoints:
        callbacks.append(CheckpointCallback(
            save_freq=max(int(tcfg.get("checkpoint_freq", 50_000)) // n_envs, 1),
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo",
            save_vecnormalize=True,
        ))
    if lagrangian:
        prev = load_lagrangian_state(Path(init_model_path).parent / "lagrangian.json") if init_model_path else None
        callbacks.append(LagrangianCallback(
            cvar_target=float(lagrangian["cvar_target"]),
            drawdown_target=float(lagrangian["drawdown_target"]),
            lr=float(lagrangian.get("lr", 0.2)),
            lambda_max=float(lagrangian.get("lambda_max", 5.0)),
            init=prev or (float(lagrangian.get("init_cvar", 0.0)), float(lagrangian.get("init_drawdown", 0.0))),
            state_path=run_dir / "lagrangian.json",
        ))
    eval_env = None
    if val_range is not None:
        eval_env = build_vec_env(market, featureset, cfg, val_range, 1, seed + 10_000, False, reward_overrides,
                                 env_overrides)
        callbacks.append(
            EvalCallback(
                eval_env,
                n_eval_episodes=int(tcfg.get("n_eval_episodes", 1)),
                eval_freq=max(int(tcfg.get("eval_freq", 20_000)) // n_envs, 1),
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval"),
                deterministic=True,
                verbose=0,
            )
        )

    try:
        model.learn(
            total_timesteps=steps,
            callback=CallbackList(callbacks),
            log_interval=int(tcfg.get("log_interval", 1)),
            reset_num_timesteps=True,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    model_path = run_dir / "final_model.zip"
    model.save(str(model_path))
    vec_path: Path | None = None
    if isinstance(train_env, VecNormalize):
        vec_path = run_dir / "vecnormalize.pkl"
        train_env.save(str(vec_path))
    best = run_dir / "best_model" / "best_model.zip"
    train_env.close()
    if eval_env is not None:
        eval_env.close()
    logger.info("Saved model to %s", model_path)
    return TrainResult(run_dir, model_path, best if best.exists() else model_path, vec_path)


def prepare_data(cfg: Mapping[str, Any]) -> tuple[MarketData, FeatureSet]:
    """Load market data and compute features according to the config.

    Args:
        cfg: Full configuration.

    Returns:
        ``(market, featureset)``.
    """
    market = load_market_data(cfg, seed=int(cfg.get("seed", 0)))
    featureset = compute_features(market, cfg)
    logger.info(
        "Data: %d periods x %d assets, %d features/asset, warm-up %d",
        market.n_periods, market.n_assets, featureset.n_features, featureset.warmup,
    )
    return market, featureset


def run_training(cfg: Mapping[str, Any]) -> TrainResult:
    """Train on the configured train split, selecting on the validation split.

    Args:
        cfg: Full configuration.

    Returns:
        The training result.
    """
    market, featureset = prepare_data(cfg)
    tcfg = cfg["training"]
    splits = split_indices(
        market.n_periods,
        first_valid_index(featureset, cfg),
        float(tcfg["train_fraction"]),
        float(tcfg["val_fraction"]),
        int(tcfg.get("embargo", 0)),
    )
    logger.info("Splits: %s", splits)
    run_dir = resolve_path(cfg.get("output_dir", "runs")) / cfg.get("experiment_name", "ppo")
    return train_agent(cfg, market, featureset, splits["train"], run_dir, val_range=splits["val"])


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for training."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="Path to hyperparameters.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="Overrides: section.key=value")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config, parse_overrides(args.set))
    run_training(cfg)


if __name__ == "__main__":
    main()
