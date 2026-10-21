"""Walk-forward out-of-sample evaluation of PPO agents and classical baselines.

Look-ahead safeguards:

* Folds are strictly chronological: ``train_end + embargo <= test_start``.
* Each fold's agent is trained (and model-selected on a validation tail) using only
  periods inside its training window; features are causal so the shared feature
  tensor contains no future information.
* Test episodes start from cash at ``test_start``; decisions at ``t`` see only data
  ``<= t``. The final mark-to-market of a fold uses the close at ``test_end``, which
  is also the first decision period of the next fold, so stitched out-of-sample
  returns are contiguous and non-overlapping.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.baselines import BaselineStrategy, build_baselines
from src.data import MarketData
from src.environment import PortfolioEnv, make_portfolio_env
from src.features import FeatureSet
from src.metrics import METRIC_LABELS, metrics_from_records, summary_table, to_markdown_table
from src.train import first_valid_index, prepare_data, train_agent
from src.utils import configure_logging, get_logger, load_config, parse_overrides, resolve_path

logger = get_logger(__name__)

Range = tuple[int, int]
ActionFn = Callable[[np.ndarray], np.ndarray]


# ---------------------------------------------------------------------------
# Fold construction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Fold:
    """One walk-forward split (half-open period ranges)."""

    index: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    @property
    def train_range(self) -> Range:
        """Training periods ``[train_start, train_end)``."""
        return self.train_start, self.train_end

    def evaluation_range(self, n_periods: int) -> Range:
        """Env range for testing: decisions in ``[test_start, test_end)``, last mark at ``test_end``."""
        return self.test_start, min(self.test_end + 1, n_periods)


def walk_forward_splits(
    n_periods: int,
    first_valid: int,
    train_window: int,
    test_window: int,
    step: int | None = None,
    embargo: int = 0,
    expanding: bool = False,
    n_folds: int | None = None,
    min_train: int = 50,
) -> list[Fold]:
    """Generate chronological walk-forward folds.

    With ``n_folds`` set, the folds are anchored to the *end* of the data (the last
    test window ends at ``n_periods``) and spaced by ``step``; otherwise folds roll
    forward from ``first_valid`` until the data is exhausted.

    Args:
        n_periods: Total periods ``T``.
        first_valid: First period with a complete feature window.
        train_window: Training length (maximum length if ``expanding``).
        test_window: Test length.
        step: Distance between consecutive test starts (defaults to ``test_window``).
        embargo: Gap between the end of training and the start of testing.
        expanding: Use all history from ``first_valid`` (anchored) instead of a rolling window.
        n_folds: Number of end-anchored folds.
        min_train: Minimum acceptable training length.

    Returns:
        List of :class:`Fold`.

    Raises:
        ValueError: If no valid fold fits in the data.
    """
    step = int(step or test_window)
    folds: list[Fold] = []
    if n_folds is not None:
        last_test_start = n_periods - test_window
        starts = [last_test_start - (n_folds - 1 - k) * step for k in range(n_folds)]
    else:
        starts = []
        s = first_valid + train_window + embargo
        while s + 2 <= n_periods:
            starts.append(s)
            s += step
    for test_start in starts:
        train_end = test_start - embargo
        train_start = first_valid if expanding else max(first_valid, train_end - train_window)
        if train_end - train_start < min_train:
            logger.warning("Skipping fold at test_start=%d: training window too short", test_start)
            continue
        test_end = min(test_start + test_window, n_periods)
        folds.append(Fold(len(folds), train_start, train_end, test_start, test_end))
    if not folds:
        raise ValueError(
            f"No walk-forward fold fits: T={n_periods}, first_valid={first_valid}, "
            f"train_window={train_window}, test_window={test_window}"
        )
    return folds


def split_validation(train_range: Range, fraction: float, min_len: int = 30) -> tuple[Range, Range | None]:
    """Carve a validation tail from a training range.

    Args:
        train_range: ``[start, end)``.
        fraction: Fraction reserved for validation (0 disables).
        min_len: Minimum validation length.

    Returns:
        ``(train_range, val_range_or_None)``.
    """
    s, e = train_range
    n_val = int((e - s) * fraction)
    if fraction <= 0 or n_val < min_len or (e - s - n_val) < min_len:
        return train_range, None
    return (s, e - n_val), (e - n_val, e)


# ---------------------------------------------------------------------------
# Episode runners
# ---------------------------------------------------------------------------
def make_eval_env(
    cfg: Mapping[str, Any],
    market: MarketData,
    featureset: FeatureSet,
    period_range: Range,
    reward_overrides: Mapping[str, Any] | None = None,
    env_overrides: Mapping[str, Any] | None = None,
) -> PortfolioEnv:
    """Deterministic full-range evaluation env (no random start, no early bankruptcy stop).

    Args:
        cfg: Full configuration.
        market: Market panel.
        featureset: Features.
        period_range: ``[start, end)``.
        reward_overrides: Reward overrides (only affect the logged reward).
        env_overrides: Variant-specific environment overrides (e.g. trade smoothing).

    Returns:
        The environment.
    """
    return make_portfolio_env(
        featureset.values,
        market.close,
        market.volume,
        cfg,
        start_index=period_range[0],
        end_index=period_range[1],
        overrides={**(env_overrides or {}),
                   "random_start": False, "episode_length": None, "bankruptcy_threshold": 0.0},
        reward_overrides=reward_overrides,
        asset_names=market.assets,
        dates=list(market.dates),
        open_prices=market.open,
    )


def run_episode(
    env: PortfolioEnv,
    action_fn: ActionFn | None = None,
    weights_fn: Callable[[PortfolioEnv], np.ndarray] | None = None,
) -> pd.DataFrame:
    """Run one deterministic episode over the env's full range.

    Exactly one of ``action_fn`` (policy actions) or ``weights_fn`` (target weights,
    e.g. a baseline strategy) must be given.

    Args:
        env: Evaluation environment.
        action_fn: Maps an observation to an action.
        weights_fn: Maps the env to target weights.

    Returns:
        Per-period records indexed by the date at which each return is realised.

    Raises:
        ValueError: If neither or both policies are provided.
    """
    if (action_fn is None) == (weights_fn is None):
        raise ValueError("Provide exactly one of action_fn or weights_fn")
    if isinstance(weights_fn, BaselineStrategy):
        weights_fn.reset()
    obs, _ = env.reset(options={"start_index": env.start_index})
    rows: list[dict[str, Any]] = []
    done = False
    while not done:
        if weights_fn is not None:
            obs, reward, term, trunc, info = env.step_with_weights(weights_fn(env))
        else:
            obs, reward, term, trunc, info = env.step(action_fn(obs))  # type: ignore[misc]
        row = {
            "date": info.get("date", info["t"]),
            "nav": info["nav"],
            "return": info["portfolio_return"],
            "turnover": info["turnover"],
            "cost": info["cost"],
            "cost_usd": info["costs"]["total"],
            "drawdown": info["drawdown"],
            "reward": reward,
            "gross_exposure": info["gross_exposure"],
            "cash_weight": info["cash_weight"],
        }
        row.update({f"w_{name}": float(w) for name, w in zip(env.asset_names, info["weights"])})
        rows.append(row)
        done = term or trunc
    if term:  # type: ignore[possibly-undefined]
        logger.warning("Episode terminated early at t=%d (NAV wiped out)", env.t)
    return pd.DataFrame(rows).set_index("date")


def load_policy(
    model_path: str | Path, env: PortfolioEnv, vecnormalize_path: str | Path | None = None, deterministic: bool = True
) -> ActionFn:
    """Load an SB3 PPO model as an observation -> action function.

    Args:
        model_path: Saved ``.zip`` model.
        env: Env whose spaces match the model (used to restore VecNormalize stats).
        vecnormalize_path: Optional VecNormalize statistics (applied if obs-normalised).
        deterministic: Use the policy mean instead of sampling.

    Returns:
        Action function.
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    model = PPO.load(str(model_path), device="cpu")
    normalize: Callable[[np.ndarray], np.ndarray] | None = None
    if vecnormalize_path is not None and Path(vecnormalize_path).exists():
        vn = VecNormalize.load(str(vecnormalize_path), DummyVecEnv([lambda: env]))
        vn.training = False
        if vn.norm_obs:
            normalize = vn.normalize_obs

    def act(obs: np.ndarray) -> np.ndarray:
        x = normalize(obs) if normalize is not None else obs
        action, _ = model.predict(x, deterministic=deterministic)
        return np.asarray(action)

    return act


# ---------------------------------------------------------------------------
# Walk-forward driver
# ---------------------------------------------------------------------------
@dataclass
class WalkForwardResult:
    """Outputs of :func:`run_walk_forward`.

    Attributes:
        records: ``{strategy: stitched out-of-sample records}`` with an ``equity`` column.
        fold_metrics: Long frame of per-fold metrics (``fold``, ``strategy``, metrics...).
        summary: Strategy x metric table on the stitched out-of-sample returns.
        folds: The folds used.
        model_paths: ``{(variant, fold): model path}``.
    """

    records: dict[str, pd.DataFrame]
    fold_metrics: pd.DataFrame
    summary: pd.DataFrame
    folds: list[Fold]
    model_paths: dict[tuple[str, int], Path] = field(default_factory=dict)


def _stitch(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    df = pd.concat(frames)
    df["equity"] = np.cumprod(1.0 + df["return"].clip(lower=-1.0).fillna(0.0).to_numpy())
    peak = np.maximum.accumulate(df["equity"].to_numpy())
    df["oos_drawdown"] = 1.0 - df["equity"] / peak
    return df


def run_walk_forward(
    cfg: Mapping[str, Any],
    market: MarketData,
    featureset: FeatureSet,
    folds: Sequence[Fold],
    variants: Mapping[str, Mapping[str, Any]],
    baselines: Mapping[str, BaselineStrategy],
    output_dir: str | Path,
    timesteps: int,
    validation_fraction: float = 0.15,
    warm_start: bool = False,
    deterministic: bool = True,
    risk_free_rate: float = 0.0,
) -> WalkForwardResult:
    """Train every RL variant per fold and evaluate all strategies out of sample.

    Args:
        cfg: Full configuration.
        market: Market panel.
        featureset: Features.
        folds: Walk-forward folds.
        variants: ``{name: reward_overrides}`` - one PPO agent per entry per fold.
        baselines: ``{name: strategy}`` classical strategies.
        output_dir: Root for models and per-fold artifacts.
        timesteps: PPO timesteps per variant per fold.
        validation_fraction: Validation tail of each training window.
        warm_start: Initialise each fold from the same variant's previous-fold model.
        deterministic: Deterministic policy at test time.
        risk_free_rate: Annual risk-free rate for metrics.

    Returns:
        The :class:`WalkForwardResult`.
    """
    out = Path(output_dir)
    P = int(cfg["environment"].get("periods_per_year", 252))
    per_strategy: dict[str, list[pd.DataFrame]] = {}
    fold_rows: list[dict[str, Any]] = []
    model_paths: dict[tuple[str, int], Path] = {}
    prev_model: dict[str, Path] = {}

    for fold in folds:
        eval_range = fold.evaluation_range(market.n_periods)
        train_range, val_range = split_validation(fold.train_range, validation_fraction)
        logger.info(
            "Fold %d | train %s..%s | test %s..%s",
            fold.index,
            market.dates[fold.train_start].date(), market.dates[fold.train_end - 1].date(),
            market.dates[eval_range[0]].date(), market.dates[eval_range[1] - 1].date(),
        )
        fold_results: dict[str, pd.DataFrame] = {}

        for name, reward_overrides in variants.items():
            run_dir = out / "models" / _slug(name) / f"fold_{fold.index}"
            try:
                result = train_agent(
                    cfg, market, featureset, train_range, run_dir,
                    val_range=val_range,
                    total_timesteps=timesteps,
                    reward_overrides=reward_overrides,
                    init_model_path=prev_model.get(name) if warm_start else None,
                    run_name=f"{name}_fold{fold.index}",
                    seed=int(cfg.get("seed", 0)) + fold.index,
                )
            except Exception:
                logger.exception("Training failed for %s fold %d; skipping", name, fold.index)
                continue
            model_paths[(name, fold.index)] = result.best_model_path
            prev_model[name] = result.model_path
            env = make_eval_env(cfg, market, featureset, eval_range, reward_overrides)
            policy = load_policy(result.best_model_path, env, result.vecnormalize_path, deterministic)
            fold_results[name] = run_episode(env, action_fn=policy)

        for name, strategy in baselines.items():
            env = make_eval_env(cfg, market, featureset, eval_range)
            fold_results[name] = run_episode(env, weights_fn=strategy)

        for name, rec in fold_results.items():
            rec = rec.assign(fold=fold.index)
            per_strategy.setdefault(name, []).append(rec)
            m = metrics_from_records(rec, P, risk_free_rate)
            fold_rows.append({"fold": fold.index, "strategy": name, **m})
            fold_dir = out / "folds" / f"fold_{fold.index}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            rec.to_csv(fold_dir / f"{_slug(name)}.csv")

    records = {name: _stitch(frames) for name, frames in per_strategy.items()}
    summary = pd.DataFrame(
        {name: metrics_from_records(df, P, risk_free_rate) for name, df in records.items()}
    ).T
    summary.index.name = "strategy"
    return WalkForwardResult(records, pd.DataFrame(fold_rows), summary, list(folds), model_paths)


def _slug(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def plot_equity_curves(
    records: Mapping[str, pd.DataFrame],
    path: str | Path,
    folds: Sequence[Fold] | None = None,
    dates: pd.DatetimeIndex | None = None,
    title: str = "Out-of-sample equity (walk-forward, net of costs)",
) -> Path:
    """Plot stitched out-of-sample NAV (growth of $1) and drawdowns.

    Args:
        records: ``{strategy: stitched records}`` with ``equity`` / ``oos_drawdown``.
        path: Output image path.
        folds: Folds whose test starts are marked with vertical lines.
        dates: Full date index (to locate fold boundaries).
        title: Figure title.

    Returns:
        The saved path.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]}
    )
    cmap = plt.get_cmap("tab10")
    for i, (name, df) in enumerate(records.items()):
        is_rl = not any(k in name for k in ("1/N", "MVO", "Risk Parity"))
        style = {"lw": 2.0 if is_rl else 1.3, "ls": "-" if is_rl else "--", "color": cmap(i % 10)}
        ax1.plot(df.index, df["equity"], label=name, **style)
        ax2.plot(df.index, -100.0 * df["oos_drawdown"], **style)
    if folds is not None and dates is not None:
        for f in folds:
            for ax in (ax1, ax2):
                ax.axvline(dates[f.test_start], color="grey", lw=0.8, alpha=0.5)
    ax1.set_ylabel("NAV (growth of $1)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def walk_forward_backtest(cfg: Mapping[str, Any], output_dir: str | Path | None = None) -> WalkForwardResult:
    """Walk-forward backtest of the configured PPO agent against classical baselines.

    Args:
        cfg: Full configuration (uses the ``backtest`` section).
        output_dir: Output directory (defaults to ``<output_dir>/<experiment>/backtest``).

    Returns:
        The walk-forward result (also written to disk).
    """
    market, featureset = prepare_data(cfg)
    bcfg = cfg["backtest"]
    folds = walk_forward_splits(
        market.n_periods,
        first_valid_index(featureset, cfg),
        int(bcfg["train_window"]),
        int(bcfg["test_window"]),
        step=int(bcfg.get("step", bcfg["test_window"])),
        embargo=int(bcfg.get("embargo", 0)),
        expanding=bool(bcfg.get("expanding", False)),
    )
    out = Path(output_dir) if output_dir else resolve_path(cfg.get("output_dir", "runs")) / cfg.get(
        "experiment_name", "ppo") / "backtest"
    baseline_cfg = dict(cfg.get("experiments", {}).get("baselines", {}))
    baseline_cfg["rebalance_every"] = bcfg.get("benchmark_rebalance_every", baseline_cfg.get("rebalance_every", 21))
    result = run_walk_forward(
        cfg, market, featureset, folds,
        variants={"PPO (risk-constrained)": {}},
        baselines=build_baselines(baseline_cfg, float(cfg["environment"].get("max_gross_leverage", 1.0))),
        output_dir=out,
        timesteps=int(bcfg["timesteps_per_fold"]),
        validation_fraction=float(bcfg.get("validation_fraction", 0.0)),
        warm_start=bool(bcfg.get("warm_start", False)),
        deterministic=bool(bcfg.get("deterministic", True)),
        risk_free_rate=float(bcfg.get("risk_free_rate", 0.0)),
    )
    save_walk_forward(result, out, market.dates, bool(bcfg.get("save_plots", True)))
    return result


def save_walk_forward(
    result: WalkForwardResult, out: Path, dates: pd.DatetimeIndex, save_plots: bool = True,
    summary_name: str = "summary.csv",
) -> None:
    """Persist summary tables, stitched returns and plots; print a Markdown summary.

    Args:
        result: Walk-forward result.
        out: Output directory.
        dates: Full date index.
        save_plots: Whether to write the equity plot.
        summary_name: File name of the summary CSV.
    """
    out.mkdir(parents=True, exist_ok=True)
    result.summary.to_csv(out / summary_name)
    result.fold_metrics.to_csv(out / "fold_metrics.csv", index=False)
    pd.DataFrame({k: v["return"] for k, v in result.records.items()}).to_csv(out / "oos_returns.csv")
    pd.DataFrame({k: v["equity"] for k, v in result.records.items()}).to_csv(out / "oos_equity.csv")
    if save_plots and result.records:
        plot_equity_curves(result.records, out / "equity_curves.png", result.folds, dates)
    table = summary_table(result.summary.to_dict(orient="index"))
    print("\n" + to_markdown_table(table, column_formats={METRIC_LABELS["total_cost_usd"]: ",.0f"}) + "\n")


def main(argv: Sequence[str] | None = None) -> None:
    """CLI entry point for the walk-forward backtest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--set", nargs="*", default=[])
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    walk_forward_backtest(load_config(args.config, parse_overrides(args.set)), args.output_dir)


if __name__ == "__main__":
    main()
