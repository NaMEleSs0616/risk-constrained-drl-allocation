#!/usr/bin/env python
"""Ablation study + classical baselines under rolling walk-forward cross-validation.

For each of ``experiments.n_folds`` end-anchored rolling folds and each training seed
in ``experiments.seeds``, this script:

1. trains one PPO agent per reward variant (A: vanilla, B: CVaR only, C: drawdown
   only, D: CVaR + drawdown, plus supplementary variants) with identical
   hyper-parameters except ``lambda_1`` / ``lambda_2``, selecting the best checkpoint
   on a validation tail of the training window. Jobs run in a process pool;
2. evaluates every agent and the classical baselines (1/N rebalanced, 1/N
   buy-and-hold, Ledoit-Wolf MVO, ERC risk parity, HRP) on the fold's out-of-sample
   test window through the *same* friction-aware environment;
3. stitches the test windows into one out-of-sample track per (strategy, seed) and
   aggregates metrics across seeds (mean +/- std).

Artifacts (under ``experiments.output_dir``, default ``outputs/``):

* ``experiment_results_summary.csv`` - core metrics, mean across seeds (+ std columns)
* ``experiment_results_full.csv``    - all metrics, mean across seeds
* ``seed_metrics.csv``               - every metric for every (strategy, seed)
* ``fold_metrics.csv``               - per-fold metrics for every (strategy, seed)
* ``regime_metrics.csv``             - calm vs stress regime statistics (synthetic data)
* ``significance.csv``               - block-bootstrap tests of Sharpe differences
* ``behaviour.csv``                  - exposure / cash / concentration / turnover
* ``oos_returns.csv`` / ``oos_equity.csv`` / ``oos_records.pkl``
* ``equity_curves.png``              - NAV and drawdown comparison (seed bands)
* ``experiment_results_summary.md``  - Markdown report printed to the console
* ``models/<variant>/seed_<s>/fold_<k>/`` - checkpoints, best/final weights, TensorBoard logs

Usage:
    python scripts/run_experiments.py
    python scripts/run_experiments.py --timesteps 300000 --n-folds 3 --seeds 0 1 2 --workers 4
    python scripts/run_experiments.py --variants A_vanilla D_full --resume
    python scripts/run_experiments.py --smoke          # minutes-scale end-to-end check
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import pickle
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.backtest import (  # noqa: E402
    Fold,
    load_policy,
    make_eval_env,
    run_episode,
    split_validation,
    walk_forward_splits,
)
from src.baselines import build_baselines  # noqa: E402
from src.metrics import (  # noqa: E402
    METRIC_LABELS,
    conditional_metrics,
    metrics_from_records,
    sharpe_difference_bootstrap,
    summary_table,
    tail_statistics,
    to_markdown_table,
)
from src.train import first_valid_index, prepare_data, train_agent  # noqa: E402
from src.utils import (  # noqa: E402
    SMOKE_OVERRIDES,
    configure_logging,
    deep_update,
    get_logger,
    load_config,
    parse_overrides,
    resolve_path,
    save_config,
)

logger = get_logger("run_experiments")

#: Columns of the headline comparison table (raw metric keys).
CORE_METRICS: list[str] = [
    "cumulative_return_pct",
    "sharpe",
    "sortino",
    "max_drawdown_pct",
    "calmar",
    "avg_daily_turnover",
    "total_cost_pct",
    "total_cost_usd",
]
EW_NAME = "Equal Weight (1/N)"


def slugify(name: str) -> str:
    """Filesystem-safe lower-case identifier."""
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Variant:
    """One reward-ablation variant."""

    key: str
    label: str
    reward_overrides: dict[str, float]
    lagrangian: dict[str, float] | None = None
    vol_target_overlay: float | None = None
    env_overrides: dict[str, float] | None = None


@dataclass(frozen=True)
class Job:
    """Train-and-evaluate unit: one variant on one fold with one seed."""

    variant: Variant
    fold: Fold
    seed: int
    run_dir: str
    timesteps: int


def build_variants(ecfg: Mapping[str, Any], selected: Sequence[str] | None) -> list[Variant]:
    """Resolve the requested ablation variants from the ``experiments`` section.

    Args:
        ecfg: The ``experiments`` config section.
        selected: Variant keys to run (``None`` = all).

    Returns:
        Variants in config order.

    Raises:
        ValueError: If an unknown variant is requested.
    """
    all_variants: Mapping[str, Mapping[str, Any]] = ecfg.get("variants", {})
    keys = list(selected) if selected else list(all_variants)
    unknown = set(keys) - set(all_variants)
    if unknown:
        raise ValueError(f"Unknown variants {sorted(unknown)}; available: {list(all_variants)}")
    out = []
    for k in keys:
        spec = all_variants[k]
        overrides = {"lambda_cvar": float(spec.get("lambda_cvar", 0.0)),
                     "lambda_drawdown": float(spec.get("lambda_drawdown", 0.0))}
        if "lambda_turnover" in spec:
            overrides["lambda_turnover"] = float(spec["lambda_turnover"])
        rcpo = spec.get("rcpo")
        out.append(Variant(
            key=k,
            label=str(spec.get("label", k)),
            reward_overrides=overrides,
            lagrangian={kk: float(vv) for kk, vv in rcpo.items()} if rcpo else None,
            vol_target_overlay=float(spec["vol_target_overlay"]) if spec.get("vol_target_overlay") else None,
            env_overrides={kk: float(vv) for kk, vv in spec["env"].items()} if spec.get("env") else None,
        ))
    return out


def _init_worker(threads: int) -> None:
    import torch

    torch.set_num_threads(max(1, threads))
    configure_logging(logging.WARNING)


def run_job(
    cfg: Mapping[str, Any], job: Job, validation_fraction: float, deterministic: bool, resume: bool
) -> dict[str, Any]:
    """Train (or reuse) one agent and evaluate it on its fold's test window.

    Runs inside a worker process; never raises (errors are returned as text).

    Args:
        cfg: Full configuration.
        job: The job.
        validation_fraction: Validation tail of the training window for model selection.
        deterministic: Deterministic policy at test time.
        resume: Reuse an existing trained model in ``job.run_dir`` if present.

    Returns:
        Dict with the job, out-of-sample records (or ``None``), timing and error text.
    """
    t0 = time.perf_counter()
    try:
        market, featureset = prepare_data(cfg)
        run_dir = Path(job.run_dir)
        final = run_dir / "final_model.zip"
        if resume and final.exists():
            best = run_dir / "best_model" / "best_model.zip"
            model_path = best if best.exists() else final
            vec = run_dir / "vecnormalize.pkl"
            vec_path: Path | None = vec if vec.exists() else None
        else:
            train_range, val_range = split_validation(job.fold.train_range, validation_fraction)
            result = train_agent(
                cfg, market, featureset, train_range, run_dir,
                val_range=val_range,
                total_timesteps=job.timesteps,
                reward_overrides=job.variant.reward_overrides,
                run_name=f"{job.variant.key}_s{job.seed}_f{job.fold.index}",
                seed=1000 * job.seed + job.fold.index,
                log_stdout=False,
                lagrangian=job.variant.lagrangian,
                env_overrides=job.variant.env_overrides,
            )
            model_path, vec_path = result.best_model_path, result.vecnormalize_path
        env = make_eval_env(
            cfg, market, featureset, job.fold.evaluation_range(market.n_periods), job.variant.reward_overrides,
            job.variant.env_overrides,
        )
        records = run_episode(env, action_fn=load_policy(model_path, env, vec_path, deterministic))
        records = records.assign(fold=job.fold.index, seed=job.seed)
        return {"job": job, "records": records, "model_path": str(model_path),
                "seconds": time.perf_counter() - t0, "error": None}
    except Exception:  # noqa: BLE001 - report, never kill the pool
        return {"job": job, "records": None, "model_path": None,
                "seconds": time.perf_counter() - t0, "error": traceback.format_exc()}


def execute_jobs(
    cfg: Mapping[str, Any], jobs: Sequence[Job], n_workers: int, threads: int,
    validation_fraction: float, deterministic: bool, resume: bool, progress_path: Path,
) -> list[dict[str, Any]]:
    """Run jobs sequentially or in a spawn-based process pool, logging progress / ETA.

    Args:
        cfg: Full configuration.
        jobs: Jobs to run.
        n_workers: Number of worker processes (``<= 1`` runs in-process).
        threads: Torch threads per worker.
        validation_fraction: Validation tail fraction.
        deterministic: Deterministic evaluation policy.
        resume: Reuse existing trained models.
        progress_path: File receiving one line per finished job.

    Returns:
        Job results in completion order.
    """
    results: list[dict[str, Any]] = []
    t0 = time.perf_counter()

    def report(res: dict[str, Any]) -> None:
        results.append(res)
        job: Job = res["job"]
        done = len(results)
        elapsed = time.perf_counter() - t0
        eta = elapsed / done * (len(jobs) - done)
        status = "ok" if res["error"] is None else "FAILED"
        line = (f"[{done}/{len(jobs)}] {status} {job.variant.key} seed={job.seed} fold={job.fold.index} "
                f"({res['seconds'] / 60:.1f} min) | elapsed {elapsed / 60:.1f} min, ETA {eta / 60:.1f} min")
        logger.info(line)
        with progress_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")
        if res["error"]:
            logger.error("Job failed:\n%s", res["error"])

    if n_workers <= 1:
        _init_worker(threads)
        configure_logging(logging.INFO)
        for job in jobs:
            report(run_job(cfg, job, validation_fraction, deterministic, resume))
        return results

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx, initializer=_init_worker,
                             initargs=(threads,)) as pool:
        futures = [pool.submit(run_job, cfg, job, validation_fraction, deterministic, resume) for job in jobs]
        for fut in as_completed(futures):
            report(fut.result())
    return results


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------
def stitch(frames: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate fold records chronologically and add ``equity`` / ``oos_drawdown``."""
    df = pd.concat(sorted(frames, key=lambda f: f.index[0])).sort_index()
    df["equity"] = np.cumprod(1.0 + df["return"].clip(lower=-1.0).fillna(0.0).to_numpy())
    df["oos_drawdown"] = 1.0 - df["equity"] / np.maximum.accumulate(df["equity"].to_numpy())
    return df


def behaviour_stats(df: pd.DataFrame) -> dict[str, float]:
    """Allocation behaviour: exposure, cash, concentration and weight dynamics."""
    w = df[[c for c in df.columns if c.startswith("w_")]].to_numpy()
    gross = np.abs(w).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        eff_n = np.where(gross > 1e-9, gross**2 / np.maximum((w**2).sum(axis=1), 1e-18), np.nan)
    return {
        "avg_gross_exposure": float(np.mean(gross)),
        "avg_cash_weight": float(df["cash_weight"].mean()),
        "min_gross_exposure": float(np.min(gross)),
        "avg_effective_n": float(np.nanmean(eff_n)) if np.isfinite(eff_n).any() else float("nan"),
        "max_single_weight": float(np.max(np.abs(w))) if w.size else float("nan"),
        "avg_daily_turnover": float(df["turnover"].mean()),
    }


@dataclass
class ExperimentResult:
    """All aggregated experiment outputs."""

    stitched: dict[str, dict[int, pd.DataFrame]]
    seed_metrics: pd.DataFrame
    summary_mean: pd.DataFrame
    summary_std: pd.DataFrame
    fold_metrics: pd.DataFrame
    regime_metrics: pd.DataFrame
    significance: pd.DataFrame
    behaviour: pd.DataFrame
    folds: list[Fold]
    rl_labels: list[str]
    failures: list[str] = field(default_factory=list)


def aggregate(
    rl_frames: Mapping[str, Mapping[int, list[pd.DataFrame]]],
    baseline_frames: Mapping[str, list[pd.DataFrame]],
    folds: Sequence[Fold],
    regimes: pd.Series | None,
    periods_per_year: int,
    risk_free_rate: float,
    reference: str | None,
) -> ExperimentResult:
    """Compute every table from per-fold out-of-sample records.

    Args:
        rl_frames: ``{label: {seed: [fold records]}}``.
        baseline_frames: ``{name: [fold records]}`` (deterministic; stored as seed 0).
        folds: Folds used.
        regimes: Regime label indexed by date (``None`` for real data).
        periods_per_year: Annualisation factor.
        risk_free_rate: Annual risk-free rate.
        reference: RL label used as the reference for Sharpe-difference tests.

    Returns:
        The :class:`ExperimentResult`.
    """
    P = periods_per_year
    all_frames: dict[str, dict[int, list[pd.DataFrame]]] = {k: dict(v) for k, v in rl_frames.items()}
    for name, frames in baseline_frames.items():
        all_frames[name] = {0: list(frames)}
    rl_labels = list(rl_frames)

    stitched: dict[str, dict[int, pd.DataFrame]] = {}
    seed_rows, fold_rows, regime_rows, beh_rows = [], [], [], []
    for name, by_seed in all_frames.items():
        kind = "PPO" if name in rl_frames else "baseline"
        stitched[name] = {}
        for seed, frames in by_seed.items():
            if not frames:
                continue
            df = stitch(frames)
            stitched[name][seed] = df
            row = {"strategy": name, "kind": kind, "seed": seed, "n_folds": len(frames)}
            row.update(metrics_from_records(df, P, risk_free_rate))
            row.update(tail_statistics(df["return"].to_numpy()))
            seed_rows.append(row)
            beh_rows.append({"strategy": name, "seed": seed, **behaviour_stats(df)})
            for fr in frames:
                fold_rows.append({"strategy": name, "kind": kind, "seed": seed, "fold": int(fr["fold"].iloc[0]),
                                  **metrics_from_records(fr, P, risk_free_rate)})
            if regimes is not None:
                reg = regimes.reindex(df.index).fillna(0).to_numpy().astype(int)
                for label, mask in (("calm", reg == 0), ("stress", reg == 1), ("all", np.ones_like(reg, bool))):
                    regime_rows.append({"strategy": name, "seed": seed, "regime": label,
                                        **conditional_metrics(df["return"].to_numpy(), mask, P)})

    seed_metrics = pd.DataFrame(seed_rows)
    order = list(all_frames)
    numeric = seed_metrics.drop(columns=["kind", "seed"])
    summary_mean = numeric.groupby("strategy", sort=False).mean(numeric_only=True).reindex(order)
    summary_std = numeric.groupby("strategy", sort=False).std(ddof=1, numeric_only=True).reindex(order)
    summary_mean.index.name = summary_std.index.name = "strategy"

    regime_df = pd.DataFrame(regime_rows)
    if not regime_df.empty:
        regime_df = (regime_df.drop(columns="seed").groupby(["strategy", "regime"], sort=False)
                     .mean(numeric_only=True).reset_index())
    behaviour = (pd.DataFrame(beh_rows).drop(columns="seed").groupby("strategy", sort=False)
                 .mean(numeric_only=True).reindex(order))

    # Seed-averaged daily return per strategy (= equal-weight ensemble of seeds).
    mean_returns = pd.DataFrame(
        {name: pd.concat([d["return"] for d in by_seed.values()], axis=1).mean(axis=1)
         for name, by_seed in stitched.items() if by_seed}
    ).dropna()
    sig_rows = []
    comparisons: list[tuple[str, str]] = []
    if reference in mean_returns:
        comparisons += [(lab, reference) for lab in rl_labels if lab != reference and lab in mean_returns]
    if EW_NAME in mean_returns:
        comparisons += [(name, EW_NAME) for name in mean_returns if name != EW_NAME]
    for a, b in comparisons:
        res = sharpe_difference_bootstrap(mean_returns[a].to_numpy(), mean_returns[b].to_numpy(), P,
                                          block=21, n_boot=2000, seed=7)
        sig_rows.append({"strategy": a, "versus": b, **res})
    return ExperimentResult(
        stitched=stitched,
        seed_metrics=seed_metrics,
        summary_mean=summary_mean,
        summary_std=summary_std,
        fold_metrics=pd.DataFrame(fold_rows),
        regime_metrics=regime_df,
        significance=pd.DataFrame(sig_rows),
        behaviour=behaviour,
        folds=list(folds),
        rl_labels=rl_labels,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _pm(mean: float, std: float, fmt: str) -> str:
    if not np.isfinite(mean):
        return "n/a"
    if not np.isfinite(std):
        return format(mean, fmt)
    return f"{format(mean, fmt)} ± {format(std, fmt)}"


def mean_std_table(res: ExperimentResult, metrics: Sequence[str], formats: Mapping[str, str]) -> pd.DataFrame:
    """Strategy x metric table of ``mean ± std`` strings (std across seeds)."""
    data = {}
    for m in metrics:
        fmt = formats.get(m, ".3f")
        data[METRIC_LABELS.get(m, m)] = [
            _pm(res.summary_mean.at[s, m], res.summary_std.at[s, m] if s in res.rl_labels else float("nan"), fmt)
            for s in res.summary_mean.index
        ]
    df = pd.DataFrame(data, index=res.summary_mean.index)
    df.index.name = "Strategy"
    return df


def plot_equity_bands(res: ExperimentResult, path: Path, dates: pd.DatetimeIndex, title: str) -> None:
    """Seed-mean NAV with min-max seed bands for PPO variants; lines for baselines."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    cmap = plt.get_cmap("tab10")
    for i, (name, by_seed) in enumerate(res.stitched.items()):
        if not by_seed:
            continue
        eq = pd.concat([d["equity"] for d in by_seed.values()], axis=1)
        dd = pd.concat([d["oos_drawdown"] for d in by_seed.values()], axis=1)
        is_rl = name in res.rl_labels
        color = cmap(i % 10)
        ax1.plot(eq.index, eq.mean(axis=1), color=color, lw=2.0 if is_rl else 1.3,
                 ls="-" if is_rl else "--", label=name)
        ax2.plot(dd.index, -100 * dd.mean(axis=1), color=color, lw=1.6 if is_rl else 1.1, ls="-" if is_rl else "--")
        if is_rl and eq.shape[1] > 1:
            ax1.fill_between(eq.index, eq.min(axis=1), eq.max(axis=1), color=color, alpha=0.12, lw=0)
    for f in res.folds:
        for ax in (ax1, ax2):
            ax.axvline(dates[f.test_start], color="grey", lw=0.8, alpha=0.5)
    ax1.set_ylabel("NAV (growth of $1)")
    ax1.set_title(title)
    ax1.legend(loc="upper left", fontsize=8, ncol=2, frameon=False)
    ax1.grid(alpha=0.3)
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def write_artifacts(res: ExperimentResult, out: Path, dates: pd.DatetimeIndex, regimes: pd.Series | None) -> str:
    """Persist every table / plot and return the Markdown report.

    Args:
        res: Aggregated results.
        out: Output directory.
        dates: Full date index.
        regimes: Regime labels by date (stored for downstream analysis).

    Returns:
        The Markdown report (also written to ``experiment_results_summary.md``).
    """
    out.mkdir(parents=True, exist_ok=True)
    core = summary_table(res.summary_mean.to_dict(orient="index"), CORE_METRICS)
    std = summary_table(res.summary_std.to_dict(orient="index"), CORE_METRICS).add_suffix(" (std)")
    pd.concat([core, std], axis=1).to_csv(out / "experiment_results_summary.csv", float_format="%.6f")
    res.summary_mean.to_csv(out / "experiment_results_full.csv", float_format="%.6f")
    res.summary_std.to_csv(out / "experiment_results_full_std.csv", float_format="%.6f")
    res.seed_metrics.to_csv(out / "seed_metrics.csv", index=False, float_format="%.6f")
    res.fold_metrics.to_csv(out / "fold_metrics.csv", index=False, float_format="%.6f")
    res.regime_metrics.to_csv(out / "regime_metrics.csv", index=False, float_format="%.6f")
    res.significance.to_csv(out / "significance.csv", index=False, float_format="%.6f")
    res.behaviour.to_csv(out / "behaviour.csv", float_format="%.6f")
    rets = {f"{n}|seed{s}": d["return"] for n, bs in res.stitched.items() for s, d in bs.items()}
    eqs = {f"{n}|seed{s}": d["equity"] for n, bs in res.stitched.items() for s, d in bs.items()}
    pd.DataFrame(rets).to_csv(out / "oos_returns.csv")
    pd.DataFrame(eqs).to_csv(out / "oos_equity.csv")
    with (out / "oos_records.pkl").open("wb") as fh:
        pickle.dump({"stitched": res.stitched, "folds": res.folds, "rl_labels": res.rl_labels,
                     "dates": dates, "regimes": regimes}, fh)
    plot_equity_bands(res, out / "equity_curves.png", dates,
                      f"Out-of-sample NAV - {len(res.folds)}-fold rolling walk-forward, net of frictions "
                      f"(PPO: seed mean, band = min/max)")

    formats = {"cumulative_return_pct": ".2f", "max_drawdown_pct": ".2f", "avg_daily_turnover": ".4f",
               "total_cost_pct": ".3f", "total_cost_usd": ",.0f", "sharpe": ".3f", "sortino": ".3f", "calmar": ".3f"}
    md_core = to_markdown_table(mean_std_table(res, CORE_METRICS, formats))

    fm = res.fold_metrics
    if not fm.empty:
        pivot = fm.groupby(["strategy", "fold"], sort=False)["sharpe"].mean().unstack("fold").sort_index(axis=1)
        pivot = pivot.reindex(res.summary_mean.index)
        pivot.columns = [f"Fold {c}" for c in pivot.columns]
        pivot.index.name = "Strategy"
        md_folds = to_markdown_table(pivot, floatfmt=".3f")
    else:
        md_folds = "(no folds)"

    md_regime = "(no regime labels)"
    if not res.regime_metrics.empty:
        rm = res.regime_metrics[res.regime_metrics["regime"] != "all"].pivot_table(
            index="strategy", columns="regime", values=["ann_mean_pct", "cvar_95_pct", "worst_day_pct"], sort=False)
        rm.columns = [f"{a} [{b}]" for a, b in rm.columns]
        rm = rm.reindex(res.summary_mean.index)
        rm.index.name = "Strategy"
        md_regime = to_markdown_table(rm, floatfmt=".2f")

    md_sig = to_markdown_table(res.significance.set_index("strategy"), floatfmt=".3f") \
        if not res.significance.empty else "(none)"
    md_beh = to_markdown_table(res.behaviour, floatfmt=".3f")
    fold_lines = "\n".join(
        f"- Fold {f.index}: train {dates[f.train_start].date()} -> {dates[f.train_end - 1].date()}, "
        f"test {dates[f.test_start].date()} -> {dates[min(f.test_end, len(dates) - 1)].date()}"
        for f in res.folds
    )
    fail = ("\n### Failed jobs\n\n" + "\n".join(f"- {x}" for x in res.failures) + "\n") if res.failures else ""
    report = (
        "## Out-of-sample performance (stitched walk-forward test windows; PPO = mean ± std over seeds)\n\n"
        f"{md_core}\n\n### Sharpe ratio by fold (mean over seeds)\n\n{md_folds}\n\n"
        f"### Regime-conditional statistics (annualised mean %, daily CVaR95 %, worst day %)\n\n{md_regime}\n\n"
        f"### Sharpe-difference tests (paired circular block bootstrap, block=21, B=2000)\n\n{md_sig}\n\n"
        f"### Allocation behaviour\n\n{md_beh}\n\n### Folds\n\n{fold_lines}\n{fail}"
    )
    (out / "experiment_results_summary.md").write_text(report, encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run_experiments(
    cfg: Mapping[str, Any],
    variant_keys: Sequence[str] | None = None,
    skip_baselines: bool = False,
    resume: bool = False,
) -> ExperimentResult:
    """Execute the full ablation + baseline walk-forward experiment.

    Args:
        cfg: Full configuration.
        variant_keys: Subset of ablation variants (``None`` = all).
        skip_baselines: Evaluate only the RL variants.
        resume: Reuse trained models found in the output directory.

    Returns:
        The aggregated :class:`ExperimentResult`.
    """
    ecfg = cfg["experiments"]
    out = resolve_path(ecfg.get("output_dir", "outputs"))
    out.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config_used.yaml")
    P = int(cfg["environment"].get("periods_per_year", 252))

    market, featureset = prepare_data(cfg)
    test_window = int(ecfg["test_window"])
    n_folds = int(ecfg.get("n_folds", 3))
    folds = walk_forward_splits(
        market.n_periods, first_valid_index(featureset, cfg),
        train_window=int(ecfg["train_window"]), test_window=test_window, step=test_window,
        embargo=int(ecfg.get("embargo", 0)), expanding=False, n_folds=n_folds,
    )
    if len(folds) < n_folds:
        logger.warning("Only %d of %d folds fit the data", len(folds), n_folds)

    variants = build_variants(ecfg, variant_keys)
    seeds = [int(s) for s in ecfg.get("seeds", [cfg.get("seed", 0)])]
    timesteps = int(ecfg["timesteps"])
    jobs = [
        Job(v, f, s, str(out / "models" / slugify(v.key) / f"seed_{s}" / f"fold_{f.index}"), timesteps)
        for v in variants for s in seeds for f in folds
    ]
    logger.info("%d variants x %d seeds x %d folds = %d training jobs (%s timesteps each)",
                len(variants), len(seeds), len(folds), len(jobs), f"{timesteps:,}")

    baseline_frames: dict[str, list[pd.DataFrame]] = {}
    if not skip_baselines:
        baselines = build_baselines(ecfg.get("baselines", {}), float(cfg["environment"].get("max_gross_leverage", 1.0)))
        for f in folds:
            for name, strat in baselines.items():
                env = make_eval_env(cfg, market, featureset, f.evaluation_range(market.n_periods))
                baseline_frames.setdefault(name, []).append(run_episode(env, weights_fn=strat).assign(fold=f.index))
        logger.info("Evaluated %d baselines on %d folds", len(baselines), len(folds))

    t0 = time.perf_counter()
    results = execute_jobs(
        cfg, jobs,
        n_workers=min(int(ecfg.get("n_workers", 1)), len(jobs)),
        threads=int(ecfg.get("torch_threads", 2)),
        validation_fraction=float(ecfg.get("validation_fraction", 0.15)),
        deterministic=bool(ecfg.get("deterministic", True)),
        resume=resume,
        progress_path=out / "progress.log",
    )
    rl_frames: dict[str, dict[int, list[pd.DataFrame]]] = {v.label: {} for v in variants}
    failures = []
    for r in results:
        job: Job = r["job"]
        if r["records"] is None:
            failures.append(f"{job.variant.key} seed={job.seed} fold={job.fold.index}")
            continue
        rl_frames[job.variant.label].setdefault(job.seed, []).append(r["records"])
    # Only seeds with every fold present are stitched (a partial track would bias metrics).
    for label in rl_frames:
        rl_frames[label] = {s: fr for s, fr in rl_frames[label].items() if len(fr) == len(folds)}

    regimes = pd.Series(market.regimes, index=market.dates) if market.regimes is not None else None
    reference = variants[0].label if variants and variants[0].key == "A_vanilla" else None
    res = aggregate(rl_frames, baseline_frames, folds, regimes, P,
                    float(ecfg.get("risk_free_rate", 0.0)), reference)
    res.failures = failures
    report = write_artifacts(res, out, market.dates, regimes)
    print("\n" + report)
    logger.info("Finished in %.1f min (%d failed jobs); artifacts in %s",
                (time.perf_counter() - t0) / 60, len(failures), out)
    return res


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="Run the risk-penalty ablation and baseline comparison.")
    p.add_argument("--config", default=None, help="Path to hyperparameters.yaml")
    p.add_argument("--timesteps", type=int, default=None, help="PPO timesteps per job")
    p.add_argument("--n-folds", type=int, default=None, help="Number of rolling walk-forward folds")
    p.add_argument("--seeds", type=int, nargs="*", default=None, help="Training seeds")
    p.add_argument("--workers", type=int, default=None, help="Parallel training processes")
    p.add_argument("--output-dir", default=None, help="Artifact directory (default: outputs/)")
    p.add_argument("--variants", nargs="*", default=None, help="Subset of variant keys, e.g. A_vanilla D_full")
    p.add_argument("--skip-baselines", action="store_true")
    p.add_argument("--resume", action="store_true", help="Reuse trained models already in the output dir")
    p.add_argument("--smoke", action="store_true", help="Tiny data / budgets for a quick pipeline check")
    p.add_argument("--set", nargs="*", default=[], help="Config overrides: section.key=value")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point.

    Returns:
        Process exit code.
    """
    args = parse_args(argv)
    configure_logging()
    overrides: dict[str, Any] = dict(SMOKE_OVERRIDES) if args.smoke else {}
    overrides = deep_update(overrides, parse_overrides(args.set))
    cli: dict[str, Any] = {}
    for key, val in (("timesteps", args.timesteps), ("n_folds", args.n_folds), ("seeds", args.seeds),
                     ("n_workers", args.workers), ("output_dir", args.output_dir)):
        if val is not None:
            cli[key] = val
    cfg = load_config(args.config, deep_update(overrides, {"experiments": cli}))
    try:
        run_experiments(cfg, args.variants, args.skip_baselines, args.resume)
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
