#!/usr/bin/env python
"""Orchestration entry point.

Commands:
    train        Train PPO on the train split (model selection on the validation split).
    evaluate     Evaluate a trained model and the baselines on the held-out test split.
    backtest     Walk-forward backtest (re-train per fold) vs. classical baselines.
    experiments  Risk-penalty ablation (models A-D) + baselines, see scripts/run_experiments.py.

Examples:
    python main.py train --set ppo.total_timesteps=200000
    python main.py evaluate --model runs/ppo_risk_constrained/best_model/best_model.zip
    python main.py backtest --smoke
    python main.py experiments --smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from src.utils import (
    SMOKE_OVERRIDES,
    configure_logging,
    deep_update,
    get_logger,
    load_config,
    parse_overrides,
    resolve_path,
)

logger = get_logger("main")


def _evaluate(cfg: dict[str, Any], model_path: str | None, vecnorm_path: str | None) -> pd.DataFrame:
    """Evaluate a trained model and all baselines on the configured test split."""
    from src.backtest import load_policy, make_eval_env, plot_equity_curves, run_episode
    from src.baselines import build_baselines
    from src.metrics import METRIC_LABELS, metrics_from_records, summary_table, to_markdown_table
    from src.train import first_valid_index, prepare_data, split_indices

    market, featureset = prepare_data(cfg)
    tcfg = cfg["training"]
    splits = split_indices(
        market.n_periods, first_valid_index(featureset, cfg),
        float(tcfg["train_fraction"]), float(tcfg["val_fraction"]), int(tcfg.get("embargo", 0)),
    )
    run_dir = resolve_path(cfg.get("output_dir", "runs")) / cfg.get("experiment_name", "ppo")
    if model_path is None:
        best = run_dir / "best_model" / "best_model.zip"
        model_path = str(best if best.exists() else run_dir / "final_model.zip")
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model not found: {model_path}; run `python main.py train` first")
    vecnorm_path = vecnorm_path or str(run_dir / "vecnormalize.pkl")

    P = int(cfg["environment"].get("periods_per_year", 252))
    records: dict[str, pd.DataFrame] = {}
    env = make_eval_env(cfg, market, featureset, splits["test"])
    records["PPO (risk-constrained)"] = run_episode(env, action_fn=load_policy(model_path, env, vecnorm_path))
    baselines = build_baselines(cfg.get("experiments", {}).get("baselines", {}),
                                float(cfg["environment"].get("max_gross_leverage", 1.0)))
    for name, strat in baselines.items():
        records[name] = run_episode(make_eval_env(cfg, market, featureset, splits["test"]), weights_fn=strat)

    metrics = {name: metrics_from_records(df, P) for name, df in records.items()}
    out = run_dir / "evaluation"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metrics).T.to_csv(out / "test_metrics.csv")
    for name, df in records.items():
        df = df.copy()
        df["equity"] = (1.0 + df["return"]).cumprod()
        df["oos_drawdown"] = 1.0 - df["equity"] / df["equity"].cummax()
        records[name] = df
    plot_equity_curves(records, out / "equity_curves.png", title="Test split: NAV (net of costs)")
    table = summary_table(metrics)
    print("\n" + to_markdown_table(table, column_formats={METRIC_LABELS["total_cost_usd"]: ",.0f"}) + "\n")
    logger.info("Evaluation artifacts written to %s", out)
    return table


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["train", "evaluate", "backtest", "experiments"])
    parser.add_argument("--config", default=None, help="Path to hyperparameters.yaml")
    parser.add_argument("--set", nargs="*", default=[], help="Overrides: section.key=value")
    parser.add_argument("--smoke", action="store_true", help="Tiny budgets for a quick end-to-end check")
    parser.add_argument("--model", default=None, help="[evaluate] model .zip path")
    parser.add_argument("--vecnormalize", default=None, help="[evaluate] VecNormalize stats path")
    parser.add_argument("--log-level", default="INFO")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a CLI command.

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    configure_logging(args.log_level.upper())
    overrides = deep_update(dict(SMOKE_OVERRIDES) if args.smoke else {}, parse_overrides(args.set))
    cfg = load_config(args.config, overrides)

    if args.command == "train":
        from src.train import run_training

        result = run_training(cfg)
        logger.info("Best model: %s", result.best_model_path)
    elif args.command == "evaluate":
        _evaluate(cfg, args.model, args.vecnormalize)
    elif args.command == "backtest":
        from src.backtest import walk_forward_backtest

        walk_forward_backtest(cfg)
    elif args.command == "experiments":
        from scripts.run_experiments import run_experiments

        run_experiments(cfg)
    return 0


if __name__ == "__main__":
    sys.exit(main())
