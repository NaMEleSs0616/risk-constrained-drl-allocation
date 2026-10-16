#!/usr/bin/env python
"""Selection-free robustness check: evaluate every trained agent at fixed checkpoints.

``run_experiments.py`` keeps the checkpoint with the best *validation reward*. Because
that reward includes the variant's own penalties, model selection can interact with
the penalty (e.g. favouring early, barely-trained policies). This script removes the
selection step: every (variant, seed, fold) agent is evaluated out of sample at the
same fixed checkpoints (50k, 100k and the final 150k steps), and two non-learning
controls are added:

* ``Initial policy (1/6 each incl. cash)`` - what an untrained softmax policy outputs
  (zero logits -> equal weight on the five assets and cash), rebalanced daily.
* ``Selected (validation)`` numbers are read from the main run for comparison.

Usage:
    python scripts/checkpoint_analysis.py --results outputs
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.run_experiments import build_variants, slugify, stitch  # noqa: E402
from src.backtest import load_policy, make_eval_env, run_episode, walk_forward_splits  # noqa: E402
from src.metrics import metrics_from_records, sharpe_difference_bootstrap  # noqa: E402
from src.train import first_valid_index, prepare_data  # noqa: E402
from src.utils import configure_logging, load_config  # noqa: E402

CHECKPOINTS = {"50k": "checkpoints/ppo_50000_steps.zip", "100k": "checkpoints/ppo_100000_steps.zip",
               "150k (final)": "final_model.zip", "selected": "best_model/best_model.zip"}
KEYS = ["cumulative_return_pct", "sharpe", "sortino", "max_drawdown_pct", "calmar", "cvar_95_pct",
        "avg_daily_turnover", "total_cost_pct"]


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="outputs")
    args = p.parse_args(argv)
    configure_logging(logging.WARNING)
    res = Path(args.results)
    cfg = load_config(res / "config_used.yaml")
    ecfg = cfg["experiments"]
    market, featureset = prepare_data(cfg)
    folds = walk_forward_splits(
        market.n_periods, first_valid_index(featureset, cfg), int(ecfg["train_window"]),
        int(ecfg["test_window"]), step=int(ecfg["test_window"]), embargo=int(ecfg["embargo"]),
        n_folds=int(ecfg["n_folds"]),
    )
    P = int(cfg["environment"]["periods_per_year"])
    rows, series = [], {}

    # Non-learning control: the untrained policy's allocation.
    n = market.n_assets
    ctrl = []
    for f in folds:
        env = make_eval_env(cfg, market, featureset, f.evaluation_range(market.n_periods))
        ctrl.append(run_episode(env, weights_fn=lambda e: np.full(n, 1.0 / (n + 1))).assign(fold=f.index))
    df = stitch(ctrl)
    series[("Initial policy (1/6 incl. cash)", "-", 0)] = df["return"]
    rows.append({"strategy": "Initial policy (1/6 incl. cash)", "checkpoint": "-", "seed": 0,
                 **metrics_from_records(df, P)})

    for v in build_variants(ecfg, None):
        for ck_name, rel in CHECKPOINTS.items():
            for seed in ecfg["seeds"]:
                frames = []
                for f in folds:
                    run_dir = res / "models" / slugify(v.key) / f"seed_{seed}" / f"fold_{f.index}"
                    path = run_dir / rel
                    if not path.exists():
                        break
                    env = make_eval_env(cfg, market, featureset, f.evaluation_range(market.n_periods),
                                        v.reward_overrides)
                    frames.append(run_episode(env, action_fn=load_policy(path, env, None)).assign(fold=f.index))
                if len(frames) != len(folds):
                    continue
                df = stitch(frames)
                series[(v.label, ck_name, seed)] = df["return"]
                rows.append({"strategy": v.label, "checkpoint": ck_name, "seed": seed, **metrics_from_records(df, P)})
            print(f"evaluated {v.key} @ {ck_name}", flush=True)

    long = pd.DataFrame(rows)
    long.to_csv(res / "checkpoint_seed_metrics.csv", index=False)
    summ = long.groupby(["strategy", "checkpoint"], sort=False)[KEYS].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.to_csv(res / "checkpoint_summary.csv")

    # Sharpe-difference tests vs variant A at the same checkpoint (seed-mean returns).
    tests = []
    labels = [v.label for v in build_variants(ecfg, None)]
    for ck in CHECKPOINTS:
        def mean_ret(lab: str) -> pd.Series:
            return pd.concat([s for (l, c, _), s in series.items() if l == lab and c == ck], axis=1).mean(axis=1)
        ref = mean_ret(labels[0])
        for lab in labels[1:]:
            r = sharpe_difference_bootstrap(mean_ret(lab).to_numpy(), ref.to_numpy(), P, n_boot=2000, seed=7)
            tests.append({"checkpoint": ck, "strategy": lab, "versus": labels[0], **r})
    pd.DataFrame(tests).to_csv(res / "checkpoint_tests.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(summ[[f"{k}_mean" for k in ("sharpe", "max_drawdown_pct", "cvar_95_pct", "avg_daily_turnover",
                                          "total_cost_pct")] + ["sharpe_std"]].round(3))
        print(pd.DataFrame(tests).round(3))
    return 0


if __name__ == "__main__":
    sys.exit(main())
