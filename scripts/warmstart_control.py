#!/usr/bin/env python
"""Warm-start vs from-scratch control for the ETF walk-forward study.

In the main study each (variant, seed) chain trains fold 0 from scratch and fine-tunes
every later fold from the previous fold's model. This script retrains a subset of
folds from scratch with the fold-0 budget (``initial_timesteps``) on exactly the same
rolling training window and PPO seed, then evaluates on the same test window. The
comparison is paired by (seed, fold) against the warm-started records in
``<results>/oos_records.pkl``.

Usage:
    python scripts/warmstart_control.py --config config/real_etf_nextopen.yaml \
        --variants S_smoothed --seeds 0 1 2 --folds 4 8 13
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from scripts.run_experiments import _init_worker, build_variants  # noqa: E402
from scripts.run_real_experiments import ChainJob, run_chain  # noqa: E402
from src.backtest import walk_forward_splits  # noqa: E402
from src.metrics import compute_performance_metrics  # noqa: E402
from src.train import first_valid_index, prepare_data  # noqa: E402
from src.utils import configure_logging, get_logger, load_config, resolve_path  # noqa: E402

logger = get_logger("warmstart_control")


def fold_metrics(df: pd.DataFrame, P: int = 252) -> dict[str, float]:
    m = compute_performance_metrics(df["return"].to_numpy(), periods_per_year=P)
    return {"sharpe": m["sharpe"], "ann_return_pct": m["annualized_return_pct"],
            "maxdd_pct": m["max_drawdown_pct"], "turnover_pct_day": 100 * float(df["turnover"].mean())}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="config/real_etf_nextopen.yaml")
    p.add_argument("--variants", nargs="*", default=["S_smoothed"])
    p.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    p.add_argument("--folds", type=int, nargs="*", default=[4, 8, 13])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--timesteps", type=int, default=None, help="override initial_timesteps (smoke tests)")
    p.add_argument("--out", default=None, help="output directory (default <results>/warmstart_control)")
    args = p.parse_args(argv)
    configure_logging()
    cfg = load_config(resolve_path(args.config))
    ecfg = cfg["experiments"]
    res_dir = resolve_path(ecfg["output_dir"])
    out = Path(args.out) if args.out else res_dir / "warmstart_control"
    out.mkdir(parents=True, exist_ok=True)

    market, featureset = prepare_data(cfg)
    folds = walk_forward_splits(
        market.n_periods, first_valid_index(featureset, cfg),
        train_window=int(ecfg["train_window"]), test_window=int(ecfg["test_window"]),
        step=int(ecfg["test_window"]), embargo=int(ecfg.get("embargo", 0)), expanding=False,
        n_folds=int(ecfg["n_folds"]), anchor=str(ecfg.get("fold_anchor", "end")),
    )
    chosen = [f for f in folds if f.index in set(args.folds)]
    variants = build_variants(ecfg, args.variants)
    budget = args.timesteps or int(ecfg["initial_timesteps"])
    jobs = [ChainJob(v, s, (f,), str(out / "models"), budget, 0)
            for v in variants for s in args.seeds for f in chosen]
    logger.info("%d from-scratch jobs x %s steps", len(jobs), f"{budget:,}")

    t0 = time.perf_counter()
    scratch: dict[tuple[str, int, int], pd.DataFrame] = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs)), mp_context=mp.get_context("spawn"),
                             initializer=_init_worker, initargs=(int(ecfg.get("torch_threads", 2)),)) as pool:
        futs = [pool.submit(run_chain, cfg, j, True, True) for j in jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            j = r["job"]
            if r["error"]:
                logger.error("FAILED %s seed=%d fold=%d\n%s", j.variant.key, j.seed, j.folds[0].index, r["error"])
                continue
            scratch[(j.variant.label, j.seed, j.folds[0].index)] = r["records"][j.variant.label][0]
            line = (f"[{i}/{len(jobs)}] {j.variant.key} seed={j.seed} fold={j.folds[0].index} "
                    f"({r['seconds'] / 60:.1f} min) | elapsed {(time.perf_counter() - t0) / 60:.1f} min")
            logger.info(line)
            with (out / "progress.log").open("a") as fh:
                fh.write(line + "\n")
    with (out / "scratch_records.pkl").open("wb") as fh:
        pickle.dump(scratch, fh)

    warm = pickle.load((res_dir / "oos_records.pkl").open("rb"))["stitched"]
    rows = []
    for (label, seed, fold), df_s in sorted(scratch.items()):
        df_w = warm[label][seed]
        df_w = df_w[df_w["fold"] == fold]
        for name, df in (("warm_start", df_w), ("from_scratch", df_s)):
            rows.append({"variant": label, "seed": seed, "fold": fold, "training": name, **fold_metrics(df)})
    long = pd.DataFrame(rows)
    long.to_csv(out / "warmstart_vs_scratch_by_fold.csv", index=False)
    wide = long.pivot_table(index=["variant", "seed", "fold"], columns="training",
                            values=["sharpe", "ann_return_pct", "maxdd_pct", "turnover_pct_day"])
    summary = []
    for label in wide.index.get_level_values(0).unique():
        sub = wide.loc[label]
        for metric in ("sharpe", "ann_return_pct", "maxdd_pct", "turnover_pct_day"):
            d = (sub[(metric, "from_scratch")] - sub[(metric, "warm_start")]).to_numpy()
            rng = np.random.default_rng(7)
            boots = np.array([rng.choice(d, d.size).mean() for _ in range(5000)])
            summary.append({"variant": label, "metric": metric, "n_pairs": d.size,
                            "warm_mean": sub[(metric, "warm_start")].mean(),
                            "scratch_mean": sub[(metric, "from_scratch")].mean(),
                            "diff_mean": d.mean(), "ci_lo": np.quantile(boots, 0.025),
                            "ci_hi": np.quantile(boots, 0.975), "n_scratch_better": int((d > 0).sum())})
    pd.DataFrame(summary).to_csv(out / "warmstart_vs_scratch_summary.csv", index=False)
    print(pd.DataFrame(summary).round(3).to_string(index=False))
    logger.info("Finished in %.1f min", (time.perf_counter() - t0) / 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
