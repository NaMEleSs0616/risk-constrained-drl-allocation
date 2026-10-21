#!/usr/bin/env python
"""Paired comparison of follow-up variants against their counterparts in the original study.

Both studies use the same data, folds, seeds and evaluation dates, so seed-averaged daily
return series can be compared with the paired block bootstrap.

Usage:
    python scripts/compare_studies.py --base outputs_real --followup outputs_real_followup
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402

from src.metrics import compute_performance_metrics, sharpe_difference_bootstrap  # noqa: E402

PAIRS = [  # (follow-up label prefix, base label prefix)
    ("PPO-S ", "PPO-A Vanilla (l1=0, l2=0)"),
    ("PPO-SE ", "PPO-E "),
    ("PPO-RT ", "PPO-R "),
    ("PPO-SRT ", "PPO-R "),
    ("PPO-SRT ", "PPO-S "),
]


def seed_mean(by_seed: dict, col: str = "return") -> pd.Series:
    return pd.concat([d[col] for d in by_seed.values()], axis=1).mean(axis=1)


def find(st: dict, prefix: str) -> str | None:
    return next((k for k in st if k.startswith(prefix) and "+ vol" not in k), None)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base", default="outputs_real")
    p.add_argument("--followup", default="outputs_real_followup")
    args = p.parse_args(argv)
    base = pickle.load((Path(args.base) / "oos_records.pkl").open("rb"))["stitched"]
    fu = pickle.load((Path(args.followup) / "oos_records.pkl").open("rb"))["stitched"]
    pool = {**base, **{k: v for k, v in fu.items() if k not in base}}
    rows = []
    for a_pfx, b_pfx in PAIRS:
        a, b = find(fu, a_pfx), find(pool, b_pfx)
        if a is None or b is None:
            continue
        ra, rb = seed_mean(pool[a]), seed_mean(pool[b])
        ga, gb = ra + seed_mean(pool[a], "cost"), rb + seed_mean(pool[b], "cost")
        idx = ra.index.intersection(rb.index)
        net = sharpe_difference_bootstrap(ra[idx].to_numpy(), rb[idx].to_numpy(), n_boot=2000, seed=7)
        gross = sharpe_difference_bootstrap(ga[idx].to_numpy(), gb[idx].to_numpy(), n_boot=2000, seed=7)
        ma, mb = compute_performance_metrics(ra[idx]), compute_performance_metrics(rb[idx])
        to_a = float(seed_mean(pool[a], "turnover").mean())
        to_b = float(seed_mean(pool[b], "turnover").mean())
        rows.append({
            "strategy": a, "versus": b,
            "net_diff": net["diff"], "net_ci_low": net["ci_low"], "net_ci_high": net["ci_high"], "net_p": net["p_value"],
            "gross_diff": gross["diff"], "gross_p": gross["p_value"],
            "maxdd_a": ma["max_drawdown_pct"], "maxdd_b": mb["max_drawdown_pct"],
            "turnover_a": to_a, "turnover_b": to_b,
        })
    out = pd.DataFrame(rows)
    out.to_csv(Path(args.followup) / "followup_vs_original.csv", index=False)
    with pd.option_context("display.width", 220):
        print(out.assign(strategy=out["strategy"].str[:26], versus=out["versus"].str[:26]).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
