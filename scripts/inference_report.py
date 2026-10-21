#!/usr/bin/env python
"""Apply the pre-registered inference plan (ANALYSIS_PLAN.md) to experiment outputs.

Writes ``inference_primary.csv`` (Holm-adjusted primary family with equivalence verdicts
and minimum detectable effects) and ``inference_secondary.csv`` (all metrics for the
primary pairs plus extra comparisons) to the first results directory.

Usage:
    python scripts/inference_report.py --results outputs_real_nextopen
    python scripts/inference_report.py --results outputs_real outputs_real_followup   # pooled
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.inference import METRICS, holm, minimum_detectable_effect, tost_equivalence, two_level_bootstrap  # noqa: E402

EW = "Equal Weight (1/N)"
MARGIN = 0.15
PRIMARY = [  # (id, strategy A prefix, strategy B prefix, gross?)
    ("P1", "PPO-E ", "PPO-A ", False),
    ("P2", "PPO-S ", "PPO-A ", False),
    ("P3", "PPO-SE ", EW, False),
    ("P4", "PPO-A ", EW, False),
    ("P5", "PPO-A ", EW, True),
]
SECONDARY_PAIRS = [
    ("PPO-SE ", "PPO-E "), ("PPO-E ", EW), ("PPO-S ", EW), ("PPO-SE ", "Risk Parity (ERC)"),
    ("PPO-A ", "Untrained policy"), ("PPO-SE ", "Untrained policy"), (EW + "", "Equal Weight (1/N, daily)"),
]


def find(st: dict, prefix: str) -> str | None:
    return next((k for k in st if k.startswith(prefix) and "+ vol" not in k), None)


def series(st: dict, name: str, gross: bool = False) -> tuple[list[np.ndarray], list[np.ndarray], pd.Index]:
    by_seed = st[name]
    frames = [by_seed[s] for s in sorted(by_seed)]
    idx = frames[0].index
    rets = [(f["return"] + (f["cost"] if gross else 0.0)).reindex(idx).to_numpy() for f in frames]
    tos = [f["turnover"].reindex(idx).to_numpy() for f in frames]
    return rets, tos, idx


def compare(st: dict, a: str, b: str, metric: str, gross: bool, n_boot: int) -> dict:
    ra, ta, ia = series(st, a, gross)
    rb, tb, ib = series(st, b, gross)
    idx = ia.intersection(ib)
    pos_a, pos_b = ia.get_indexer(idx), ib.get_indexer(idx)
    c = two_level_bootstrap([x[pos_a] for x in ra], [x[pos_b] for x in rb], metric,
                            [x[pos_a] for x in ta], [x[pos_b] for x in tb], n_boot=n_boot)
    return {"A": a, "B": b, "metric": metric, "gross": gross, "estimate": c.estimate, "ci_low": c.ci_low,
            "ci_high": c.ci_high, "ci90_low": c.ci90_low, "ci90_high": c.ci90_high, "se": c.se,
            "p_value": c.p_value, "mde80": minimum_detectable_effect(c.se), "seeds_A": c.n_seeds_a,
            "seeds_B": c.n_seeds_b, "_comp": c}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", default=["outputs_real_nextopen"])
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    st: dict = {}
    for d in args.results:
        for k, v in pickle.load((Path(d) / "oos_records.pkl").open("rb"))["stitched"].items():
            st.setdefault(k, v)
    out = Path(args.results[0])

    rows = []
    for pid, pa, pb, gross in PRIMARY:
        a, b = find(st, pa), find(st, pb)
        if a is None or b is None:
            print(f"{pid}: missing strategy ({pa} / {pb}); skipped")
            continue
        r = compare(st, a, b, "sharpe", gross, args.n_boot)
        r["id"] = pid
        r["equivalence_margin"] = MARGIN
        r["equivalence"] = tost_equivalence(r.pop("_comp"), MARGIN)
        rows.append(r)
    prim = pd.DataFrame(rows)
    if not prim.empty:
        prim["p_holm"] = holm(prim["p_value"].tolist())
        prim["significant_holm_0.05"] = prim["p_holm"] < 0.05
        prim = prim[["id", "A", "B", "gross", "estimate", "ci_low", "ci_high", "ci90_low", "ci90_high", "se",
                     "mde80", "p_value", "p_holm", "significant_holm_0.05", "equivalence", "equivalence_margin",
                     "seeds_A", "seeds_B", "metric"]]
        prim.to_csv(out / "inference_primary.csv", index=False)

    srows = []
    pairs = [(pa, pb) for _, pa, pb, _ in PRIMARY] + SECONDARY_PAIRS
    for pa, pb in dict.fromkeys(pairs):
        a, b = find(st, pa), find(st, pb)
        if a is None or b is None:
            continue
        for m in METRICS:
            r = compare(st, a, b, m, False, args.n_boot)
            r.pop("_comp")
            srows.append(r)
    sec = pd.DataFrame(srows)
    sec.to_csv(out / "inference_secondary.csv", index=False)

    with pd.option_context("display.width", 230, "display.max_columns", 30):
        if not prim.empty:
            show = prim.assign(A=prim["A"].str[:18], B=prim["B"].str[:18])
            print(show.drop(columns=["metric", "equivalence_margin"]).round(3).to_string(index=False))
        if not sec.empty:
            show = sec.assign(A=sec["A"].str[:16], B=sec["B"].str[:18])
            print(show[["A", "B", "metric", "estimate", "ci_low", "ci_high", "p_value", "mde80"]].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
