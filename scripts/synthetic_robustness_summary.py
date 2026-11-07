#!/usr/bin/env python
"""Summarise the independent simulated markets (outputs_syn_nextopen/<market>/).

For each market: 1/N and untrained-policy Sharpe, PPO-S / PPO-SE Sharpe (mean and SD over
seeds), two-level bootstrap intervals for S − 1/N and SE − S, maximum drawdowns, and the
average cash weight of each agent in calm vs stress regimes (ground-truth labels).

Usage:
    python scripts/synthetic_robustness_summary.py --root outputs_syn_nextopen
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

from src.inference import m_maxdd, m_sharpe, two_level_bootstrap  # noqa: E402

EW = "Equal Weight (1/N)"
UNTRAINED = "Untrained policy (1/(N+1) incl. cash)"
NAMES = {"seed_101": "Standard, data seed 101", "seed_202": "Standard, data seed 202",
         "seed_303": "Standard, data seed 303", "bear_404": "Bear, data seed 404"}


def find(st: dict, prefix: str) -> str:
    return next(k for k in st if k.startswith(prefix))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="outputs_syn_nextopen")
    p.add_argument("--n-boot", type=int, default=2000)
    args = p.parse_args(argv)
    root = Path(args.root)
    rows = []
    for d in sorted(root.iterdir(), key=lambda x: (x.name.startswith("bear"), x.name)):
        f = d / "oos_records.pkl"
        if not f.exists():
            continue
        rec = pickle.load(f.open("rb"))
        st, regimes = rec["stitched"], rec["regimes"]
        s_lab, se_lab = find(st, "PPO-S "), find(st, "PPO-SE ")
        idx = st[EW][0].index

        def arr(lab: str) -> list[np.ndarray]:
            return [x["return"].reindex(idx).to_numpy() for x in st[lab].values()]

        s, se, ew, un = arr(s_lab), arr(se_lab), arr(EW), arr(UNTRAINED)
        c1 = two_level_bootstrap(s, ew, "sharpe", n_boot=args.n_boot)
        c2 = two_level_bootstrap(se, s, "sharpe", n_boot=args.n_boot)
        reg = regimes.reindex(idx).fillna(0).to_numpy() if regimes is not None else np.zeros(len(idx))

        def cash(lab: str, r: int) -> float:
            return float(np.mean([x["cash_weight"].reindex(idx).to_numpy()[reg == r].mean() for x in st[lab].values()]))

        rows.append({
            "market": NAMES.get(d.name, d.name), "n_days": len(idx), "stress_share": float((reg == 1).mean()),
            "ew_sharpe": m_sharpe(ew[0]), "untrained_sharpe": m_sharpe(un[0]),
            "s_sharpe": np.mean([m_sharpe(x) for x in s]), "s_sharpe_sd": np.std([m_sharpe(x) for x in s], ddof=1),
            "se_sharpe": np.mean([m_sharpe(x) for x in se]),
            "se_sharpe_sd": np.std([m_sharpe(x) for x in se], ddof=1),
            "s_minus_ew": c1.estimate, "s_minus_ew_lo": c1.ci_low, "s_minus_ew_hi": c1.ci_high, "s_minus_ew_p": c1.p_value,
            "se_minus_s": c2.estimate, "se_minus_s_lo": c2.ci_low, "se_minus_s_hi": c2.ci_high, "se_minus_s_p": c2.p_value,
            "ew_maxdd": m_maxdd(ew[0]), "s_maxdd": np.mean([m_maxdd(x) for x in s]),
            "se_maxdd": np.mean([m_maxdd(x) for x in se]),
            "s_cash_calm": cash(s_lab, 0), "s_cash_stress": cash(s_lab, 1),
            "se_cash_calm": cash(se_lab, 0), "se_cash_stress": cash(se_lab, 1),
            "n_seeds": len(s),
        })
    out = pd.DataFrame(rows)
    out.to_csv(root / "synthetic_markets_summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 40):
        print(out.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
