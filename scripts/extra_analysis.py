#!/usr/bin/env python
"""Secondary analyses on a results directory written by either experiment runner.

Writes to the results directory:

* ``gross_and_tracking.csv`` - net vs gross-of-cost Sharpe, correlation / beta /
  tracking error against 1/N, and bootstrap tests of gross Sharpe vs 1/N.
* ``derisking.csv``          - per (strategy, seed) cash weight in calm vs stress
  regimes and the correlation of cash weight with trailing 21-day realised vol.
* ``lagrangian_path.csv``    - RCPO multipliers at the end of every fold (if any).
* ``timing_vs_tilt.csv``     - average asset weights of each PPO variant and the gross Sharpe
  of a static portfolio holding those average weights (separates timing from static tilts).

Usage:
    python scripts/extra_analysis.py --results outputs_real
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.metrics import compute_performance_metrics, sharpe_difference_bootstrap  # noqa: E402

EW = "Equal Weight (1/N)"


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="outputs")
    args = p.parse_args(argv)
    res = Path(args.results)
    rec = pickle.load((res / "oos_records.pkl").open("rb"))
    st, regimes = rec["stitched"], rec["regimes"]
    ew = st[EW][0]
    ew_net, ew_gross = ew["return"], ew["return"] + ew["cost"]

    rows = []
    for name, by_seed in st.items():
        if not by_seed:
            continue
        net, gross, corr, beta, te = [], [], [], [], []
        for d in by_seed.values():
            a = d["return"].reindex(ew_net.index)
            net.append(compute_performance_metrics(d["return"])["sharpe"])
            gross.append(compute_performance_metrics(d["return"] + d["cost"])["sharpe"])
            corr.append(float(np.corrcoef(a, ew_net)[0, 1]))
            beta.append(float(np.cov(a, ew_net)[0, 1] / ew_net.var()))
            te.append(float((a - ew_net).std() * np.sqrt(252) * 100))
        mean_gross = pd.concat([d["return"] + d["cost"] for d in by_seed.values()], axis=1).mean(axis=1)
        test = (sharpe_difference_bootstrap(mean_gross.to_numpy(), ew_gross.reindex(mean_gross.index).to_numpy(),
                                            n_boot=2000, seed=7) if name != EW else {})
        rows.append({
            "strategy": name, "net_sharpe": np.mean(net), "gross_sharpe": np.mean(gross),
            "gross_sharpe_std": np.std(gross, ddof=1) if len(gross) > 1 else np.nan,
            "corr_ew": np.mean(corr), "beta_ew": np.mean(beta), "te_vs_ew_pct": np.mean(te),
            "gross_diff_vs_ew": test.get("diff", np.nan), "gross_ci_low": test.get("ci_low", np.nan),
            "gross_ci_high": test.get("ci_high", np.nan), "gross_p_value": test.get("p_value", np.nan),
        })
    gt = pd.DataFrame(rows)
    gt.to_csv(res / "gross_and_tracking.csv", index=False)

    drows = []
    if regimes is not None:
        for name, by_seed in st.items():
            if not name.startswith("PPO"):
                continue
            for s, d in by_seed.items():
                r = regimes.reindex(d.index).fillna(0).to_numpy()
                vol = d["return"].rolling(21).std() * np.sqrt(252)
                ok = vol.notna().to_numpy()
                drows.append({
                    "strategy": name, "seed": s,
                    "cash_calm": float(d["cash_weight"][r == 0].mean()),
                    "cash_stress": float(d["cash_weight"][r == 1].mean()),
                    "corr_cash_trailvol": float(np.corrcoef(d["cash_weight"][ok], vol[ok])[0, 1]),
                })
    pd.DataFrame(drows).to_csv(res / "derisking.csv", index=False)

    lrows = []
    for path in sorted(res.glob("models/*/seed_*/fold_*/lagrangian.json")):
        d = json.loads(path.read_text())
        lrows.append({"variant": path.parts[-4], "seed": path.parts[-3], "fold": path.parts[-2], **d})
    if lrows:
        pd.DataFrame(lrows).to_csv(res / "lagrangian_path.csv", index=False)

    # Exposure-matched control: 1/N scaled to each PPO variant's average gross exposure (rest in cash).
    from src.metrics import conditional_metrics

    beh = pd.read_csv(res / "behaviour.csv", index_col=0)
    crows = []
    for name, by_seed in st.items():
        if not name.startswith("PPO") or "+ vol" in name:
            continue
        cash = float(beh.loc[name, "avg_cash_weight"])
        r = pd.concat([d["return"] for d in by_seed.values()], axis=1).mean(axis=1)
        delev = ((1.0 - cash) * ew_net).reindex(r.index)
        ma, mb = compute_performance_metrics(r), compute_performance_metrics(delev)
        row = {"strategy": name, "avg_cash": cash, "sharpe": ma["sharpe"], "sharpe_delev": mb["sharpe"],
               "maxdd": ma["max_drawdown_pct"], "maxdd_delev": mb["max_drawdown_pct"]}
        test = sharpe_difference_bootstrap(r.to_numpy(), delev.to_numpy(), n_boot=2000, seed=7)
        row.update({"diff": test["diff"], "p_value": test["p_value"]})
        if regimes is not None:
            mask = regimes.reindex(r.index).fillna(0).to_numpy() == 1
            sa, sb = conditional_metrics(r.to_numpy(), mask), conditional_metrics(delev.to_numpy(), mask)
            row.update({"stress_cvar": sa["cvar_95_pct"], "stress_cvar_delev": sb["cvar_95_pct"],
                        "stress_worst": sa["worst_day_pct"], "stress_worst_delev": sb["worst_day_pct"]})
        crows.append(row)
    pd.DataFrame(crows).to_csv(res / "delevered_ew_control.csv", index=False)

    # Timing vs static tilt: gross Sharpe of holding each variant's average weights.
    from src.train import prepare_data
    from src.utils import load_config

    market, _ = prepare_data(load_config(res / "config_used.yaml"))
    rets = pd.DataFrame(market.close[1:] / market.close[:-1] - 1.0, index=market.dates[1:], columns=market.assets)
    rets = rets.reindex(ew_net.index)
    trows = []
    for name, by_seed in st.items():
        if not name.startswith("PPO") or "+ vol" in name:
            continue
        w = pd.concat([d[[f"w_{a}" for a in market.assets]] for d in by_seed.values()]).mean().to_numpy()
        static = compute_performance_metrics(rets.to_numpy() @ w)["sharpe"]
        dyn = gt.loc[gt["strategy"] == name, "gross_sharpe"].iloc[0]
        trows.append({"strategy": name, **{f"w_{a}": float(x) for a, x in zip(market.assets, w)},
                      "cash": float(1 - w.sum()), "static_tilt_gross_sharpe": static,
                      "dynamic_gross_sharpe": dyn, "timing_contribution": dyn - static})
    trows.append({"strategy": "1/N (static, daily rebalanced)",
                  "static_tilt_gross_sharpe": compute_performance_metrics(rets.mean(axis=1).to_numpy())["sharpe"]})
    pd.DataFrame(trows).to_csv(res / "timing_vs_tilt.csv", index=False)

    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(pd.DataFrame(trows)[["strategy", "static_tilt_gross_sharpe", "dynamic_gross_sharpe",
                                   "timing_contribution"]].round(3).to_string(index=False))
        print(gt.assign(strategy=gt["strategy"].str[:30]).round(3).to_string(index=False))
        if drows:
            dd = pd.DataFrame(drows).drop(columns="seed").groupby("strategy", sort=False).agg(["mean", "std"])
            print(dd.round(3).to_string())
        if lrows:
            lp = pd.DataFrame(lrows)
            print(lp.groupby("fold")[["lambda_cvar", "lambda_drawdown"]].mean().round(3).T.to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
