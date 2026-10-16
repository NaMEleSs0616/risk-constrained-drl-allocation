#!/usr/bin/env python
"""Fill the paper template's ``<!--TABLE:name-->`` placeholders from experiment outputs.

Every number in the results tables is read from the CSVs written by
``scripts/run_experiments.py``; nothing is typed by hand.

Usage:
    python scripts/build_paper.py --results outputs --template paper/paper_template.html --out paper/paper.html
"""

from __future__ import annotations

import argparse
import html
import math
import re
import sys
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VARIANT_SHORT = {
    "A": "A · return only",
    "B": "B · CVaR only",
    "C": "C · drawdown only",
    "D": "D · CVaR + drawdown",
    "E": "E · calibrated CVaR + DD",
}


def label(name: str) -> str:
    """Display label for a strategy name."""
    m = re.match(r"PPO-([A-Z]) .*\((l1=[^,]+), (l2=[^)]+)\)", name)
    if m:
        letter, l1, l2 = m.groups()
        lam = f"λ₁={l1[3:]}, λ₂={l2[3:]}"
        return f"PPO-{VARIANT_SHORT.get(letter, letter)} <span class='lam'>({lam})</span>"
    return html.escape(name)


def is_rl(name: str) -> bool:
    return name.startswith("PPO-")


def fmt(x: float, spec: str) -> str:
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "–"
    s = format(x, spec)
    return s.replace("-", "−")


def pm(mean: float, std: float, spec: str) -> str:
    if not math.isfinite(mean):
        return "–"
    if std is None or not math.isfinite(std):
        return fmt(mean, spec)
    return f"{fmt(mean, spec)}<span class='pm'> ± {fmt(std, spec)}</span>"


def render(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str, tid: str,
           group_break: int | None = None, num_from: int = 1, notes: str = "") -> str:
    head = "".join(f"<th{' class=num' if i >= num_from else ''}>{h}</th>" for i, h in enumerate(headers))
    body = []
    for r_i, row in enumerate(rows):
        cls = " class='group-start'" if group_break is not None and r_i == group_break else ""
        cells = "".join(f"<td{' class=num' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(row))
        body.append(f"<tr{cls}>{cells}</tr>")
    note = f"<p class='table-note'>{notes}</p>" if notes else ""
    return (f"<figure class='table' id='{tid}'><figcaption>{caption}</figcaption><div class='scroll'>"
            f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>{note}</figure>")


def best_marks(values: dict[str, float], higher: bool) -> set[str]:
    finite = {k: v for k, v in values.items() if math.isfinite(v)}
    if not finite:
        return set()
    target = max(finite.values()) if higher else min(finite.values())
    return {k for k, v in finite.items() if abs(v - target) < 1e-12}


def table_main(res: Path) -> str:
    mean = pd.read_csv(res / "experiment_results_full.csv", index_col=0)
    std = pd.read_csv(res / "experiment_results_full_std.csv", index_col=0)
    cols = [("cumulative_return_pct", "Cum. return (%)", ".1f", True),
            ("annualized_return_pct", "Ann. return (%)", ".1f", True),
            ("annualized_vol_pct", "Ann. vol (%)", ".1f", False),
            ("sharpe", "Sharpe", ".2f", True), ("sortino", "Sortino", ".2f", True),
            ("max_drawdown_pct", "MaxDD (%)", ".1f", False), ("calmar", "Calmar", ".2f", True),
            ("cvar_95_pct", "CVaR<sub>95</sub> (%)", ".2f", False)]
    best = {c: best_marks(mean[c].to_dict(), hi) for c, _, _, hi in cols}
    rows = []
    for name in mean.index:
        row = [label(name)]
        for c, _, spec, _ in cols:
            cell = pm(mean.at[name, c], std.at[name, c] if is_rl(name) else float("nan"), spec)
            row.append(f"<strong>{cell}</strong>" if name in best[c] else cell)
        rows.append(row)
    n_rl = sum(is_rl(n) for n in mean.index)
    return render(["Strategy"] + [h for _, h, _, _ in cols], rows,
                  "<b>Table 3.</b> Out-of-sample performance on the stitched walk-forward test period "
                  "(3 folds × 252 days, net of all frictions). PPO rows: mean ± standard deviation across "
                  "three training seeds. Bold marks the best value in each column.", "tab-main",
                  group_break=n_rl,
                  notes="CVaR<sub>95</sub> is the daily historical expected shortfall of net returns. "
                        "Annualised return is geometric; Sharpe and Sortino use a zero risk-free rate.")


def table_costs(res: Path) -> str:
    mean = pd.read_csv(res / "experiment_results_full.csv", index_col=0)
    std = pd.read_csv(res / "experiment_results_full_std.csv", index_col=0)
    beh = pd.read_csv(res / "behaviour.csv", index_col=0)
    rows = []
    for name in mean.index:
        s = (lambda c: std.at[name, c]) if is_rl(name) else (lambda c: float("nan"))
        rows.append([
            label(name),
            pm(100 * mean.at[name, "avg_daily_turnover"], 100 * s("avg_daily_turnover"), ".2f"),
            pm(mean.at[name, "annualized_turnover"], s("annualized_turnover"), ".1f"),
            pm(100 * mean.at[name, "total_cost_pct"], 100 * s("total_cost_pct"), ".0f"),
            pm(mean.at[name, "total_cost_usd"] / 1e3, s("total_cost_usd") / 1e3, ",.0f"),
            fmt(beh.at[name, "avg_gross_exposure"] * 100, ".1f"),
            fmt(beh.at[name, "avg_effective_n"], ".2f"),
            fmt(beh.at[name, "max_single_weight"] * 100, ".1f"),
        ])
    return render(["Strategy", "Daily turnover (%)", "Annual turnover (×)", "Total cost (bp of NAV)",
                   "Total cost ($k)", "Avg gross exposure (%)", "Effective N", "Max weight (%)"], rows,
                  "<b>Table 4.</b> Trading activity, transaction costs and allocation behaviour over the "
                  "756-day out-of-sample period (initial NAV $10m per fold).", "tab-costs",
                  group_break=sum(is_rl(n) for n in mean.index),
                  notes="Total cost sums commission, half-spread, temporary and permanent impact, "
                        "expressed per period as a fraction of pre-trade NAV. Effective N = (Σ|w|)² / Σw².")


def table_folds(res: Path) -> str:
    fm = pd.read_csv(res / "fold_metrics.csv")
    order = list(dict.fromkeys(fm["strategy"]))
    rows = []
    for name in order:
        g = fm[fm["strategy"] == name]
        row = [label(name)]
        for f in sorted(g["fold"].unique()):
            sub = g[g["fold"] == f]
            row.append(pm(sub["sharpe"].mean(), sub["sharpe"].std(ddof=1) if is_rl(name) else float("nan"), ".2f"))
            row.append(pm(sub["max_drawdown_pct"].mean(),
                          sub["max_drawdown_pct"].std(ddof=1) if is_rl(name) else float("nan"), ".1f"))
        rows.append(row)
    folds = sorted(fm["fold"].unique())
    headers = ["Strategy"] + [h for f in folds for h in (f"Fold {f} Sharpe", f"Fold {f} MaxDD (%)")]
    return render(headers, rows, "<b>Table 5.</b> Per-fold out-of-sample Sharpe ratio and maximum drawdown "
                  "(PPO: mean ± std over seeds).", "tab-folds", group_break=sum(is_rl(n) for n in order))


def table_regime(res: Path) -> str:
    rm = pd.read_csv(res / "regime_metrics.csv")
    order = list(dict.fromkeys(rm["strategy"]))
    rows = []
    for name in order:
        g = rm[rm["strategy"] == name].set_index("regime")
        rows.append([label(name)] + [fmt(g.at[r, c], ".2f" if c != "ann_mean_pct" else ".1f")
                                     for r in ("calm", "stress")
                                     for c in ("ann_mean_pct", "ann_vol_pct", "cvar_95_pct", "worst_day_pct")])
    n_calm = int(rm[(rm["regime"] == "calm")]["n"].iloc[0])
    n_stress = int(rm[(rm["regime"] == "stress")]["n"].iloc[0])
    headers = ["Strategy", "Calm: ann. mean (%)", "Calm: ann. vol (%)", "Calm: CVaR<sub>95</sub> (%)",
               "Calm: worst day (%)", "Stress: ann. mean (%)", "Stress: ann. vol (%)",
               "Stress: CVaR<sub>95</sub> (%)", "Stress: worst day (%)"]
    return render(headers, rows, f"<b>Table 6.</b> Regime-conditional statistics of net daily returns, using the "
                  f"simulator's ground-truth regime labels ({n_calm} calm and {n_stress} stress days out of sample). "
                  "PPO rows average the per-seed statistics.", "tab-regime",
                  group_break=sum(is_rl(n) for n in order),
                  notes="Regime labels are never observed by any strategy; they are used only for this analysis.")


def table_sig(res: Path) -> str:
    sig = pd.read_csv(res / "significance.csv")
    rows = []
    for _, r in sig.iterrows():
        p = r["p_value"]
        star = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        rows.append([label(r["strategy"]), label(r["versus"]), fmt(r["diff"], "+.3f").replace("+", "+"),
                     f"[{fmt(r['ci_low'], '.3f')}, {fmt(r['ci_high'], '.3f')}]", f"{fmt(p, '.3f')}{star}"])
    return render(["Strategy", "Versus", "ΔSharpe", "95% CI", "p-value"], rows,
                  "<b>Table 7.</b> Paired circular block-bootstrap tests of annualised Sharpe-ratio differences "
                  "(block length 21 days, 2,000 replications). PPO return series are the equal-weight "
                  "average of the three seeds.", "tab-sig", num_from=2,
                  notes="* p &lt; 0.05, ** p &lt; 0.01 (two-sided, uncorrected for multiple comparisons).")


def table_seeds(res: Path) -> str:
    sm = pd.read_csv(res / "seed_metrics.csv")
    sm = sm[sm["kind"] == "PPO"]
    rows = []
    for _, r in sm.iterrows():
        rows.append([label(r["strategy"]), str(int(r["seed"])), fmt(r["cumulative_return_pct"], ".1f"),
                     fmt(r["sharpe"], ".2f"), fmt(r["sortino"], ".2f"), fmt(r["max_drawdown_pct"], ".1f"),
                     fmt(r["calmar"], ".2f"), fmt(r["cvar_95_pct"], ".2f"), fmt(100 * r["avg_daily_turnover"], ".2f"),
                     fmt(r["worst_day_pct"], ".2f"), fmt(r["skew"], ".2f")])
    return render(["Variant", "Seed", "Cum. ret (%)", "Sharpe", "Sortino", "MaxDD (%)", "Calmar",
                   "CVaR<sub>95</sub> (%)", "Turnover (%/day)", "Worst day (%)", "Skew"], rows,
                  "<b>Table A2.</b> Every PPO training seed (stitched out-of-sample track).", "tab-seeds", num_from=1)


def table_checkpoints(res: Path) -> str:
    s = pd.read_csv(res / "checkpoint_summary.csv")
    order = ["50k", "100k", "150k (final)", "selected"]
    rows = []
    ctrl = s[s["checkpoint"] == "-"].iloc[0]
    rows.append([html.escape(ctrl["strategy"]), "untrained", fmt(ctrl["sharpe_mean"], ".2f"),
                 fmt(ctrl["max_drawdown_pct_mean"], ".1f"), fmt(ctrl["cvar_95_pct_mean"], ".2f"),
                 fmt(100 * ctrl["avg_daily_turnover_mean"], ".1f"), fmt(100 * ctrl["total_cost_pct_mean"], ".0f")])
    first = True
    for name in dict.fromkeys(s.loc[s["checkpoint"] != "-", "strategy"]):
        for ck in order:
            r = s[(s["strategy"] == name) & (s["checkpoint"] == ck)]
            if r.empty:
                continue
            r = r.iloc[0]
            rows.append([label(name) if ck == order[0] else "", "validation-selected" if ck == "selected" else ck,
                         pm(r["sharpe_mean"], r["sharpe_std"], ".2f"),
                         pm(r["max_drawdown_pct_mean"], r["max_drawdown_pct_std"], ".1f"),
                         pm(r["cvar_95_pct_mean"], r["cvar_95_pct_std"], ".2f"),
                         pm(100 * r["avg_daily_turnover_mean"], 100 * r["avg_daily_turnover_std"], ".1f"),
                         pm(100 * r["total_cost_pct_mean"], 100 * r["total_cost_pct_std"], ".0f")])
    return render(["Strategy", "Checkpoint", "Sharpe", "MaxDD (%)", "CVaR<sub>95</sub> (%)", "Turnover (%/day)",
                   "Total cost (bp)"], rows,
                  "<b>Table 8.</b> Selection-free robustness check. Every agent evaluated out of sample at fixed "
                  "training budgets, versus the untrained policy (equal weight on the five assets and cash, "
                  "rebalanced daily). Mean ± std over seeds.", "tab-ckpt", group_break=1, num_from=2)


def table_derisk(res: Path) -> str:
    d = pd.read_csv(res / "derisking.csv")
    g = d.drop(columns="seed").groupby("strategy", sort=False)
    mean, std = g.mean(), g.std(ddof=1)
    full = {n[:22]: n for n in pd.read_csv(res / "experiment_results_full.csv", index_col=0).index}
    rows = []
    for n in mean.index:
        rows.append([label(full.get(n, n))] + [pm(100 * mean.at[n, c], 100 * std.at[n, c], ".1f")
                                                for c in ("cash_calm", "cash_stress")]
                    + [pm(mean.at[n, "corr_cash_trailvol"], std.at[n, "corr_cash_trailvol"], ".2f")])
    return render(["Variant", "Cash weight, calm (%)", "Cash weight, stress (%)", "Corr(cash, trailing 21d vol)"],
                  rows, "<b>Table 9.</b> Is de-risking state-dependent? Average cash weight by ground-truth regime, "
                  "and the correlation between cash weight and the portfolio's trailing realised volatility "
                  "(mean ± std over seeds).", "tab-derisk")


TABLES: dict[str, Callable[[Path], str]] = {
    "main": table_main, "costs": table_costs, "folds": table_folds, "regime": table_regime,
    "sig": table_sig, "seeds": table_seeds,
    "checkpoints": table_checkpoints, "derisk": table_derisk,
}


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="outputs")
    p.add_argument("--template", default="paper/paper_template.html")
    p.add_argument("--out", default="paper/paper.html")
    args = p.parse_args(argv)
    res = Path(args.results)
    text = Path(args.template).read_text(encoding="utf-8")
    for name, fn in TABLES.items():
        text = text.replace(f"<!--TABLE:{name}-->", fn(res))
    missing = re.findall(r"<!--TABLE:(\w+)-->", text)
    if missing:
        print(f"warning: unfilled placeholders {missing}", file=sys.stderr)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
