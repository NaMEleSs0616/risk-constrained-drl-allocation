#!/usr/bin/env python
"""Build the paper: fill table placeholders from experiment outputs and number everything.

Placeholders in the template:

* ``<!--TABLE:name@study-->`` - a results table generated from the study's CSVs
  (``study`` is a key of ``--results``, e.g. ``real=outputs_real syn=outputs``).
* ``<b>Table §.</b>`` / ``<b>Figure §.</b>`` in captions - numbered in document order.
* ``[[tab:id]]`` / ``[[fig:id]]`` in text - replaced by a link "Table n" / "Figure n".

Every number in a generated table is read from the CSVs; nothing is typed by hand.

Usage:
    python scripts/build_paper.py --results real=outputs_real syn=outputs \\
        --template paper/paper_template.html --out paper/paper.html
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

VARIANT_SHORT = {
    "A": "A · return only", "B": "B · CVaR only", "C": "C · drawdown only", "D": "D · CVaR + drawdown",
    "E": "E · calibrated CVaR + DD", "T": "T · turnover penalty", "R": "R · RCPO (adaptive λ)",
    "S": "S · smoothed execution", "SE": "SE · smoothed + calibrated CVaR + DD",
    "RT": "RT · RCPO, 1/N-matched targets", "SRT": "SRT · smoothed + RCPO 1/N-matched",
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def label(name: str) -> str:
    """Display label for a strategy name."""
    m = re.match(r"PPO-([A-Z]+) .*?\((l1=[^,]+), (l2=[^)]+)\)(.*)", name)
    if m:
        letter, l1, l2, rest = m.groups()
        lam = "adaptive λ" if "auto" in l1 else f"λ₁={l1[3:]}, λ₂={l2[3:]}"
        extra = " + vol-target overlay" if "vol target" in rest else ""
        return f"PPO-{VARIANT_SHORT.get(letter, letter)}{extra} <span class='lam'>({lam})</span>"
    return html.escape(name)


def is_rl(name: str) -> bool:
    return name.startswith("PPO-")


def fmt(x: float, spec: str) -> str:
    try:
        xf = float(x)
    except (TypeError, ValueError):
        return "–"
    if not math.isfinite(xf):
        return "–"
    return format(xf, spec).replace("-", "−")


def pm(mean: float, std: float, spec: str) -> str:
    try:
        if not math.isfinite(float(mean)):
            return "–"
    except (TypeError, ValueError):
        return "–"
    if std is None or not math.isfinite(float(std)):
        return fmt(mean, spec)
    return f"{fmt(mean, spec)}<span class='pm'> ± {fmt(std, spec)}</span>"


def stars(p: float) -> str:
    return "**" if p < 0.01 else ("*" if p < 0.05 else "")


def render(headers: Sequence[str], rows: Sequence[Sequence[str]], caption: str, tid: str,
           group_break: int | None = None, num_from: int = 1, notes: str = "") -> str:
    head = "".join(f"<th{' class=num' if i >= num_from else ''}>{h}</th>" for i, h in enumerate(headers))
    body = []
    for r_i, row in enumerate(rows):
        cls = " class='group-start'" if group_break is not None and r_i == group_break else ""
        cells = "".join(f"<td{' class=num' if i >= num_from else ''}>{c}</td>" for i, c in enumerate(row))
        body.append(f"<tr{cls}>{cells}</tr>")
    note = f"<p class='table-note'>{notes}</p>" if notes else ""
    return (f"<figure class='table' id='{tid}'><figcaption><b>Table §.</b> {caption}</figcaption>"
            f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody>"
            f"</table></div>{note}</figure>")


def best_marks(values: dict[str, float], higher: bool) -> set[str]:
    finite = {k: v for k, v in values.items() if math.isfinite(v)}
    if not finite:
        return set()
    target = max(finite.values()) if higher else min(finite.values())
    return {k for k, v in finite.items() if abs(v - target) < 1e-12}


def study_info(res: Path) -> dict[str, float]:
    sm = pd.read_csv(res / "seed_metrics.csv")
    fm = pd.read_csv(res / "fold_metrics.csv")
    rl = sm[sm["kind"] == "PPO"]
    return {"days": int(sm["n_periods"].iloc[0]), "folds": int(fm["fold"].nunique()),
            "seeds": int(rl["seed"].nunique()) if not rl.empty else 0}


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def t_main(res: Path, tid: str) -> str:
    mean = pd.read_csv(res / "experiment_results_full.csv", index_col=0)
    std = pd.read_csv(res / "experiment_results_full_std.csv", index_col=0)
    info = study_info(res)
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
    return render(["Strategy"] + [h for _, h, _, _ in cols], rows,
                  f"Out-of-sample performance on the stitched walk-forward test period ({info['folds']} folds, "
                  f"{info['days']:,} trading days, net of all frictions). PPO rows: mean ± standard deviation "
                  f"across {info['seeds']} training seeds. Bold marks the best value in each column.", tid,
                  group_break=sum(is_rl(n) for n in mean.index),
                  notes="CVaR<sub>95</sub> is the daily historical expected shortfall of net returns. Annualised "
                        "return is geometric; Sharpe and Sortino use a zero risk-free rate.")


def t_costs(res: Path, tid: str) -> str:
    mean = pd.read_csv(res / "experiment_results_full.csv", index_col=0)
    std = pd.read_csv(res / "experiment_results_full_std.csv", index_col=0)
    beh = pd.read_csv(res / "behaviour.csv", index_col=0)
    info = study_info(res)
    rows = []
    for name in mean.index:
        s = (lambda c: std.at[name, c]) if is_rl(name) else (lambda c: float("nan"))
        rows.append([
            label(name),
            pm(100 * mean.at[name, "avg_daily_turnover"], 100 * s("avg_daily_turnover"), ".1f"),
            pm(mean.at[name, "annualized_turnover"], s("annualized_turnover"), ".0f"),
            pm(mean.at[name, "total_cost_pct"], s("total_cost_pct"), ".1f"),
            pm(mean.at[name, "total_cost_pct"] / (info["days"] / 252), s("total_cost_pct") / (info["days"] / 252), ".2f"),
            fmt(beh.at[name, "avg_gross_exposure"] * 100, ".1f"),
            fmt(beh.at[name, "avg_effective_n"], ".2f"),
            fmt(beh.at[name, "max_single_weight"] * 100, ".1f"),
        ])
    return render(["Strategy", "Turnover (%/day)", "Turnover (×/yr)", "Total cost (% NAV)", "Cost drag (%/yr)",
                   "Avg gross exposure (%)", "Effective N", "Max weight (%)"], rows,
                  f"Trading activity, transaction costs and allocation behaviour over the {info['days']:,}-day "
                  "out-of-sample period. Each fold starts from cash with a $10m NAV.", tid,
                  group_break=sum(is_rl(n) for n in mean.index),
                  notes="Costs sum commission, half-spread, temporary and permanent impact, each period expressed "
                        "as a fraction of pre-trade NAV. Effective N = (Σ|w|)² / Σw².")


def t_gross(res: Path, tid: str) -> str:
    g = pd.read_csv(res / "gross_and_tracking.csv")
    rows = []
    for _, r in g.iterrows():
        test = ("–" if not math.isfinite(r["gross_diff_vs_ew"]) else
                f"{fmt(r['gross_diff_vs_ew'], '+.3f')} [{fmt(r['gross_ci_low'], '.3f')}, {fmt(r['gross_ci_high'], '.3f')}]"
                f" p={fmt(r['gross_p_value'], '.3f')}{stars(r['gross_p_value'])}")
        rows.append([label(r["strategy"]), fmt(r["net_sharpe"], ".2f"),
                     pm(r["gross_sharpe"], r["gross_sharpe_std"], ".2f"), test,
                     fmt(r["corr_ew"], ".2f"), fmt(r["beta_ew"], ".2f"), fmt(r["te_vs_ew_pct"], ".1f")])
    return render(["Strategy", "Net Sharpe", "Gross Sharpe", "Gross ΔSharpe vs 1/N [95% CI]",
                   "Corr. with 1/N", "Beta to 1/N", "Tracking error vs 1/N (%)"], rows,
                  "Performance before transaction costs, and co-movement with the 1/N portfolio. Gross returns add "
                  "each period's cost back to the net return. Tests are paired block bootstraps on seed-averaged "
                  "gross returns.", tid, group_break=int(g["strategy"].str.startswith("PPO-").sum()),
                  notes="* p &lt; 0.05, ** p &lt; 0.01, uncorrected.")


def t_folds(res: Path, tid: str) -> str:
    fm = pd.read_csv(res / "fold_metrics.csv")
    order = list(dict.fromkeys(fm["strategy"]))
    folds = sorted(fm["fold"].unique())
    rows = []
    for name in order:
        g = fm[fm["strategy"] == name]
        rows.append([label(name)] + [fmt(g[g["fold"] == f]["sharpe"].mean(), ".2f") for f in folds])
    return render(["Strategy"] + [f"F{f}" for f in folds], rows,
                  "Out-of-sample Sharpe ratio in each walk-forward fold (PPO: mean over seeds).", tid,
                  group_break=sum(is_rl(n) for n in order))


def t_regime(res: Path, tid: str) -> str:
    rm = pd.read_csv(res / "regime_metrics.csv")
    order = list(dict.fromkeys(rm["strategy"]))
    rows = []
    for name in order:
        g = rm[rm["strategy"] == name].set_index("regime")
        rows.append([label(name)] + [fmt(g.at[r, c], ".1f" if c == "ann_mean_pct" else ".2f")
                                     for r in ("calm", "stress")
                                     for c in ("ann_mean_pct", "ann_vol_pct", "cvar_95_pct", "worst_day_pct")])
    n_calm = int(rm[rm["regime"] == "calm"]["n"].iloc[0])
    n_stress = int(rm[rm["regime"] == "stress"]["n"].iloc[0])
    headers = ["Strategy", "Calm: ann. mean (%)", "Calm: ann. vol (%)", "Calm: CVaR<sub>95</sub> (%)",
               "Calm: worst day (%)", "Stress: ann. mean (%)", "Stress: ann. vol (%)",
               "Stress: CVaR<sub>95</sub> (%)", "Stress: worst day (%)"]
    return render(headers, rows, f"Regime-conditional statistics of net daily returns ({n_calm:,} calm and "
                  f"{n_stress:,} stress days out of sample). PPO rows average the per-seed statistics.", tid,
                  group_break=sum(is_rl(n) for n in order),
                  notes="Regime labels are never observed by any strategy; they are used only for this analysis.")


def t_sig(res: Path, tid: str) -> str:
    sig = pd.read_csv(res / "significance.csv")
    rows = [[label(r["strategy"]), label(r["versus"]), fmt(r["diff"], "+.3f"),
             f"[{fmt(r['ci_low'], '.3f')}, {fmt(r['ci_high'], '.3f')}]",
             f"{fmt(r['p_value'], '.3f')}{stars(r['p_value'])}"] for _, r in sig.iterrows()]
    return render(["Strategy", "Versus", "ΔSharpe", "95% CI", "p-value"], rows,
                  "Paired circular block-bootstrap tests of annualised Sharpe-ratio differences (block length 21 "
                  "days, 2,000 replications) on net returns; PPO series are the equal-weight average of seeds.",
                  tid, num_from=2, notes="* p &lt; 0.05, ** p &lt; 0.01 (two-sided, uncorrected).")


def t_seeds(res: Path, tid: str) -> str:
    sm = pd.read_csv(res / "seed_metrics.csv")
    sm = sm[sm["kind"] == "PPO"]
    rows = [[label(r["strategy"]), str(int(r["seed"])), fmt(r["cumulative_return_pct"], ".1f"), fmt(r["sharpe"], ".2f"),
             fmt(r["sortino"], ".2f"), fmt(r["max_drawdown_pct"], ".1f"), fmt(r["calmar"], ".2f"),
             fmt(r["cvar_95_pct"], ".2f"), fmt(100 * r["avg_daily_turnover"], ".1f"), fmt(r["worst_day_pct"], ".2f")]
            for _, r in sm.iterrows()]
    return render(["Variant", "Seed", "Cum. ret (%)", "Sharpe", "Sortino", "MaxDD (%)", "Calmar",
                   "CVaR<sub>95</sub> (%)", "Turnover (%/day)", "Worst day (%)"], rows,
                  "Every PPO training seed (stitched out-of-sample track).", tid)


def t_checkpoints(res: Path, tid: str) -> str:
    s = pd.read_csv(res / "checkpoint_summary.csv")
    order = ["50k", "100k", "150k (final)", "selected"]
    ctrl = s[s["checkpoint"] == "-"].iloc[0]
    rows = [[html.escape(ctrl["strategy"]), "untrained", fmt(ctrl["sharpe_mean"], ".2f"),
             fmt(ctrl["max_drawdown_pct_mean"], ".1f"), fmt(ctrl["cvar_95_pct_mean"], ".2f"),
             fmt(100 * ctrl["avg_daily_turnover_mean"], ".1f"), fmt(100 * ctrl["total_cost_pct_mean"], ".0f")]]
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
                  "Selection-free robustness check (controlled study). Every agent evaluated out of sample at fixed "
                  "training budgets, versus the untrained policy. Mean ± std over seeds.", tid,
                  group_break=1, num_from=2)


def t_derisk(res: Path, tid: str) -> str:
    d = pd.read_csv(res / "derisking.csv")
    g = d.drop(columns="seed").groupby("strategy", sort=False)
    mean, std = g.mean(), g.std(ddof=1)
    rows = [[label(n)] + [pm(100 * mean.at[n, c], 100 * std.at[n, c], ".1f") for c in ("cash_calm", "cash_stress")]
            + [pm(mean.at[n, "corr_cash_trailvol"], std.at[n, "corr_cash_trailvol"], ".2f")] for n in mean.index]
    return render(["Variant", "Cash weight, calm (%)", "Cash weight, stress (%)", "Corr(cash, trailing 21d vol)"],
                  rows, "Is de-risking state-dependent? Average cash weight by regime, and the correlation between "
                  "cash weight and the portfolio's trailing realised volatility (mean ± std over seeds).", tid)


def t_lagrangian(res: Path, tid: str) -> str:
    path = res / "lagrangian_path.csv"
    if not path.exists():
        return ""
    lp = pd.read_csv(path)
    folds = sorted(lp["fold"].unique())
    rows = []
    for variant, g in lp.groupby("variant", sort=True):
        m = g.groupby("fold")[["lambda_cvar", "lambda_drawdown"]].mean()
        name = {"r_rcpo": "R", "rt_rcpo_ew": "RT", "srt_smoothed_rcpo_ew": "SRT"}.get(variant, variant)
        rows.append([f"{name}: λ₁ (CVaR)"] + [fmt(m.at[f, "lambda_cvar"], ".2f") if f in m.index else "–" for f in folds])
        rows.append([f"{name}: λ₂ (drawdown)"] + [fmt(m.at[f, "lambda_drawdown"], ".2f") if f in m.index else "–"
                                                for f in folds])
    return render(["Multiplier (mean over seeds)"] + [f"F{int(str(f).split('_')[-1])}" for f in folds], rows,
                  "RCPO multipliers at the end of each fold's training.", tid)

def t_timing(res: Path, tid: str) -> str:
    t = pd.read_csv(res / "timing_vs_tilt.csv")
    rows = []
    for _, r in t.iterrows():
        rows.append([label(r["strategy"]), fmt(r["static_tilt_gross_sharpe"], ".2f"),
                     fmt(r.get("dynamic_gross_sharpe"), ".2f"), fmt(r.get("timing_contribution"), "+.2f"),
                     fmt(100 * r["cash"], ".1f") if "cash" in r and pd.notna(r.get("cash")) else "–"])
    return render(["Strategy", "Static tilt (gross Sharpe)", "Actual policy (gross Sharpe)", "Timing contribution",
                   "Avg cash (%)"], rows,
                  "Timing versus static tilts, before costs. The static-tilt portfolio holds each variant's average "
                  "out-of-sample weights constantly (rebalanced daily, no costs); the difference to the actual "
                  "policy's gross Sharpe is the contribution of day-to-day re-allocation.", tid)


def t_compare(res: Path, tid: str) -> str:
    c = pd.read_csv(res / "followup_vs_original.csv")
    rows = [[label(r["strategy"]), label(r["versus"]),
             f"{fmt(r['net_diff'], '+.3f')} [{fmt(r['net_ci_low'], '.3f')}, {fmt(r['net_ci_high'], '.3f')}]",
             f"{fmt(r['net_p'], '.3f')}{stars(r['net_p'])}",
             f"{fmt(r['gross_diff'], '+.3f')} (p={fmt(r['gross_p'], '.3f')}{stars(r['gross_p'])})",
             f"{fmt(100 * r['turnover_a'], '.1f')} vs {fmt(100 * r['turnover_b'], '.1f')}",
             f"{fmt(r['maxdd_a'], '.1f')} vs {fmt(r['maxdd_b'], '.1f')}"] for _, r in c.iterrows()]
    return render(["Follow-up variant", "Versus", "Net ΔSharpe [95% CI]", "p (net)", "Gross ΔSharpe",
                   "Turnover %/day (new vs old)", "MaxDD % (new vs old)"], rows,
                  "Follow-up variants against their counterparts in the original ETF run (same data, folds, seeds "
                  "and test dates). Paired block-bootstrap tests on seed-averaged daily returns; MaxDD here is "
                  "computed on the seed-averaged return series.", tid, num_from=2,
                  notes="* p &lt; 0.05, ** p &lt; 0.01, uncorrected.")


def t_delev(res: Path, tid: str) -> str:
    d = pd.read_csv(res / "delevered_ew_control.csv")
    rows = []
    for _, r in d.iterrows():
        rows.append([label(r["strategy"]), fmt(100 * r["avg_cash"], ".1f"),
                     f"{fmt(r['sharpe'], '.2f')} vs {fmt(r['sharpe_delev'], '.2f')} (p={fmt(r['p_value'], '.2f')})",
                     f"{fmt(r['maxdd'], '.1f')} vs {fmt(r['maxdd_delev'], '.1f')}",
                     f"{fmt(r.get('stress_cvar'), '.2f')} vs {fmt(r.get('stress_cvar_delev'), '.2f')}",
                     f"{fmt(r.get('stress_worst'), '.2f')} vs {fmt(r.get('stress_worst_delev'), '.2f')}"])
    return render(["Variant", "Avg cash (%)", "Sharpe: agent vs control", "MaxDD %: agent vs control",
                   "Stress CVaR<sub>95</sub> %: agent vs control", "Worst stress day %: agent vs control"], rows,
                  "Exposure-matched control. Each agent (seed-averaged returns) against 1/N scaled to the agent's "
                  "average invested fraction, with the remainder in zero-yield cash.", tid)


TABLES: dict[str, Callable[[Path, str], str]] = {
    "compare": t_compare, "delev": t_delev,
    "timing": t_timing,
    "main": t_main, "costs": t_costs, "gross": t_gross, "folds": t_folds, "regime": t_regime, "sig": t_sig,
    "seeds": t_seeds, "checkpoints": t_checkpoints, "derisk": t_derisk, "lagrangian": t_lagrangian,
}


# ---------------------------------------------------------------------------
# Numbering
# ---------------------------------------------------------------------------
def number(text: str) -> str:
    """Number table / figure captions in document order and resolve [[tab:id]] / [[fig:id]]."""
    pat = re.compile(r"<figure class=['\"](table|fig)['\"][^>]*?id=['\"]([^'\"]+)['\"]")
    counters = {"table": 0, "fig": 0}
    ids: dict[str, str] = {}
    pieces, last = [], 0
    for m in pat.finditer(text):
        kind, fid = m.group(1), m.group(2)
        counters[kind] += 1
        word = "Table" if kind == "table" else "Figure"
        ids[fid] = f"{word} {counters[kind]}"
        token = f"<b>{word} §.</b>"
        pos = text.find(token, m.end())
        pieces.append(text[last:m.end()])
        if pos != -1:
            pieces.append(text[m.end():pos] + f"<b>{word} {counters[kind]}.</b>")
            last = pos + len(token)
        else:
            last = m.end()
    pieces.append(text[last:])
    text = "".join(pieces)

    def ref(m: re.Match) -> str:
        fid = m.group(2)
        if fid not in ids:
            print(f"warning: unresolved reference {m.group(0)}", file=sys.stderr)
            return m.group(0)
        return f"<a href='#{fid}'>{ids[fid]}</a>"

    return re.sub(r"\[\[(tab|fig):([\w-]+)\]\]", ref, text)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", nargs="+", default=["real=outputs_real", "syn=outputs"])
    p.add_argument("--template", default="paper/paper_template.html")
    p.add_argument("--out", default="paper/paper.html")
    args = p.parse_args(argv)
    studies = dict(kv.split("=", 1) for kv in args.results)
    text = Path(args.template).read_text(encoding="utf-8")

    def fill(m: re.Match) -> str:
        name, study = m.group(1), m.group(2)
        return TABLES[name](Path(studies[study]), f"tab-{name}-{study}")

    text = re.sub(r"<!--TABLE:(\w+)@(\w+)-->", fill, text)
    text = number(text)
    leftover = re.findall(r"<!--TABLE:[^>]*-->|<b>(?:Table|Figure) §\.</b>|\[\[(?:tab|fig):[^\]]+\]\]", text)
    if leftover:
        print(f"warning: unfilled placeholders: {leftover[:5]}", file=sys.stderr)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
