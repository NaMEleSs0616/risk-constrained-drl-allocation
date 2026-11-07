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
    "E": "E · calibrated CVaR + DD", "T": "T · turnover penalty", "R": "R · RCPO, adaptive λ",
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
        extra = " + vol-target overlay" if "vol target" in rest else ""
        if "auto" in l1:  # RCPO variants: the short name already says "adaptive"/"RCPO"
            return f"PPO-{VARIANT_SHORT.get(letter, letter)}{extra}"
        lam = f"λ₁={l1[3:]}, λ₂={l2[3:]}"
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


# ---------------------------------------------------------------------------
# Corrected (next-open) study: pre-specified inference and robustness
# ---------------------------------------------------------------------------
def short_label(name: str) -> str:
    """Compact strategy label without the λ annotation."""
    m = re.match(r"PPO-([A-Z]+) ", name)
    if m:
        return f"PPO-{m.group(1)}"
    return html.escape({"Equal Weight (1/N)": "1/N", "Untrained policy (1/(N+1) incl. cash)": "Untrained policy",
                        "Risk Parity (ERC)": "ERC"}.get(name, name))


def ci(est: float, lo: float, hi: float, spec: str = "+.3f") -> str:
    return f"{fmt(est, spec)} [{fmt(lo, spec.lstrip('+'))}, {fmt(hi, spec.lstrip('+'))}]"


def pval(p: float) -> str:
    return "&lt;0.001" if p < 0.001 else fmt(p, ".3f")


def t_primary(res: Path, tid: str) -> str:
    d = pd.read_csv(res / "inference_primary.csv")
    rows = []
    for _, r in d.iterrows():
        comp = f"{short_label(r['A'])} − {short_label(r['B'])}" + (" (gross of costs)" if r["gross"] else "")
        verdict = ("significant" if r["significant_holm_0.05"] else "not significant") + f"; {r['equivalence']}"
        rows.append([r["id"], comp, ci(r["estimate"], r["ci_low"], r["ci_high"]),
                     f"[{fmt(r['ci90_low'], '.3f')}, {fmt(r['ci90_high'], '.3f')}]", fmt(r["mde80"], ".2f"),
                     pval(r["p_value"]), pval(r["p_holm"]), verdict])
    return render(["ID", "Comparison (Sharpe)", "Estimate [95% CI]", "90% CI", "MDE<sub>80</sub>", "p", "p (Holm)",
                   "Verdict"], rows,
                  "Pre-specified primary comparisons (ANALYSIS_PLAN.md, committed before the rerun's results were "
                  "seen). Two-level bootstrap: training seeds resampled within each strategy, then days resampled in "
                  "circular 21-day blocks shared by all series (2,000 replications). The estimand is the expected "
                  "Sharpe ratio of a newly trained agent (mean of per-seed Sharpe ratios).", tid, num_from=2,
                  notes="Holm adjustment over the five primary tests (family-wise α = 0.05). Equivalence: TOST with "
                        "margin ±0.15 Sharpe, i.e. the 90% CI inside (−0.15, 0.15). “Inconclusive” means neither a "
                        "significant difference nor equivalence. MDE<sub>80</sub>: smallest true difference detectable "
                        "with 80% power, 2.80 × bootstrap SE.")


def t_secondary(res: Path, tid: str) -> str:
    d = pd.read_csv(res / "inference_secondary.csv")
    pairs = [("PPO-E", "PPO-A"), ("PPO-S", "PPO-A"), ("PPO-SE", "PPO-E"), ("PPO-S", "Equal Weight (1/N)"),
             ("PPO-SE", "Equal Weight (1/N)"), ("PPO-SE", "Risk Parity (ERC)"),
             ("PPO-SE", "Untrained policy (1/(N+1) incl. cash)")]
    metrics = [("ann_return_pct", "Ann. return (pp)", "+.2f"), ("maxdd_pct", "MaxDD (pp)", "+.1f"),
               ("cvar95_pct", "CVaR<sub>95</sub> (pp)", "+.2f"), ("turnover_pct_day", "Turnover (pp/day)", "+.1f")]
    rows = []
    for a, b in pairs:
        sub = d[d["A"].str.startswith(a + " ") & (d["B"].str.startswith(b + " ") | (d["B"] == b))]
        if sub.empty:
            continue
        row = [f"{short_label(sub['A'].iloc[0])} − {short_label(sub['B'].iloc[0])}"]
        for m, _, spec in metrics:
            r = sub[sub["metric"] == m].iloc[0]
            row.append(f"{ci(r['estimate'], r['ci_low'], r['ci_high'], spec)}{stars(r['p_value'])}")
        rows.append(row)
    return render(["Comparison"] + [h for _, h, _ in metrics], rows,
                  "Secondary comparisons (not used for claims; uncorrected): differences in other metrics with 95% "
                  "two-level bootstrap intervals.", tid,
                  notes="* p &lt; 0.05, ** p &lt; 0.01, uncorrected. Negative MaxDD / CVaR differences mean the first "
                        "strategy has less tail risk. The bootstrap distribution of a maximum-drawdown difference is "
                        "strongly skewed, so its percentile interval and its centred p-value can disagree; both are "
                        "reported as computed.")


def _legacy_dirs(res: Path) -> list[Path]:
    root = res.resolve().parent
    return [root / "outputs_real", root / "outputs_real_followup"]


def t_legacy(res: Path, tid: str) -> str:
    new = pd.read_csv(res / "experiment_results_full.csv", index_col=0)
    gnew = pd.read_csv(res / "gross_and_tracking.csv", index_col=0)
    old, gold = [], []
    for d in _legacy_dirs(res):
        old.append(pd.read_csv(d / "experiment_results_full.csv", index_col=0))
        gold.append(pd.read_csv(d / "gross_and_tracking.csv", index_col=0))
    old = pd.concat(old)
    old = old[~old.index.duplicated()]
    gold = pd.concat(gold)
    gold = gold[~gold.index.duplicated()]
    rows = []
    for name in new.index:
        if name not in old.index:
            continue
        rows.append([label(name), fmt(old.at[name, "sharpe"], ".2f"), fmt(new.at[name, "sharpe"], ".2f"),
                     fmt(new.at[name, "sharpe"] - old.at[name, "sharpe"], "+.2f"),
                     fmt(gold.at[name, "gross_sharpe"], ".2f"), fmt(gnew.at[name, "gross_sharpe"], ".2f"),
                     fmt(100 * old.at[name, "avg_daily_turnover"], ".1f"),
                     fmt(100 * new.at[name, "avg_daily_turnover"], ".1f")])
    return render(["Strategy", "Net Sharpe, same-close", "Net Sharpe, next-open", "Change",
                   "Gross Sharpe, same-close", "Gross Sharpe, next-open", "Turnover %/day, same-close",
                   "Turnover %/day, next-open"], rows,
                  "The execution-timing correction. Same data, folds, test dates, features, frictions and PPO "
                  "settings; only the fill convention differs. Same-close (v1–v5, not tradable): decide and fill at the "
                  "close of t. Next-open (this version): decide after the close of t, fill at the open of t+1. "
                  "Same-close PPO rows are from 3 seeds, next-open rows from 5 seeds (new training runs).", tid,
                  group_break=sum(is_rl(n) for n in new.index if n in old.index))


SCEN_LABELS = {"base": "Base", "cost_x0": "No costs", "cost_x0.5": "Costs ×0.5", "cost_x2": "Costs ×2",
               "cost_x3": "Costs ×3", "spread_only": "No impact", "impact_x3": "Impact ×3",
               "exec_next_close": "Fill next close", "exec_close": "Same close*", "init_1N": "Start from 1/N"}


def t_sens(res: Path, tid: str) -> str:
    if not (res / "sensitivity_levels.csv").exists():
        return ""
    lv = pd.read_csv(res / "sensitivity_levels.csv")
    scen = [s for s in SCEN_LABELS if s in set(lv["scenario"])]
    order = list(dict.fromkeys(lv["strategy"]))
    order = [n for n in order if is_rl(n)] + [n for n in order if not is_rl(n)]
    rows = []
    for n in order:
        g = lv[lv["strategy"] == n].set_index("scenario")
        rows.append([label(n)] + [fmt(g.at[s, "sharpe"], ".2f") for s in scen])
    return render(["Strategy"] + [SCEN_LABELS[s] for s in scen], rows,
                  "Evaluation-only sensitivity: net Sharpe ratio of the same trained models (and of every baseline) "
                  "re-run on the out-of-sample windows under perturbed execution assumptions. PPO: mean over 5 seeds.",
                  tid, group_break=sum(is_rl(n) for n in order),
                  notes="Costs ×k scales commission, half-spread and both impact coefficients; “No costs” also lifts "
                        "the participation cap. “No impact” keeps fees and spread only. “Fill next close” adds a "
                        "further session of delay. *“Same close” is the non-tradable legacy convention, shown only "
                        "to measure how much it flatters the same policies. “Start from 1/N”: every fold inherits "
                        "a 1/N book instead of starting in cash. Policies were trained under the base assumptions.")


def t_sens_comp(res: Path, tid: str) -> str:
    if not (res / "sensitivity_comparisons.csv").exists():
        return ""
    c = pd.read_csv(res / "sensitivity_comparisons.csv")
    scen = [s for s in SCEN_LABELS if s in set(c["scenario"])]
    ids = list(dict.fromkeys(c["id"]))
    rows = []
    for s in scen:
        row = [SCEN_LABELS[s]]
        for i in ids:
            r = c[(c["scenario"] == s) & (c["id"] == i)].iloc[0]
            row.append(f"{ci(r['estimate'], r['ci_low'], r['ci_high'], '+.2f')}{stars(r['p_value'])}")
        rows.append(row)
    names = {}
    for i in ids:
        r = c[c["id"] == i].iloc[0]
        names[i] = f"{i}: {short_label(r['A'])} − {short_label(r['B'])}"
    return render(["Scenario"] + [names[i] for i in ids], rows,
                  "Do the comparisons survive the perturbations? Sharpe differences with 95% two-level bootstrap "
                  "intervals (1,000 replications) under each scenario of the previous table.", tid,
                  notes="* p &lt; 0.05, ** p &lt; 0.01, uncorrected (these are robustness checks, not new primary "
                        "tests).")


def t_warm(res: Path, tid: str) -> str:
    path = res / "warmstart_control" / "warmstart_vs_scratch_summary.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    names = {"sharpe": ("Sharpe", ".2f", "+.2f"), "ann_return_pct": ("Ann. return (%)", ".2f", "+.2f"),
             "maxdd_pct": ("MaxDD (%)", ".1f", "+.1f"), "turnover_pct_day": ("Turnover (%/day)", ".1f", "+.1f")}
    rows = []
    for _, r in d.iterrows():
        nm, spec, dspec = names[r["metric"]]
        rows.append([label(r["variant"]), nm, fmt(r["warm_mean"], spec), fmt(r["scratch_mean"], spec),
                     ci(r["diff_mean"], r["ci_lo"], r["ci_hi"], dspec), f"{int(r['n_scratch_better'])}/{int(r['n_pairs'])}"])
    return render(["Variant", "Metric (per fold)", "Warm-started chain", "From scratch", "Scratch − warm [95% CI]",
                   "Pairs where scratch is higher"], rows,
                  "Warm-start control. Selected folds retrained from scratch with the fold-0 budget on exactly the same "
                  "rolling training window and PPO seed, compared fold by fold with the warm-started chain. Intervals "
                  "bootstrap the (seed, fold) pairs.", tid, num_from=2)


def t_clip(res: Path, tid: str) -> str:
    path = res / "reward_clipping.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    g = d.groupby(["variant", "window"], sort=False).agg(n=("n", "sum"), k=("n_clipped", "sum"),
                                                         lo=("min_raw", "min"), hi=("max_raw", "max")).reset_index()
    rows = []
    for v in dict.fromkeys(g["variant"]):
        r = {w: g[(g["variant"] == v) & (g["window"] == w)].iloc[0] for w in ("train_replay", "oos")}
        rows.append([label(v), f"{int(r['train_replay']['k'])} / {int(r['train_replay']['n']):,}",
                     f"[{fmt(r['train_replay']['lo'], '.1f')}, {fmt(r['train_replay']['hi'], '.1f')}]",
                     f"{int(r['oos']['k'])} / {int(r['oos']['n']):,}"])
    return render(["Variant", "Clipped steps, training-window replay", "Range of unclipped reward (training replay)",
                   "Clipped steps, out of sample"], rows,
                  "How often the reward clip (±10 after scaling by 100) binds. Training-window replay: each fold's final "
                  "policy run deterministically over its own training window in consecutive 252-day episodes "
                  "(the training episode length), all seeds.", tid)


def t_synrob(res: Path, tid: str) -> str:
    path = res / "synthetic_markets_summary.csv"
    if not path.exists():
        return ""
    d = pd.read_csv(path)
    rows = []
    for _, r in d.iterrows():
        rows.append([html.escape(r["market"]), fmt(r["ew_sharpe"], ".2f"), fmt(r["untrained_sharpe"], ".2f"),
                     pm(r["s_sharpe"], r["s_sharpe_sd"], ".2f"), pm(r["se_sharpe"], r["se_sharpe_sd"], ".2f"),
                     ci(r["s_minus_ew"], r["s_minus_ew_lo"], r["s_minus_ew_hi"], "+.2f"),
                     ci(r["se_minus_s"], r["se_minus_s_lo"], r["se_minus_s_hi"], "+.2f"),
                     f"{fmt(r['ew_maxdd'], '.1f')} / {fmt(r['s_maxdd'], '.1f')} / {fmt(r['se_maxdd'], '.1f')}",
                     f"{fmt(100 * r['s_cash_calm'], '.0f')} → {fmt(100 * r['s_cash_stress'], '.0f')}; "
                     f"{fmt(100 * r['se_cash_calm'], '.0f')} → {fmt(100 * r['se_cash_stress'], '.0f')}"])
    return render(["Market", "1/N Sharpe", "Untrained Sharpe", "PPO-S Sharpe", "PPO-SE Sharpe", "S − 1/N [95% CI]",
                   "SE − S [95% CI]", "MaxDD %: 1/N / S / SE", "Cash % calm → stress: S; SE"], rows,
                  "Independent simulated markets under next-open execution (4 annual walk-forward folds each, warm-"
                  "started chains as in the ETF study, 2 training seeds per variant). Intervals are two-level "
                  "bootstraps (2,000 replications).", tid, num_from=1,
                  notes="Standard markets use the generator of §4.2 with new data seeds. The bear market has negative "
                        "calm-regime drift (−8% to 0% a year) and stress twice as often, so holding cash pays.")


TABLES: dict[str, Callable[[Path, str], str]] = {
    "primary": t_primary, "secondary": t_secondary, "legacy": t_legacy, "sens": t_sens,
    "senscomp": t_sens_comp, "warm": t_warm, "clip": t_clip, "synrob": t_synrob,
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
