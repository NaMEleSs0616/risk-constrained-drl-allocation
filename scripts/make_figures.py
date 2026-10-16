#!/usr/bin/env python
"""Publication figures from the artifacts written by ``scripts/run_experiments.py``.

Usage:
    python scripts/make_figures.py --results outputs --out paper/figures
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

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "font.family": "DejaVu Sans",
})

# Okabe-Ito (colour-blind safe) for the PPO variants; greys / muted for baselines.
RL_COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9"]
BASE_COLORS = ["#4D4D4D", "#8C8C8C", "#B07AA1", "#9C755F", "#76B7B2"]


def short(name: str) -> str:
    """Compact display label (``PPO-A Vanilla (l1=0, l2=0)`` -> ``PPO-A Vanilla``)."""
    return name.split(" (")[0] if name.startswith("PPO") else name


def colors(names: Sequence[str], rl: Sequence[str]) -> dict[str, str]:
    out, i, j = {}, 0, 0
    for n in names:
        if n in rl:
            out[n] = RL_COLORS[i % len(RL_COLORS)]
            i += 1
        else:
            out[n] = BASE_COLORS[j % len(BASE_COLORS)]
            j += 1
    return out


def fig_equity(rec: dict, out: Path, col: dict[str, str]) -> None:
    stitched, rl, dates, regimes = rec["stitched"], rec["rl_labels"], rec["dates"], rec["regimes"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.6, 1]})
    for name, by_seed in stitched.items():
        if not by_seed:
            continue
        eq = pd.concat([d["equity"] for d in by_seed.values()], axis=1)
        dd = pd.concat([d["oos_drawdown"] for d in by_seed.values()], axis=1)
        is_rl = name in rl
        kw = {"color": col[name], "lw": 1.6 if is_rl else 1.0, "ls": "-" if is_rl else "--"}
        ax1.plot(eq.index, eq.mean(axis=1), label=short(name), **kw)
        ax2.plot(dd.index, -100 * dd.mean(axis=1), **kw)
        if is_rl and eq.shape[1] > 1:
            ax1.fill_between(eq.index, eq.min(axis=1), eq.max(axis=1), color=col[name], alpha=0.12, lw=0)
    if regimes is not None:
        idx = next(iter(next(iter(stitched.values())).values())).index
        stress = regimes.reindex(idx).fillna(0).to_numpy() == 1
        for ax in (ax1, ax2):
            ylo, yhi = ax.get_ylim()
            ax.fill_between(idx, ylo, yhi, where=stress, color="red", alpha=0.07, lw=0, step="mid")
            ax.set_ylim(ylo, yhi)
    for f in rec["folds"]:
        for ax in (ax1, ax2):
            ax.axvline(dates[f.test_start], color="k", lw=0.6, alpha=0.4, ls=":")
    ax1.set_ylabel("NAV (growth of 1)")
    ax1.legend(ncol=2, frameon=False, loc="upper left")
    ax2.set_ylabel("Drawdown (%)")
    fig.align_ylabels()
    fig.tight_layout()
    fig.savefig(out / "fig_equity.png")
    plt.close(fig)


def fig_ablation(seed_metrics: pd.DataFrame, rl: list[str], out: Path, col: dict[str, str]) -> None:
    panels = [("sharpe", "Sharpe ratio"), ("max_drawdown_pct", "Max drawdown (%)"),
              ("cvar_95_pct", "Daily CVaR$_{95}$ (%)"), ("avg_daily_turnover", "Avg daily turnover")]
    order = list(dict.fromkeys(seed_metrics["strategy"]))
    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.9))
    for ax, (m, title) in zip(axes, panels):
        g = seed_metrics.groupby("strategy", sort=False)[m]
        mean, std = g.mean().reindex(order), g.std(ddof=1).reindex(order).fillna(0.0)
        y = np.arange(len(order))
        ax.barh(y, mean, xerr=std, color=[col[n] for n in order], height=0.7,
                error_kw={"elinewidth": 0.8, "capsize": 2})
        for n in rl:  # individual seeds
            pts = seed_metrics.loc[seed_metrics["strategy"] == n, m]
            ax.scatter(pts, np.full(len(pts), order.index(n)), s=6, color="k", zorder=3)
        ax.set_yticks(y)
        ax.set_yticklabels([short(n) for n in order] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", alpha=0)
    fig.tight_layout()
    fig.savefig(out / "fig_ablation.png")
    plt.close(fig)


def fig_regime(regime: pd.DataFrame, out: Path, col: dict[str, str]) -> None:
    if regime.empty:
        return
    order = list(dict.fromkeys(regime["strategy"]))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    for ax, (m, title) in zip(axes, [("ann_mean_pct", "Annualised mean return (%)"),
                                     ("cvar_95_pct", "Daily CVaR$_{95}$ (%)")]):
        y = np.arange(len(order))
        for k, (reg, hatch, off) in enumerate((("calm", "", -0.2), ("stress", "///", 0.2))):
            vals = regime[regime["regime"] == reg].set_index("strategy")[m].reindex(order)
            ax.barh(y + off, vals, height=0.38, color=[col[n] for n in order], hatch=hatch,
                    edgecolor="white", label=reg)
        ax.axvline(0, color="k", lw=0.6)
        ax.set_yticks(y)
        ax.set_yticklabels([short(n) for n in order] if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.set_title(title)
        ax.grid(axis="y", alpha=0)
    axes[1].legend(["calm", "stress"], frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out / "fig_regime.png")
    plt.close(fig)


def load_training_curves(models_dir: Path, labels: dict[str, str]) -> pd.DataFrame:
    rows = []
    for csv in models_dir.glob("*/seed_*/fold_*/logs/progress.csv"):
        key = csv.parts[-5]
        try:
            d = pd.read_csv(csv)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            continue
        d["variant"] = labels.get(key, key)
        d["seed"] = csv.parts[-4]
        d["fold"] = csv.parts[-3]
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def fig_training(curves: pd.DataFrame, rl: list[str], out: Path, col: dict[str, str]) -> None:
    if curves.empty:
        return
    panels = [("portfolio/cash_weight", "Cash weight"), ("trading/turnover_mean", "Daily turnover"),
              ("risk/rolling_drawdown_mean", "Mean drawdown"), ("episode/sharpe", "In-sample ep. Sharpe")]
    panels = [(c, t) for c, t in panels if c in curves]
    fig, axes = plt.subplots(1, len(panels), figsize=(7.2, 2.5))
    for ax, (c, t) in zip(np.atleast_1d(axes), panels):
        for name in rl:
            d = curves[curves["variant"] == name].dropna(subset=[c])
            if d.empty:
                continue
            step = (d["time/total_timesteps"] // 4096) * 4096
            g = d.groupby(step)[c]
            m, s = g.mean(), g.std().fillna(0)
            ax.plot(m.index / 1e3, m, color=col[name], lw=1.3, label=short(name))
            ax.fill_between(m.index / 1e3, m - s, m + s, color=col[name], alpha=0.12, lw=0)
        ax.set_title(t)
        ax.set_xlabel("Timesteps (k)")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_training.png")
    plt.close(fig)


def fig_allocation(rec: dict, names: Sequence[str], out: Path) -> None:
    stitched = rec["stitched"]
    names = [n for n in names if n in stitched and stitched[n]]
    if not names:
        return
    fig, axes = plt.subplots(len(names), 1, figsize=(7.2, 1.6 * len(names) + 0.4), sharex=True)
    palette = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9", "#D55E00", "#F0E442"]
    for ax, name in zip(np.atleast_1d(axes), names):
        df = stitched[name][min(stitched[name])]
        wcols = [c for c in df.columns if c.startswith("w_")]
        stack = [df[c].clip(lower=0) for c in wcols] + [df["cash_weight"].clip(lower=0)]
        ax.stackplot(df.index, *stack, colors=palette[: len(wcols)] + ["#DDDDDD"],
                     labels=[c[2:] for c in wcols] + ["cash"], lw=0)
        ax.set_ylim(0, 1)
        ax.set_ylabel("Weight")
        ax.set_title(f"{short(name)} (seed {min(stitched[name])})", loc="left")
    np.atleast_1d(axes)[0].legend(ncol=len(stack), frameon=False, loc="upper center",
                                  bbox_to_anchor=(0.5, 1.45), fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "fig_allocation.png")
    plt.close(fig)


def fig_cost(seed_metrics: pd.DataFrame, out: Path, col: dict[str, str], rl: list[str]) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 2.9))
    for name, g in seed_metrics.groupby("strategy", sort=False):
        ax.scatter(g["avg_daily_turnover"] * 100, g["sharpe"], color=col[name], s=22 if name in rl else 34,
                   marker="o" if name in rl else "s", label=short(name), edgecolor="white", lw=0.4)
    ax.set_xlabel("Average daily turnover (% of NAV)")
    ax.set_ylabel("Out-of-sample Sharpe")
    ax.legend(frameon=False, fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(out / "fig_turnover_sharpe.png")
    plt.close(fig)


def fig_checkpoints(res: Path, out: Path, col: dict[str, str], rl: list[str]) -> None:
    path = res / "checkpoint_summary.csv"
    if not path.exists():
        return
    s = pd.read_csv(path)
    ctrl = s[s["checkpoint"] == "-"].iloc[0]
    cks = ["50k", "100k", "150k (final)"]
    x = np.array([50, 100, 150])
    panels = [("sharpe", "Out-of-sample Sharpe"), ("max_drawdown_pct", "Max drawdown (%)"),
              ("avg_daily_turnover", "Daily turnover")]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (m, t) in zip(axes, panels):
        for name in rl:
            d = s[(s["strategy"] == name) & (s["checkpoint"].isin(cks))].set_index("checkpoint").reindex(cks)
            ax.errorbar(x, d[f"{m}_mean"], yerr=d[f"{m}_std"], color=col[name], marker="o", ms=3, lw=1.3,
                        capsize=2, elinewidth=0.7, label=short(name))
        ax.axhline(ctrl[f"{m}_mean"], color="k", ls="--", lw=1, label="Untrained policy")
        ax.set_xticks(x)
        ax.set_xlabel("Training timesteps (k)")
        ax.set_title(t)
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(out / "fig_checkpoints.png")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", default="outputs")
    p.add_argument("--out", default="paper/figures")
    args = p.parse_args(argv)
    res, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (res / "oos_records.pkl").open("rb") as fh:
        rec = pickle.load(fh)
    seed_metrics = pd.read_csv(res / "seed_metrics.csv")
    regime = pd.read_csv(res / "regime_metrics.csv") if (res / "regime_metrics.csv").stat().st_size > 1 else pd.DataFrame()
    rl = rec["rl_labels"]
    names = list(rec["stitched"])
    col = colors(names, rl)

    fig_equity(rec, out, col)
    fig_ablation(seed_metrics, rl, out, col)
    if not regime.empty:
        fig_regime(regime[regime["regime"] != "all"], out, col)
    import yaml

    cfg = yaml.safe_load((res / "config_used.yaml").read_text())
    labels = {k.lower(): v.get("label", k) for k, v in cfg["experiments"]["variants"].items()}
    fig_training(load_training_curves(res / "models", labels), rl, out, col)
    fig_allocation(rec, [rl[0], rl[min(3, len(rl) - 1)], rl[-1]] if rl else [], out)
    fig_cost(seed_metrics, out, col, rl)
    fig_checkpoints(res, out, col, rl)
    print(f"Figures written to {out}: {sorted(p.name for p in out.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
