"""Financial performance metrics and table formatting.

All metrics are computed from a series of *net* (after-cost) simple period returns.
Edge cases are handled explicitly:

* Non-finite returns are replaced by 0 (and counted in ``n_invalid_returns``) so a
  single bad period cannot poison every statistic.
* Ratios whose denominator is (numerically) zero, e.g. the Sharpe ratio of an
  all-cash strategy, return ``NaN`` rather than raising or producing ``inf``; tables
  render these as ``n/a``.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from src.reward import historical_cvar, max_drawdown
from src.utils import get_logger

logger = get_logger(__name__)

_EPS = 1e-12

#: Display names for the core metrics, in presentation order.
METRIC_LABELS: dict[str, str] = {
    "cumulative_return_pct": "Cum. Return (%)",
    "annualized_return_pct": "Ann. Return (%)",
    "annualized_vol_pct": "Ann. Vol (%)",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown_pct": "MaxDD (%)",
    "calmar": "Calmar",
    "avg_daily_turnover": "Avg Daily Turnover",
    "total_cost_pct": "Total Cost (% NAV)",
    "total_cost_usd": "Total Cost ($)",
    "cvar_95_pct": "CVaR95 daily (%)",
}


def safe_divide(numerator: float, denominator: float, default: float = float("nan")) -> float:
    """Divide, returning ``default`` when the denominator is ~0 or inputs are non-finite.

    Args:
        numerator: Dividend.
        denominator: Divisor.
        default: Value returned for an undefined ratio.

    Returns:
        ``numerator / denominator`` or ``default``.
    """
    if not (math.isfinite(numerator) and math.isfinite(denominator)) or abs(denominator) < _EPS:
        return default
    return numerator / denominator


def sanitize_returns(returns: Sequence[float] | np.ndarray | pd.Series) -> tuple[np.ndarray, int]:
    """Replace non-finite returns with 0 and floor returns at -100%.

    Args:
        returns: Period returns.

    Returns:
        Tuple ``(clean_returns, n_invalid)``.
    """
    r = np.asarray(returns, dtype=np.float64).ravel()
    bad = ~np.isfinite(r)
    n_bad = int(bad.sum())
    if n_bad:
        logger.warning("Replacing %d non-finite returns with 0.0", n_bad)
    r = np.where(bad, 0.0, r)
    return np.maximum(r, -1.0), n_bad


def annualized_sharpe(returns: np.ndarray, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio of period returns (NaN if volatility is ~0)."""
    r, _ = sanitize_returns(returns)
    if r.size < 2:
        return float("nan")
    rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf
    return safe_divide(float(excess.mean()), float(excess.std(ddof=1))) * math.sqrt(periods_per_year)


def annualized_sortino(returns: np.ndarray, periods_per_year: int = 252, risk_free_rate: float = 0.0) -> float:
    """Annualised Sortino ratio using downside deviation below the risk-free rate."""
    r, _ = sanitize_returns(returns)
    if r.size < 2:
        return float("nan")
    rf = (1.0 + risk_free_rate) ** (1.0 / periods_per_year) - 1.0
    excess = r - rf
    downside = math.sqrt(float(np.mean(np.minimum(excess, 0.0) ** 2)))
    return safe_divide(float(excess.mean()), downside) * math.sqrt(periods_per_year)


def compute_performance_metrics(
    returns: Sequence[float] | np.ndarray | pd.Series,
    turnover: Sequence[float] | np.ndarray | None = None,
    costs: Sequence[float] | np.ndarray | None = None,
    cost_dollars: Sequence[float] | np.ndarray | None = None,
    periods_per_year: int = 252,
    risk_free_rate: float = 0.0,
) -> dict[str, float]:
    """Compute the standard performance metric set.

    Args:
        returns: Net simple period returns.
        turnover: One-way turnover per period (``sum |dw|``).
        costs: Transaction cost per period as a fraction of NAV.
        cost_dollars: Transaction cost per period in currency units.
        periods_per_year: Annualisation factor.
        risk_free_rate: Annual risk-free rate for Sharpe / Sortino.

    Returns:
        Dictionary of metrics (see :data:`METRIC_LABELS` plus a few extras).
    """
    r, n_bad = sanitize_returns(returns)
    n = int(r.size)
    nan = float("nan")
    out: dict[str, float] = {"n_periods": float(n), "n_invalid_returns": float(n_bad)}
    if n == 0:
        out.update({k: nan for k in METRIC_LABELS})
        return out

    equity = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    cum = float(equity[-1] - 1.0)
    years = n / periods_per_year
    ann_ret = (1.0 + cum) ** (1.0 / years) - 1.0 if cum > -1.0 else -1.0
    ann_vol = float(r.std(ddof=1) * math.sqrt(periods_per_year)) if n > 1 else nan
    mdd = max_drawdown(equity)

    out.update(
        {
            "cumulative_return_pct": 100.0 * cum,
            "annualized_return_pct": 100.0 * ann_ret,
            "annualized_vol_pct": 100.0 * ann_vol if math.isfinite(ann_vol) else nan,
            "sharpe": annualized_sharpe(r, periods_per_year, risk_free_rate),
            "sortino": annualized_sortino(r, periods_per_year, risk_free_rate),
            "max_drawdown_pct": 100.0 * mdd,
            "calmar": safe_divide(ann_ret, mdd),
            "cvar_95_pct": 100.0 * historical_cvar(r, 0.95),
            "hit_rate": float(np.mean(r > 0)),
        }
    )
    if turnover is not None:
        to = np.nan_to_num(np.asarray(turnover, dtype=np.float64), nan=0.0)
        out["avg_daily_turnover"] = float(to.mean()) if to.size else nan
        out["annualized_turnover"] = out["avg_daily_turnover"] * periods_per_year
    else:
        out["avg_daily_turnover"] = nan
    if costs is not None:
        c = np.nan_to_num(np.asarray(costs, dtype=np.float64), nan=0.0)
        out["total_cost_pct"] = 100.0 * float(c.sum())
    else:
        out["total_cost_pct"] = nan
    if cost_dollars is not None:
        out["total_cost_usd"] = float(np.nansum(np.asarray(cost_dollars, dtype=np.float64)))
    else:
        out["total_cost_usd"] = nan
    return out


def metrics_from_records(
    records: pd.DataFrame, periods_per_year: int = 252, risk_free_rate: float = 0.0
) -> dict[str, float]:
    """Compute metrics from a backtest record frame.

    Args:
        records: Frame with a ``return`` column and optional ``turnover``, ``cost``
            (fraction of NAV) and ``cost_usd`` columns.
        periods_per_year: Annualisation factor.
        risk_free_rate: Annual risk-free rate.

    Returns:
        Metric dictionary.
    """
    def col(name: str) -> np.ndarray | None:
        return records[name].to_numpy() if name in records else None

    return compute_performance_metrics(
        records["return"].to_numpy() if "return" in records else np.array([]),
        turnover=col("turnover"),
        costs=col("cost"),
        cost_dollars=col("cost_usd"),
        periods_per_year=periods_per_year,
        risk_free_rate=risk_free_rate,
    )


def _fmt(value: object, floatfmt: str) -> str:
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "n/a"
        return format(float(value), floatfmt)
    return str(value)


def to_markdown_table(
    df: pd.DataFrame,
    floatfmt: str = ",.3f",
    index: bool = True,
    column_formats: Mapping[str, str] | None = None,
) -> str:
    """Render a DataFrame as a GitHub-flavoured Markdown table (no extra dependencies).

    Args:
        df: Table to render.
        floatfmt: Default ``format`` spec for floats.
        index: Include the index as the first column.
        column_formats: Per-column format overrides.

    Returns:
        Markdown string.
    """
    column_formats = dict(column_formats or {})
    headers = ([df.index.name or ""] if index else []) + [str(c) for c in df.columns]
    rows = []
    for idx, row in df.iterrows():
        cells = [str(idx)] if index else []
        cells += [_fmt(row[c], column_formats.get(str(c), floatfmt)) for c in df.columns]
        rows.append(cells)
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    numeric = [False] * len(headers)
    for j, c in enumerate(df.columns):
        numeric[j + (1 if index else 0)] = pd.api.types.is_numeric_dtype(df[c])

    def line(cells: list[str]) -> str:
        padded = [c.rjust(w) if numeric[i] else c.ljust(w) for i, (c, w) in enumerate(zip(cells, widths))]
        return "| " + " | ".join(padded) + " |"

    sep = "|" + "|".join(("-" * (w + 1) + ":") if numeric[i] else ("-" * (w + 2)) for i, w in enumerate(widths)) + "|"
    return "\n".join([line(headers), sep] + [line(r) for r in rows])


def summary_table(results: Mapping[str, Mapping[str, float]], metrics: Sequence[str] | None = None) -> pd.DataFrame:
    """Build a strategy x metric table with display labels.

    Args:
        results: ``{strategy_name: metric_dict}``.
        metrics: Metric keys to include (defaults to :data:`METRIC_LABELS`).

    Returns:
        DataFrame indexed by strategy with human-readable column names.
    """
    keys = list(metrics or METRIC_LABELS.keys())
    df = pd.DataFrame({name: {k: m.get(k, float("nan")) for k in keys} for name, m in results.items()}).T
    df.index.name = "Strategy"
    return df.rename(columns=METRIC_LABELS)


# ---------------------------------------------------------------------------
# Tail, regime-conditional and inferential statistics
# ---------------------------------------------------------------------------
def tail_statistics(returns: Sequence[float] | np.ndarray, horizon: int = 5) -> dict[str, float]:
    """Higher-moment and worst-case statistics of a return series.

    Args:
        returns: Net simple period returns.
        horizon: Horizon (periods) for the worst compounded multi-period return.

    Returns:
        Skewness, excess kurtosis, worst period and worst ``horizon``-period return (in %).
    """
    r, _ = sanitize_returns(returns)
    nan = float("nan")
    if r.size < 3:
        return {"skew": nan, "excess_kurtosis": nan, "worst_day_pct": nan, f"worst_{horizon}d_pct": nan}
    z = (r - r.mean()) / (r.std() + _EPS)
    log_cum = np.concatenate([[0.0], np.cumsum(np.log1p(np.maximum(r, -1.0 + 1e-12)))])
    multi = np.expm1(log_cum[horizon:] - log_cum[:-horizon]) if r.size >= horizon else np.array([nan])
    return {
        "skew": float(np.mean(z**3)),
        "excess_kurtosis": float(np.mean(z**4) - 3.0),
        "worst_day_pct": 100.0 * float(r.min()),
        f"worst_{horizon}d_pct": 100.0 * float(np.nanmin(multi)),
    }


def conditional_metrics(
    returns: Sequence[float] | np.ndarray, mask: Sequence[bool] | np.ndarray, periods_per_year: int = 252
) -> dict[str, float]:
    """Statistics of the returns on the subset of periods selected by ``mask``.

    Used for regime-conditional analysis (e.g. ``mask = regime == stress``).

    Args:
        returns: Net simple period returns.
        mask: Boolean selector of the same length.
        periods_per_year: Annualisation factor.

    Returns:
        Number / share of periods, annualised mean and volatility (%), daily CVaR95 (%),
        worst period (%) and hit rate on the subset.
    """
    r, _ = sanitize_returns(returns)
    m = np.asarray(mask, dtype=bool)
    if m.shape != r.shape:
        raise ValueError("mask and returns must have the same length")
    sub = r[m]
    nan = float("nan")
    if sub.size == 0:
        return {"n": 0.0, "share": 0.0, "ann_mean_pct": nan, "ann_vol_pct": nan,
                "cvar_95_pct": nan, "worst_day_pct": nan, "hit_rate": nan}
    return {
        "n": float(sub.size),
        "share": float(sub.size / max(r.size, 1)),
        "ann_mean_pct": 100.0 * float(sub.mean() * periods_per_year),
        "ann_vol_pct": 100.0 * float(sub.std(ddof=1) * math.sqrt(periods_per_year)) if sub.size > 1 else nan,
        "cvar_95_pct": 100.0 * historical_cvar(sub, 0.95),
        "worst_day_pct": 100.0 * float(sub.min()),
        "hit_rate": float(np.mean(sub > 0)),
    }


def _sharpe(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd) if sd > _EPS else 0.0


def sharpe_difference_bootstrap(
    returns_a: Sequence[float] | np.ndarray,
    returns_b: Sequence[float] | np.ndarray,
    periods_per_year: int = 252,
    block: int = 21,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Paired circular block bootstrap of the annualised Sharpe difference ``SR(a) - SR(b)``.

    Both series are resampled with the *same* blocks, preserving their cross-correlation
    and short-range serial dependence (Politis & Romano, 1992; Ledoit & Wolf, 2008).

    Args:
        returns_a: First return series.
        returns_b: Second return series (same dates).
        periods_per_year: Annualisation factor.
        block: Block length in periods.
        n_boot: Bootstrap replications.
        seed: RNG seed.

    Returns:
        Point estimate, 95% percentile CI and a two-sided bootstrap p-value.
    """
    a, _ = sanitize_returns(returns_a)
    b, _ = sanitize_returns(returns_b)
    if a.shape != b.shape:
        raise ValueError("series must be aligned and of equal length")
    n = a.size
    nan = float("nan")
    if n < 2 * block:
        return {"diff": nan, "ci_low": nan, "ci_high": nan, "p_value": nan}
    ann = math.sqrt(periods_per_year)
    point = (_sharpe(a) - _sharpe(b)) * ann
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(n / block))
    offsets = np.arange(block)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = ((rng.integers(0, n, n_blocks)[:, None] + offsets[None, :]) % n).ravel()[:n]
        diffs[i] = (_sharpe(a[idx]) - _sharpe(b[idx])) * ann
    # Centre the bootstrap distribution under H0: diff = 0.
    centred = diffs - diffs.mean()
    p = float(np.mean(np.abs(centred) >= abs(point)))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"diff": point, "ci_low": float(lo), "ci_high": float(hi), "p_value": p}
