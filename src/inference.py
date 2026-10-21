"""Inference for strategy comparisons that accounts for training-seed variability.

Implements the procedures in ``ANALYSIS_PLAN.md``:

* :func:`two_level_bootstrap` - resample training seeds (with replacement) within each
  strategy, then resample days with circular blocks shared by every series; returns the
  distribution of a metric difference between two strategies.
* :func:`holm` - Holm step-down family-wise adjustment.
* :func:`tost_equivalence` - two one-sided tests via the 90% bootstrap interval.
* :func:`minimum_detectable_effect` - MDE at a given power from the bootstrap SE.

The estimand for a learned strategy is the expected metric of a newly trained agent,
estimated as the mean over seeds of each seed's metric (not the metric of the
seed-averaged portfolio).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from scipy.stats import norm

from src.reward import historical_cvar, max_drawdown

Metric = Callable[[np.ndarray, np.ndarray | None], float]
_EPS = 1e-12


# ---------------------------------------------------------------------------
# Metrics on (returns, turnover) arrays
# ---------------------------------------------------------------------------
def m_sharpe(r: np.ndarray, _to: np.ndarray | None = None, periods: int = 252) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * math.sqrt(periods)) if sd > _EPS else 0.0


def m_ann_return(r: np.ndarray, _to: np.ndarray | None = None, periods: int = 252) -> float:
    g = float(np.prod(1.0 + np.maximum(r, -1.0)))
    return 100.0 * (g ** (periods / len(r)) - 1.0) if g > 0 else -100.0


def m_maxdd(r: np.ndarray, _to: np.ndarray | None = None) -> float:
    return 100.0 * max_drawdown(np.concatenate([[1.0], np.cumprod(1.0 + np.maximum(r, -1.0))]))


def m_cvar(r: np.ndarray, _to: np.ndarray | None = None) -> float:
    return 100.0 * historical_cvar(r, 0.95)


def m_turnover(_r: np.ndarray, to: np.ndarray | None) -> float:
    return 100.0 * float(np.mean(to)) if to is not None else float("nan")


METRICS: dict[str, Metric] = {
    "sharpe": m_sharpe, "ann_return_pct": m_ann_return, "maxdd_pct": m_maxdd,
    "cvar95_pct": m_cvar, "turnover_pct_day": m_turnover,
}


# ---------------------------------------------------------------------------
# Two-level bootstrap
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Comparison:
    """Result of comparing strategy A with strategy B on one metric."""

    metric: str
    estimate: float
    ci_low: float
    ci_high: float
    ci90_low: float
    ci90_high: float
    se: float
    p_value: float
    n_seeds_a: int
    n_seeds_b: int


def _circular_block_indices(n: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = int(math.ceil(n / block))
    starts = rng.integers(0, n, n_blocks)
    return ((starts[:, None] + np.arange(block)[None, :]) % n).ravel()[:n]


def two_level_bootstrap(
    returns_a: Sequence[np.ndarray],
    returns_b: Sequence[np.ndarray],
    metric: str = "sharpe",
    turnover_a: Sequence[np.ndarray] | None = None,
    turnover_b: Sequence[np.ndarray] | None = None,
    n_boot: int = 2000,
    block: int = 21,
    seed: int = 7,
) -> Comparison:
    """Bootstrap the difference ``metric(A) - metric(B)`` over seeds and time.

    Args:
        returns_a: One aligned daily return array per training seed of strategy A.
        returns_b: Same for B (a deterministic baseline is a single array).
        metric: Key of :data:`METRICS`.
        turnover_a: Optional per-seed turnover arrays (needed for the turnover metric).
        turnover_b: Same for B.
        n_boot: Bootstrap replications.
        block: Circular block length in days.
        seed: RNG seed.

    Returns:
        :class:`Comparison` with the point estimate, 95% and 90% percentile intervals,
        bootstrap SE and a two-sided p-value (bootstrap distribution centred under H0).
    """
    fn = METRICS[metric]
    ra = [np.asarray(x, float) for x in returns_a]
    rb = [np.asarray(x, float) for x in returns_b]
    ta = [np.asarray(x, float) for x in turnover_a] if turnover_a is not None else [None] * len(ra)
    tb = [np.asarray(x, float) for x in turnover_b] if turnover_b is not None else [None] * len(rb)
    n = len(ra[0])
    if any(len(x) != n for x in ra + rb):
        raise ValueError("all return series must be aligned and of equal length")

    def stat(idx: np.ndarray | None, sa: Sequence[int], sb: Sequence[int]) -> float:
        def mean_metric(rs, ts, sel):
            vals = []
            for i in sel:
                r = rs[i] if idx is None else rs[i][idx]
                t = ts[i] if (ts[i] is None or idx is None) else ts[i][idx]
                vals.append(fn(r, t))
            return float(np.mean(vals))
        return mean_metric(ra, ta, sa) - mean_metric(rb, tb, sb)

    point = stat(None, range(len(ra)), range(len(rb)))
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot)
    for k in range(n_boot):
        sa = rng.integers(0, len(ra), len(ra))
        sb = rng.integers(0, len(rb), len(rb))
        draws[k] = stat(_circular_block_indices(n, block, rng), sa, sb)
    centred = draws - draws.mean()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    lo90, hi90 = np.percentile(draws, [5.0, 95.0])
    return Comparison(metric, point, float(lo), float(hi), float(lo90), float(hi90),
                      float(draws.std(ddof=1)), float(np.mean(np.abs(centred) >= abs(point))), len(ra), len(rb))


# ---------------------------------------------------------------------------
# Multiplicity, equivalence, power
# ---------------------------------------------------------------------------
def holm(p_values: Sequence[float]) -> list[float]:
    """Holm step-down adjusted p-values (family-wise error control).

    Args:
        p_values: Raw p-values of the family.

    Returns:
        Adjusted p-values in the original order (monotone, capped at 1).
    """
    p = np.asarray(p_values, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj.tolist()


def tost_equivalence(comp: Comparison, margin: float) -> str:
    """Classify a difference using its 90% interval (TOST at alpha = 0.05).

    Returns:
        ``"equivalent"`` if the 90% CI lies inside ``(-margin, margin)``;
        ``"different"`` if the 95% CI excludes 0; otherwise ``"inconclusive"``.
    """
    if -margin < comp.ci90_low and comp.ci90_high < margin:
        return "equivalent"
    if comp.ci_low > 0 or comp.ci_high < 0:
        return "different"
    return "inconclusive"


def minimum_detectable_effect(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest true difference detectable with the given two-sided alpha and power."""
    return float((norm.ppf(1 - alpha / 2) + norm.ppf(power)) * se)
