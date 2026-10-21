r"""Classical allocation baselines run through the same friction-aware environment.

Every strategy is a callable ``strategy(env) -> target_weights`` that only reads
``env.price_history()`` (prices up to and including the current period), so it is
subject to exactly the same information set, execution model and costs as the RL
agents.

* :class:`EqualWeightStrategy` - ``1/N``, either periodically rebalanced or bought once
  and held (buy-and-hold).
* :class:`MeanVarianceStrategy` - Markowitz ``max mu'w - (gamma/2) w' Sigma w`` with a
  Ledoit-Wolf shrinkage covariance and James-Stein-style mean shrinkage.
* :class:`RiskParityStrategy` - Equal Risk Contribution via the convex formulation of
  Spinu (2013): ``min 1/2 y' Sigma y - (1/N) sum log y_i``, ``w = y / sum(y)``.
* :class:`HierarchicalRiskParityStrategy` - Lopez de Prado (2016): single-linkage
  clustering on correlation distance, quasi-diagonalisation and recursive bisection
  with inverse-variance cluster allocation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.optimize import minimize
from scipy.spatial.distance import squareform
from sklearn.covariance import LedoitWolf

from src.utils import get_logger

if TYPE_CHECKING:
    from src.environment import PortfolioEnv

logger = get_logger(__name__)


def ledoit_wolf_covariance(returns: np.ndarray) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance of a ``(T, N)`` return matrix.

    Falls back to a diagonal sample covariance if the estimator fails, and always
    returns a symmetric positive-definite matrix (a small ridge is added).

    Args:
        returns: Period returns, rows = time.

    Returns:
        ``(N, N)`` covariance matrix.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.all(np.isfinite(r), axis=1)]
    n = r.shape[1]
    try:
        cov = LedoitWolf().fit(r).covariance_
    except Exception as exc:  # noqa: BLE001 - degrade gracefully on degenerate windows
        logger.debug("Ledoit-Wolf failed (%s); using diagonal covariance", exc)
        cov = np.diag(np.var(r, axis=0, ddof=1)) if r.shape[0] > 1 else np.eye(n) * 1e-4
    cov = 0.5 * (cov + cov.T)
    ridge = 1e-10 + 1e-6 * float(np.trace(cov)) / max(n, 1)
    return cov + ridge * np.eye(n)


class BaselineStrategy(ABC):
    """Periodically rebalanced strategy driven by trailing price history."""

    name: str = "baseline"

    def __init__(
        self,
        rebalance_every: int | None = 21,
        estimation_window: int = 252,
        min_history: int = 30,
        max_weight: float = 1.0,
        max_gross: float = 1.0,
    ) -> None:
        """Initialise the strategy.

        Args:
            rebalance_every: Periods between rebalances; ``None`` -> trade once, then hold.
            estimation_window: Trailing returns used for estimation.
            min_history: Minimum returns required; otherwise equal weight is used.
            max_weight: Per-asset weight cap.
            max_gross: Total long exposure.
        """
        if rebalance_every is not None and rebalance_every < 1:
            raise ValueError("rebalance_every must be >= 1 or None")
        self.rebalance_every = rebalance_every
        self.estimation_window = int(estimation_window)
        self.min_history = int(min_history)
        self.max_weight = float(max_weight)
        self.max_gross = float(max_gross)
        self._calls = 0

    def reset(self) -> None:
        """Reset the rebalance clock (call at the start of each evaluation episode)."""
        self._calls = 0

    def _should_rebalance(self) -> bool:
        if self._calls == 0:
            return True
        return self.rebalance_every is not None and self._calls % self.rebalance_every == 0

    def __call__(self, env: "PortfolioEnv") -> np.ndarray | None:
        """Return target weights for the env's current period (``None`` = hold).

        Args:
            env: The environment (only ``price_history`` / ``current_weights`` are read).

        Returns:
            Target asset weights; the current weights on non-rebalance periods.
        """
        rebalance = self._should_rebalance()
        self._calls += 1
        if not rebalance:
            return None  # hold: no order, the book drifts
        prices = env.price_history()
        rets = np.diff(np.log(prices[-(self.estimation_window + 1) :]), axis=0)
        n = prices.shape[1]
        if rets.shape[0] < self.min_history:
            return np.full(n, self.max_gross / n)
        try:
            w = self.target_weights(rets)
        except Exception as exc:  # noqa: BLE001 - never crash a backtest on a bad window
            logger.warning("%s optimisation failed (%s); falling back to 1/N", self.name, exc)
            w = np.full(n, 1.0 / n)
        w = np.nan_to_num(w, nan=0.0)
        if not np.isfinite(w).all() or w.sum() <= 0:
            w = np.full(n, 1.0 / n)
        return self._postprocess(w)

    def _postprocess(self, w: np.ndarray) -> np.ndarray:
        w = np.clip(w, 0.0, self.max_weight)
        total = w.sum()
        if total > 1.0 + 1e-9:
            w = w / total
        return w * self.max_gross

    @abstractmethod
    def target_weights(self, returns: np.ndarray) -> np.ndarray:
        """Compute unconstrained-scale weights summing to <= 1 from trailing returns.

        Args:
            returns: ``(W, N)`` trailing log returns ending at the current period.

        Returns:
            Weights of shape ``(N,)``.
        """


class EqualWeightStrategy(BaselineStrategy):
    """``1/N`` allocation; ``rebalance_every=None`` gives buy-and-hold."""

    name = "equal_weight"

    def target_weights(self, returns: np.ndarray) -> np.ndarray:  # noqa: D102
        n = returns.shape[1]
        return np.full(n, 1.0 / n)

    def __call__(self, env: "PortfolioEnv") -> np.ndarray:  # noqa: D102 - no history needed
        rebalance = self._should_rebalance()
        self._calls += 1
        if not rebalance:
            return None  # hold: no order, the book drifts
        return self._postprocess(np.full(env.n_assets, 1.0 / env.n_assets))


class MeanVarianceStrategy(BaselineStrategy):
    """Long-only Markowitz mean-variance with Ledoit-Wolf covariance."""

    name = "mean_variance"

    def __init__(
        self,
        risk_aversion: float = 5.0,
        mean_shrinkage: float = 0.5,
        fully_invested: bool = True,
        **kwargs: Any,
    ) -> None:
        """Initialise the strategy.

        Args:
            risk_aversion: ``gamma`` in the quadratic utility.
            mean_shrinkage: Weight in ``[0, 1]`` pulling each mean toward the
                cross-sectional average (sample means are very noisy).
            fully_invested: Enforce ``sum(w) = 1``; otherwise ``sum(w) <= 1`` (cash allowed).
            **kwargs: Forwarded to :class:`BaselineStrategy`.
        """
        super().__init__(**kwargs)
        if risk_aversion <= 0:
            raise ValueError("risk_aversion must be positive")
        self.risk_aversion = float(risk_aversion)
        self.mean_shrinkage = float(np.clip(mean_shrinkage, 0.0, 1.0))
        self.fully_invested = fully_invested

    def target_weights(self, returns: np.ndarray) -> np.ndarray:  # noqa: D102
        n = returns.shape[1]
        cov = ledoit_wolf_covariance(returns)
        mu = np.nanmean(returns, axis=0)
        mu = (1.0 - self.mean_shrinkage) * mu + self.mean_shrinkage * mu.mean()
        g = self.risk_aversion

        def objective(w: np.ndarray) -> tuple[float, np.ndarray]:
            cw = cov @ w
            return float(-(mu @ w) + 0.5 * g * (w @ cw)), -mu + g * cw

        if self.fully_invested:
            cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0, "jac": lambda w: np.ones_like(w)}]
        else:
            cons = [{"type": "ineq", "fun": lambda w: 1.0 - w.sum(), "jac": lambda w: -np.ones_like(w)}]
        x0 = np.full(n, 1.0 / n)
        res = minimize(
            objective,
            x0,
            jac=True,
            method="SLSQP",
            bounds=[(0.0, self.max_weight)] * n,
            constraints=cons,
            options={"maxiter": 200, "ftol": 1e-12},
        )
        if not res.success:
            raise RuntimeError(res.message)
        return np.asarray(res.x)


class RiskParityStrategy(BaselineStrategy):
    """Equal Risk Contribution portfolio (long-only, fully invested)."""

    name = "risk_parity"

    def target_weights(self, returns: np.ndarray) -> np.ndarray:  # noqa: D102
        return equal_risk_contribution(ledoit_wolf_covariance(returns))


def equal_risk_contribution(cov: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """Solve for ERC weights given a covariance matrix.

    Args:
        cov: ``(N, N)`` positive-definite covariance.
        tol: Optimiser tolerance.

    Returns:
        Weights summing to 1 whose risk contributions ``w_i (Sigma w)_i`` are equal.

    Raises:
        RuntimeError: If the optimiser fails.
    """
    n = cov.shape[0]
    scale = float(np.sqrt(np.mean(np.diag(cov))))  # condition the problem
    c = cov / scale**2

    def objective(y: np.ndarray) -> tuple[float, np.ndarray]:
        cy = c @ y
        return float(0.5 * y @ cy - np.sum(np.log(y)) / n), cy - 1.0 / (n * y)

    y0 = 1.0 / np.sqrt(np.diag(c))
    y0 /= y0.sum()
    res = minimize(
        objective, y0, jac=True, method="L-BFGS-B", bounds=[(1e-10, None)] * n, options={"gtol": tol, "maxiter": 1000}
    )
    if not res.success and not np.isfinite(res.fun):
        raise RuntimeError(res.message)
    y = np.asarray(res.x)
    return y / y.sum()


class HierarchicalRiskParityStrategy(BaselineStrategy):
    """Hierarchical Risk Parity (long-only, fully invested)."""

    name = "hrp"

    def target_weights(self, returns: np.ndarray) -> np.ndarray:  # noqa: D102
        return hierarchical_risk_parity(ledoit_wolf_covariance(returns))


def _cluster_variance(cov: np.ndarray, idx: list[int]) -> float:
    sub = cov[np.ix_(idx, idx)]
    ivp = 1.0 / np.diag(sub)
    ivp /= ivp.sum()
    return float(ivp @ sub @ ivp)


def hierarchical_risk_parity(cov: np.ndarray, method: str = "single") -> np.ndarray:
    """Hierarchical Risk Parity weights (Lopez de Prado, 2016).

    Args:
        cov: ``(N, N)`` covariance matrix.
        method: Linkage method for the hierarchical clustering.

    Returns:
        Weights summing to 1.
    """
    n = cov.shape[0]
    if n == 1:
        return np.ones(1)
    vol = np.sqrt(np.diag(cov))
    corr = np.clip(cov / np.outer(vol, vol), -1.0, 1.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    order = leaves_list(linkage(squareform(dist, checks=False), method=method)).tolist()
    w = np.ones(n)
    clusters = [order]
    while clusters:
        nxt = []
        for c in clusters:
            if len(c) <= 1:
                continue
            half = len(c) // 2
            left, right = c[:half], c[half:]
            v_l, v_r = _cluster_variance(cov, left), _cluster_variance(cov, right)
            alpha = 1.0 - v_l / (v_l + v_r)
            w[left] *= alpha
            w[right] *= 1.0 - alpha
            nxt += [left, right]
        clusters = nxt
    return w / w.sum()


class VolTargetOverlay:
    """Post-hoc volatility-targeting overlay on top of a learned policy.

    The policy's target weights are scaled down (never up) so that the ex-ante
    portfolio volatility, estimated with a trailing Ledoit-Wolf covariance, does not
    exceed ``target_vol``; the remainder goes to cash. This is the classic
    risk-management overlay that operates outside the learning loop.
    """

    name = "vol_target_overlay"

    def __init__(self, action_fn: Any, target_vol: float = 0.10, window: int = 63,
                 periods_per_year: int = 252, min_history: int = 20) -> None:
        """Initialise the overlay.

        Args:
            action_fn: Observation -> action function of the underlying policy.
            target_vol: Annualised volatility cap.
            window: Trailing returns used for the covariance estimate.
            periods_per_year: Annualisation factor.
            min_history: Minimum returns before the overlay activates.
        """
        if target_vol <= 0:
            raise ValueError("target_vol must be positive")
        self.action_fn = action_fn
        self.target_vol, self.window = float(target_vol), int(window)
        self.periods_per_year, self.min_history = int(periods_per_year), int(min_history)

    def reset(self) -> None:
        """Stateless; present for interface compatibility."""

    def __call__(self, env: "PortfolioEnv") -> np.ndarray:
        """Scaled target weights for the env's current period."""
        w = env.action_to_weights(self.action_fn(env.observation()))
        prices = env.price_history()[-(self.window + 1):]
        rets = np.diff(np.log(prices), axis=0)
        if rets.shape[0] < self.min_history:
            return w
        vol = float(np.sqrt(max(w @ ledoit_wolf_covariance(rets) @ w, 0.0) * self.periods_per_year))
        return w * min(1.0, self.target_vol / vol) if vol > 1e-12 else w


def build_baselines(cfg: Mapping[str, Any], max_gross: float = 1.0) -> dict[str, BaselineStrategy]:
    """Instantiate the standard baseline set from the ``experiments.baselines`` section.

    Args:
        cfg: Baseline config mapping.
        max_gross: Gross exposure used by every baseline.

    Returns:
        ``{display_name: strategy}``.
    """
    common = {
        "estimation_window": int(cfg.get("estimation_window", 252)),
        "max_weight": float(cfg.get("max_weight", 1.0)),
        "max_gross": max_gross,
    }
    every = int(cfg.get("rebalance_every", 21))
    out: dict[str, BaselineStrategy] = {
        "Equal Weight (1/N)": EqualWeightStrategy(rebalance_every=every, **common),
        "Buy & Hold (1/N)": EqualWeightStrategy(rebalance_every=None, **common),
        "MVO (Ledoit-Wolf)": MeanVarianceStrategy(
            risk_aversion=float(cfg.get("mvo_risk_aversion", 5.0)),
            mean_shrinkage=float(cfg.get("mvo_mean_shrinkage", 0.5)),
            fully_invested=bool(cfg.get("mvo_fully_invested", True)),
            rebalance_every=every,
            **common,
        ),
        "Risk Parity (ERC)": RiskParityStrategy(rebalance_every=every, **common),
        "HRP": HierarchicalRiskParityStrategy(rebalance_every=every, **common),
    }
    if cfg.get("daily_variants", False):  # trading-frequency-matched versions (rebalance every day)
        out["Equal Weight (1/N, daily)"] = EqualWeightStrategy(rebalance_every=1, **common)
        out["Risk Parity (ERC, daily)"] = RiskParityStrategy(rebalance_every=1, **common)
    return out
