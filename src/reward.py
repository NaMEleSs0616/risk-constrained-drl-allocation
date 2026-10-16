r"""Tail-risk measures and the risk-adjusted reward.

The per-step reward is

.. math::

    R_t = s \cdot \left( \tilde r_t - \lambda_1 \, \mathrm{CVaR}_\alpha(r_{t-W+1:t}) - \lambda_2 \, D_t^2 \right)

where :math:`\tilde r_t` is the (log or simple) net portfolio return, the CVaR is the
historical expected shortfall of the last ``W`` simple returns (expressed as a positive
loss), :math:`D_t = 1 - \mathrm{NAV}_t / \max_{\tau \le t} \mathrm{NAV}_\tau` is the
running drawdown and :math:`s` is a reward scale. All inputs are sanitised so that the
reward is always finite, even for -100% (or worse, when short) period returns.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping, Sequence

import numpy as np

_EPS = 1e-12
_MIN_LOG_RATIO = 1e-6  # floor on 1 + r before taking logs (=> log return >= -13.8)


def _clean_returns(returns: Sequence[float] | np.ndarray) -> np.ndarray:
    """Drop non-finite values and return a float64 array."""
    arr = np.asarray(returns, dtype=np.float64).ravel()
    return arr[np.isfinite(arr)]


def historical_var(returns: Sequence[float] | np.ndarray, alpha: float = 0.95) -> float:
    """Historical Value-at-Risk expressed as a positive loss.

    Uses the lower empirical alpha-quantile of the loss distribution
    (``inverted_cdf``), i.e. the smallest loss ``l`` with ``P(L <= l) >= alpha``.

    Args:
        returns: Period returns (non-finite values are ignored).
        alpha: Confidence level in ``(0, 1)``.

    Returns:
        VaR as a loss (positive when the tail contains losses); ``0.0`` if empty.

    Raises:
        ValueError: If ``alpha`` is not in ``(0, 1)``.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    r = _clean_returns(returns)
    if r.size == 0:
        return 0.0
    return float(np.quantile(-r, alpha, method="inverted_cdf"))


def historical_cvar(
    returns: Sequence[float] | np.ndarray, alpha: float = 0.95, min_periods: int = 1
) -> float:
    r"""Historical Conditional Value-at-Risk (expected shortfall) as a positive loss.

    Uses the Rockafellar-Uryasev representation, which is exact for discrete
    distributions (it correctly weights a fractional tail atom):

    .. math::

        \mathrm{CVaR}_\alpha = \mathrm{VaR}_\alpha +
            \frac{1}{(1-\alpha) n} \sum_i \max(L_i - \mathrm{VaR}_\alpha, 0)

    Args:
        returns: Period returns (non-finite values are ignored).
        alpha: Confidence level in ``(0, 1)``.
        min_periods: Minimum number of finite observations; fewer returns ``0.0``.

    Returns:
        CVaR as a loss. Always ``>= VaR``. ``0.0`` when there is insufficient data.
    """
    r = _clean_returns(returns)
    if r.size < max(1, min_periods):
        return 0.0
    losses = -r
    var = historical_var(r, alpha)
    tail = np.maximum(losses - var, 0.0).sum() / ((1.0 - alpha) * r.size)
    return float(var + tail)


def drawdown(nav: float, peak: float) -> float:
    """Instantaneous drawdown ``1 - nav / peak`` clipped to ``[0, 1]``.

    Args:
        nav: Current net asset value.
        peak: Running maximum NAV.

    Returns:
        Drawdown in ``[0, 1]`` (``1`` if NAV is wiped out or invalid).
    """
    if not np.isfinite(nav) or not np.isfinite(peak) or peak <= _EPS:
        return 1.0
    return float(np.clip(1.0 - nav / peak, 0.0, 1.0))


def drawdown_series(nav: Sequence[float] | np.ndarray) -> np.ndarray:
    """Drawdown path of a NAV / equity series.

    Args:
        nav: NAV values in time order.

    Returns:
        Array of drawdowns in ``[0, 1]`` with the same length.
    """
    v = np.asarray(nav, dtype=np.float64)
    if v.size == 0:
        return v
    peak = np.maximum.accumulate(np.where(np.isfinite(v), v, -np.inf))
    with np.errstate(divide="ignore", invalid="ignore"):
        dd = 1.0 - v / peak
    dd = np.where(np.isfinite(dd), dd, 1.0)
    return np.clip(dd, 0.0, 1.0)


def max_drawdown(nav: Sequence[float] | np.ndarray) -> float:
    """Maximum drawdown of a NAV series as a positive fraction.

    Args:
        nav: NAV values in time order.

    Returns:
        Max drawdown in ``[0, 1]``; ``0.0`` for empty input.
    """
    dd = drawdown_series(nav)
    return float(dd.max()) if dd.size else 0.0


@dataclass
class RewardConfig:
    """Parameters of :class:`RiskAdjustedReward`.

    Attributes:
        alpha: CVaR confidence level.
        window: Rolling window of returns used for CVaR.
        min_periods: Observations required before the CVaR penalty activates.
        lambda_cvar: CVaR penalty coefficient (``lambda_1``).
        lambda_drawdown: Squared-drawdown penalty coefficient (``lambda_2``).
        return_type: ``"log"`` or ``"simple"`` return in the reward's first term.
        reward_scale: Multiplier applied to the final reward.
        clip: Symmetric bound on the scaled reward.
        cvar_floor_zero: Clamp CVaR at zero so all-gain windows give no bonus.
        lambda_turnover: Penalty per unit of one-way turnover (``sum |dw|``); 0 disables.
    """

    alpha: float = 0.95
    window: int = 63
    min_periods: int = 20
    lambda_cvar: float = 0.05
    lambda_drawdown: float = 0.5
    return_type: str = "log"
    reward_scale: float = 100.0
    clip: float = 10.0
    cvar_floor_zero: bool = True
    lambda_turnover: float = 0.0

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.window < 1 or self.min_periods < 1:
            raise ValueError("window and min_periods must be >= 1")
        if self.lambda_cvar < 0 or self.lambda_drawdown < 0 or self.lambda_turnover < 0:
            raise ValueError("penalty coefficients must be non-negative")
        if self.return_type not in ("log", "simple"):
            raise ValueError("return_type must be 'log' or 'simple'")
        if self.clip <= 0:
            raise ValueError("clip must be positive")

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "RewardConfig":
        """Build from the ``reward`` config section (``cvar_window`` aliases ``window``).

        Args:
            cfg: Reward section mapping.

        Returns:
            The config object.
        """
        aliases = {"cvar_window": "window", "cvar_min_periods": "min_periods"}
        valid = {f.name for f in fields(cls)}
        kwargs = {aliases.get(k, k): v for k, v in cfg.items()}
        return cls(**{k: v for k, v in kwargs.items() if k in valid})

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class RewardBreakdown:
    """Components of a single reward evaluation (all unscaled except ``reward``)."""

    reward: float
    portfolio_return: float
    return_term: float
    cvar: float
    drawdown: float
    cvar_penalty: float
    drawdown_penalty: float
    turnover_penalty: float = 0.0


class RiskAdjustedReward:
    """Stateful CVaR- and drawdown-penalised reward.

    Keeps a rolling window of simple returns for the CVaR estimate. Call
    :meth:`reset` at the start of each episode.

    Example:
        >>> fn = RiskAdjustedReward(RewardConfig(min_periods=1))
        >>> out = fn(portfolio_return=-0.02, drawdown=0.02)
        >>> out.reward < 0
        True
    """

    def __init__(self, config: RewardConfig | None = None) -> None:
        """Initialise the reward function.

        Args:
            config: Reward parameters; defaults are used when ``None``.
        """
        self.config = config or RewardConfig()
        self._returns: deque[float] = deque(maxlen=self.config.window)

    def reset(self) -> None:
        """Clear the rolling return history."""
        self._returns.clear()

    @property
    def returns_history(self) -> np.ndarray:
        """Returns currently held in the CVaR window."""
        return np.asarray(self._returns, dtype=np.float64)

    def current_cvar(self) -> float:
        """CVaR of the current window (respecting ``min_periods`` and the zero floor)."""
        c = historical_cvar(self._returns, self.config.alpha, self.config.min_periods)
        return max(c, 0.0) if self.config.cvar_floor_zero else c

    def set_lambdas(self, lambda_cvar: float | None = None, lambda_drawdown: float | None = None) -> None:
        """Update penalty multipliers in place (used by Lagrangian / RCPO training).

        Args:
            lambda_cvar: New CVaR multiplier (``None`` keeps the current value).
            lambda_drawdown: New drawdown multiplier (``None`` keeps the current value).
        """
        if lambda_cvar is not None:
            self.config.lambda_cvar = max(0.0, float(lambda_cvar))
        if lambda_drawdown is not None:
            self.config.lambda_drawdown = max(0.0, float(lambda_drawdown))

    def __call__(self, portfolio_return: float, drawdown: float, turnover: float = 0.0) -> RewardBreakdown:
        """Record a period return and compute the penalised reward.

        Args:
            portfolio_return: Net simple return of the period. Non-finite values are
                sanitised (NaN -> 0, -inf -> -100%, +inf -> reward clip).
            drawdown: Current drawdown ``D_t`` (clipped to ``[0, 1]``).
            turnover: One-way turnover of the period's rebalance (for the turnover penalty).

        Returns:
            The :class:`RewardBreakdown`; ``reward`` is always finite.
        """
        cfg = self.config
        r = float(np.nan_to_num(portfolio_return, nan=0.0, posinf=cfg.clip, neginf=-1.0))
        # Short books can lose more than 100%; cap at -100% for the risk statistics.
        r_stat = max(r, -1.0)
        self._returns.append(r_stat)

        if cfg.return_type == "log":
            ret_term = float(np.log(max(1.0 + r, _MIN_LOG_RATIO)))
        else:
            ret_term = r_stat

        cvar = self.current_cvar()
        dd = float(np.clip(np.nan_to_num(drawdown, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
        cvar_pen = cfg.lambda_cvar * cvar
        dd_pen = cfg.lambda_drawdown * dd**2
        to_pen = cfg.lambda_turnover * float(np.nan_to_num(max(turnover, 0.0), nan=0.0, posinf=0.0))
        raw = ret_term - cvar_pen - dd_pen - to_pen
        reward = float(np.clip(cfg.reward_scale * raw, -cfg.clip, cfg.clip))
        if not np.isfinite(reward):  # defensive: should be unreachable
            reward = -cfg.clip
        return RewardBreakdown(
            reward=reward,
            portfolio_return=r,
            return_term=ret_term,
            cvar=cvar,
            drawdown=dd,
            cvar_penalty=cvar_pen,
            drawdown_penalty=dd_pen,
            turnover_penalty=to_pen,
        )
