r"""Gymnasium portfolio-allocation environment with realistic market frictions.

Timing convention (no look-ahead):

1. At the close of period ``t`` the agent observes features computed from data up to
   ``t``, its current (drifted) weights and its drawdown.
2. It submits target weights, which are executed at the close-``t`` reference price
   plus frictions.
3. The book is held over ``(t, t+1]`` and marked to market at the close of ``t+1``.

Friction model for a trade of signed notional :math:`q_i` in asset :math:`i`:

* Commission:          :math:`c \, |q_i|`
* Slippage/half-spread: :math:`s \, |q_i|`
* Temporary impact:    :math:`\eta \, \sigma_i \sqrt{|q_i| / \mathrm{ADV}_i} \, |q_i|`
* Permanent impact:    the trade shifts the asset's *impact state* by
  :math:`\Delta_i = \gamma \sigma_i \operatorname{sign}(q_i)\sqrt{|q_i|/\mathrm{ADV}_i}`;
  the trade itself pays the average displacement :math:`\tfrac12 |\Delta_i| |q_i|`, and
  later trades pay (or recoup) the residual displacement left by earlier trades,
  :math:`s_i q_i`. The state decays with a configurable half-life (resilience).

Positions are marked at the historical (un-impacted) price, which is conservative:
impact never creates paper gains. Accounting identity enforced every step::

    NAV_after_trade = NAV_before_trade - total_cost
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Mapping, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src.reward import RewardConfig, RiskAdjustedReward, drawdown

_EPS = 1e-12
OBS_BOUND = 10.0  # features are z-scores clipped well inside this; portfolio state is clipped to it


# ---------------------------------------------------------------------------
# Weight projections
# ---------------------------------------------------------------------------
def project_to_simplex(v: np.ndarray, total: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{w >= 0, sum(w) = total}`` (Duchi et al., 2008).

    Args:
        v: Input vector.
        total: Simplex radius (> 0).

    Returns:
        The projected vector.
    """
    v = np.asarray(v, dtype=np.float64)
    if total <= 0:
        return np.zeros_like(v)
    u = np.sort(v)[::-1]
    css = np.cumsum(u) - total
    ind = np.arange(1, v.size + 1)
    cond = u - css / ind > 0
    rho = ind[cond][-1]
    theta = css[cond][-1] / rho
    return np.maximum(v - theta, 0.0)


def project_to_capped_simplex(v: np.ndarray, cap: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{w >= 0, sum(w) <= cap}`` (remainder held in cash).

    Args:
        v: Input vector.
        cap: Maximum total long exposure.

    Returns:
        The projected vector.
    """
    w = np.maximum(np.asarray(v, dtype=np.float64), 0.0)
    if w.sum() <= cap:
        return w
    return project_to_simplex(v, cap)


def project_to_l1_ball(v: np.ndarray, radius: float = 1.0) -> np.ndarray:
    """Euclidean projection onto ``{w : ||w||_1 <= radius}`` (gross-leverage cap).

    Args:
        v: Input vector.
        radius: Maximum gross exposure.

    Returns:
        The projected vector.
    """
    v = np.asarray(v, dtype=np.float64)
    if np.abs(v).sum() <= radius:
        return v.copy()
    return np.sign(v) * project_to_simplex(np.abs(v), radius)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class FrictionConfig:
    """Market-friction parameters.

    Attributes:
        commission_bps: Proportional commission on traded notional (bps).
        slippage_bps: Half-spread paid on traded notional (bps).
        temp_impact_coef: Temporary square-root impact coefficient ``eta``.
        perm_impact_coef: Permanent square-root impact coefficient ``gamma``.
        perm_impact_halflife: Half-life (periods) of the permanent impact state;
            ``None`` means no decay.
        adv_window: Window for average daily dollar volume.
        vol_window: Window for per-asset return volatility.
        max_participation: Cap on ``|q| / ADV`` per period (``None`` disables).
        min_adv: Floor on ADV to avoid division by ~zero.
        short_borrow_bps_annual: Annual borrow fee on short notional (bps).
        cash_rate_annual: Annual interest on (positive or negative) cash.
    """

    commission_bps: float = 1.0
    slippage_bps: float = 2.0
    temp_impact_coef: float = 0.15
    perm_impact_coef: float = 0.05
    perm_impact_halflife: float | None = 5.0
    adv_window: int = 20
    vol_window: int = 20
    max_participation: float | None = 0.10
    min_adv: float = 1.0e4
    short_borrow_bps_annual: float = 50.0
    cash_rate_annual: float = 0.0

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any] | None) -> "FrictionConfig":
        """Build from a mapping, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (cfg or {}).items() if k in valid})


@dataclass
class EnvConfig:
    """Environment parameters (see ``environment`` section of the YAML config)."""

    initial_nav: float = 1.0e7
    lookback: int = 20
    episode_length: int | None = 252
    random_start: bool = True
    long_only: bool = True
    action_transform: str = "softmax"
    softmax_temperature: float = 3.0
    max_gross_leverage: float = 1.0
    bankruptcy_threshold: float = 0.10
    portfolio_vol_window: int = 21
    periods_per_year: int = 252
    trade_rate: float = 1.0
    no_trade_band: float = 0.0
    friction: FrictionConfig = field(default_factory=FrictionConfig)

    def __post_init__(self) -> None:
        if isinstance(self.friction, Mapping):
            self.friction = FrictionConfig.from_dict(self.friction)
        if self.action_transform not in ("softmax", "projection"):
            raise ValueError("action_transform must be 'softmax' or 'projection'")
        if not self.long_only and self.action_transform == "softmax":
            raise ValueError("softmax transform is long-only; use 'projection' for long/short")
        if self.lookback < 1:
            raise ValueError("lookback must be >= 1")
        if self.max_gross_leverage <= 0:
            raise ValueError("max_gross_leverage must be positive")
        if not 0.0 < self.trade_rate <= 1.0:
            raise ValueError("trade_rate must be in (0, 1]")
        if self.no_trade_band < 0:
            raise ValueError("no_trade_band must be non-negative")

    @classmethod
    def from_dict(cls, cfg: Mapping[str, Any]) -> "EnvConfig":
        """Build from the ``environment`` config section, ignoring unknown keys."""
        valid = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in cfg.items() if k in valid}
        kwargs["friction"] = FrictionConfig.from_dict(cfg.get("friction"))
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a plain (nested) dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class TradeCosts:
    """Dollar cost breakdown of one rebalance (``residual_impact`` may be negative)."""

    commission: float = 0.0
    slippage: float = 0.0
    temporary_impact: float = 0.0
    permanent_impact: float = 0.0
    residual_impact: float = 0.0

    @property
    def total(self) -> float:
        """Sum of all cost components."""
        return (
            self.commission
            + self.slippage
            + self.temporary_impact
            + self.permanent_impact
            + self.residual_impact
        )

    def as_dict(self) -> dict[str, float]:
        """Components plus total as a dictionary."""
        d = asdict(self)
        d["total"] = self.total
        return d


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
class PortfolioEnv(gym.Env):
    """Multi-asset allocation environment with tail-risk-penalised reward.

    Observation (flat ``Box``), in this order:

    * ``lookback x N x F`` feature window (rows ``t-L+1 .. t``),
    * ``N`` asset weights and ``1`` cash weight,
    * current drawdown ``D_t``,
    * annualised realised portfolio volatility over ``portfolio_vol_window``.

    Action (``Box[-1, 1]``):

    * ``long_only`` + ``softmax``: ``N + 1`` logits (last is cash) -> softmax weights.
    * ``long_only`` + ``projection``: ``N`` scores -> projection onto the capped simplex.
    * long/short (``projection``): ``N`` scores -> projection onto the L1 ball of radius
      ``max_gross_leverage``; cash is the residual ``1 - sum(w)``.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        features: np.ndarray,
        prices: np.ndarray,
        volumes: np.ndarray,
        config: EnvConfig | None = None,
        reward_config: RewardConfig | None = None,
        start_index: int | None = None,
        end_index: int | None = None,
        asset_names: Sequence[str] | None = None,
        dates: Sequence[Any] | None = None,
        render_mode: str | None = None,
    ) -> None:
        """Create the environment over the period range ``[start_index, end_index)``.

        Args:
            features: Normalised features of shape ``(T, N, F)``.
            prices: Close prices of shape ``(T, N)``.
            volumes: Volumes (shares) of shape ``(T, N)``.
            config: Environment parameters.
            reward_config: Reward parameters.
            start_index: First decision period (needs ``lookback - 1`` prior rows).
            end_index: Exclusive end; the last mark-to-market uses ``end_index - 1``.
            asset_names: Optional identifiers for ``info``/rendering.
            dates: Optional timestamps of length ``T``.
            render_mode: ``"human"`` or ``None``.

        Raises:
            ValueError: On inconsistent shapes or an empty period range.
        """
        super().__init__()
        self.config = config or EnvConfig()
        self.reward_fn = RiskAdjustedReward(reward_config or RewardConfig())
        self.render_mode = render_mode

        self.features = np.asarray(features, dtype=np.float32)
        self.prices = np.asarray(prices, dtype=np.float64)
        self.volumes = np.asarray(volumes, dtype=np.float64)
        if self.features.ndim != 3:
            raise ValueError("features must have shape (T, N, F)")
        T, N, F = self.features.shape
        if self.prices.shape != (T, N) or self.volumes.shape != (T, N):
            raise ValueError("prices and volumes must have shape (T, N) matching features")
        if not np.all(np.isfinite(self.prices)) or np.any(self.prices <= 0):
            raise ValueError("prices must be finite and strictly positive")

        L = self.config.lookback
        self.start_index = L - 1 if start_index is None else int(start_index)
        self.end_index = T if end_index is None else int(end_index)
        if self.start_index < L - 1:
            raise ValueError(f"start_index={self.start_index} < lookback - 1 = {L - 1}")
        if not self.start_index + 2 <= self.end_index <= T:
            raise ValueError(
                f"invalid range [{self.start_index}, {self.end_index}) for T={T}; need >= 2 periods"
            )

        self.n_periods, self.n_assets, self.n_features = T, N, F
        self.asset_names = list(asset_names) if asset_names is not None else [f"A{i}" for i in range(N)]
        self.dates = list(dates) if dates is not None else None

        self._precompute_market_state()

        self.market_dim = L * N * F
        self.portfolio_dim = N + 3
        obs_dim = self.market_dim + self.portfolio_dim
        self.observation_space = spaces.Box(-OBS_BOUND, OBS_BOUND, shape=(obs_dim,), dtype=np.float32)
        self.action_dim = N + 1 if self._uses_cash_logit else N
        self.action_space = spaces.Box(-1.0, 1.0, shape=(self.action_dim,), dtype=np.float32)

        fc = self.config.friction
        self._commission = fc.commission_bps * 1e-4
        self._slippage = fc.slippage_bps * 1e-4
        self._borrow_daily = fc.short_borrow_bps_annual * 1e-4 / self.config.periods_per_year
        self._cash_daily = (1.0 + fc.cash_rate_annual) ** (1.0 / self.config.periods_per_year) - 1.0
        hl = fc.perm_impact_halflife
        self._impact_decay = 1.0 if hl is None or hl <= 0 else float(0.5 ** (1.0 / hl))

        self._init_state(self.start_index)

    # ------------------------------------------------------------------ setup
    @property
    def _uses_cash_logit(self) -> bool:
        return self.config.long_only and self.config.action_transform == "softmax"

    def _precompute_market_state(self) -> None:
        """Causal rolling volatility and dollar ADV used by the impact model."""
        fc = self.config.friction
        log_ret = np.vstack([np.zeros((1, self.n_assets)), np.diff(np.log(self.prices), axis=0)])
        T = self.n_periods
        sigma = np.empty((T, self.n_assets))
        adv = np.empty((T, self.n_assets))
        dollar_vol = self.prices * self.volumes
        csum_r = np.cumsum(log_ret, axis=0)
        csum_r2 = np.cumsum(log_ret**2, axis=0)
        csum_dv = np.cumsum(dollar_vol, axis=0)
        for t in range(T):
            lo_v = max(1, t - fc.vol_window + 1)
            n_v = t - lo_v + 1
            if n_v >= 2:
                s1 = csum_r[t] - csum_r[lo_v - 1]
                s2 = csum_r2[t] - csum_r2[lo_v - 1]
                var = np.maximum((s2 - s1**2 / n_v) / (n_v - 1), 0.0)
                sigma[t] = np.sqrt(var)
            else:
                sigma[t] = np.nan
            lo_a = max(0, t - fc.adv_window + 1)
            prev = csum_dv[lo_a - 1] if lo_a > 0 else 0.0
            adv[t] = (csum_dv[t] - prev) / (t - lo_a + 1)
        fallback = 0.01  # 1% daily vol until enough history exists
        sigma = np.where(np.isfinite(sigma) & (sigma > 0), sigma, fallback)
        self.sigma = np.maximum(sigma, 1e-4)
        self.adv = np.maximum(adv, fc.min_adv)

    def _init_state(self, t0: int) -> None:
        self._t = t0
        self._episode_start = t0
        self._steps = 0
        self.cash = float(self.config.initial_nav)
        self.holdings = np.zeros(self.n_assets, dtype=np.float64)  # shares
        self.peak_nav = float(self.config.initial_nav)
        self.impact_state = np.zeros(self.n_assets, dtype=np.float64)
        self._ret_window: deque[float] = deque(maxlen=self.config.portfolio_vol_window)
        self._ep_returns: list[float] = []
        self._ep_turnover: list[float] = []
        self._ep_costs: list[float] = []
        self._ep_rewards: list[float] = []
        self._ep_navs: list[float] = [self.cash]
        self.reward_fn.reset()

    # ------------------------------------------------------------ properties
    @property
    def t(self) -> int:
        """Current decision period index."""
        return self._t

    @property
    def nav(self) -> float:
        """Net asset value at the current period's close."""
        return self._nav_at(self._t)

    @property
    def current_weights(self) -> np.ndarray:
        """Current (drifted) asset weights; zeros if NAV is non-positive."""
        nav = self.nav
        if nav <= _EPS:
            return np.zeros(self.n_assets)
        return self.holdings * self.prices[self._t] / nav

    @property
    def current_drawdown(self) -> float:
        """Drawdown of the current NAV relative to its running peak."""
        return drawdown(self.nav, self.peak_nav)

    @property
    def observation_layout(self) -> dict[str, int]:
        """Dimensions needed by the feature extractor to un-flatten observations."""
        return {
            "lookback": self.config.lookback,
            "n_assets": self.n_assets,
            "n_features": self.n_features,
            "portfolio_dim": self.portfolio_dim,
        }

    def price_history(self) -> np.ndarray:
        """Close prices up to and including the current period (no look-ahead)."""
        return self.prices[: self._t + 1]

    def _nav_at(self, t: int) -> float:
        return float(self.cash + self.holdings @ self.prices[t])

    # ------------------------------------------------------------ actions
    def action_to_weights(self, action: np.ndarray) -> np.ndarray:
        """Map a raw policy action to feasible target asset weights.

        Args:
            action: Array of shape ``action_space.shape``; non-finite entries -> 0.

        Returns:
            Target weights of shape ``(N,)``; cash weight is ``1 - sum(w)``.
        """
        a = np.clip(np.nan_to_num(np.asarray(action, dtype=np.float64), nan=0.0), -1.0, 1.0)
        if a.shape != (self.action_dim,):
            raise ValueError(f"action shape {a.shape} != {(self.action_dim,)}")
        G = self.config.max_gross_leverage
        if self._uses_cash_logit:
            logits = a * self.config.softmax_temperature
            z = np.exp(logits - logits.max())
            p = z / z.sum()
            return p[: self.n_assets] * G
        if self.config.long_only:
            return project_to_capped_simplex((a + 1.0) / 2.0, G)
        return project_to_l1_ball(a * G, G)

    def weights_to_action(self, weights: np.ndarray) -> np.ndarray:
        """Approximate inverse of :meth:`action_to_weights` (for warm starts / tests).

        Args:
            weights: Target asset weights of shape ``(N,)``.

        Returns:
            An action that maps (approximately) to ``weights``.
        """
        w = np.asarray(weights, dtype=np.float64)
        G = self.config.max_gross_leverage
        if self._uses_cash_logit:
            p = np.append(w / G, max(1.0 - w.sum() / G, 0.0))
            logits = np.log(np.maximum(p, 1e-6))
            logits -= logits.max()
            return np.clip(logits / self.config.softmax_temperature + 1.0, -1.0, 1.0)
        if self.config.long_only:
            return np.clip(2.0 * w - 1.0, -1.0, 1.0)
        return np.clip(w / G, -1.0, 1.0)

    # ------------------------------------------------------------ frictions
    def compute_trade_costs(self, trade: np.ndarray, t: int | None = None) -> tuple[TradeCosts, np.ndarray]:
        """Dollar costs of a trade vector and the resulting permanent-impact increment.

        Args:
            trade: Signed dollar notional per asset (positive = buy).
            t: Period of execution (defaults to the current period).

        Returns:
            Tuple ``(costs, impact_increment)``.
        """
        t = self._t if t is None else t
        fc = self.config.friction
        q = np.asarray(trade, dtype=np.float64)
        abs_q = np.abs(q)
        sqrt_part = np.sqrt(abs_q / self.adv[t])
        sig = self.sigma[t]
        costs = TradeCosts(
            commission=float(self._commission * abs_q.sum()),
            slippage=float(self._slippage * abs_q.sum()),
            temporary_impact=float(fc.temp_impact_coef * np.sum(sig * sqrt_part * abs_q)),
            permanent_impact=float(0.5 * fc.perm_impact_coef * np.sum(sig * sqrt_part * abs_q)),
            residual_impact=float(np.sum(self.impact_state * q)),
        )
        increment = fc.perm_impact_coef * sig * np.sign(q) * sqrt_part
        return costs, increment

    # ------------------------------------------------------------ gym API
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset to a (possibly random) start period with an all-cash book.

        Args:
            seed: RNG seed.
            options: Optional ``{"start_index": int}`` to pin the start period.

        Returns:
            ``(observation, info)``.
        """
        super().reset(seed=seed)
        options = options or {}
        t0 = self.start_index
        ep_len = self.config.episode_length
        if "start_index" in options:
            t0 = int(options["start_index"])
            if not self.start_index <= t0 <= self.end_index - 2:
                raise ValueError(f"start_index {t0} outside [{self.start_index}, {self.end_index - 2}]")
        elif self.config.random_start and ep_len:
            max_start = self.end_index - 1 - int(ep_len)
            if max_start > self.start_index:
                t0 = int(self.np_random.integers(self.start_index, max_start + 1))
        self._init_state(t0)
        return self._get_obs(), self._base_info()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Rebalance toward the weights implied by ``action`` and advance one period.

        Args:
            action: Raw policy action.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.
        """
        return self.step_with_weights(self.action_to_weights(action))

    def step_with_weights(
        self, target_weights: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Rebalance toward explicit target weights (used by baselines) and advance.

        Args:
            target_weights: Desired asset weights ``(N,)``; cash is the residual.

        Returns:
            ``(observation, reward, terminated, truncated, info)``.

        Raises:
            RuntimeError: If called after the episode has ended.
        """
        if self._t >= self.end_index - 1:
            raise RuntimeError("Episode is over; call reset()")
        w_target = np.nan_to_num(np.asarray(target_weights, dtype=np.float64), nan=0.0)
        if self.config.trade_rate < 1.0 or self.config.no_trade_band > 0.0:
            # Partial adjustment toward the target ("aim") portfolio, plus a no-trade band.
            w_cur = self.current_weights
            w_target = w_cur + self.config.trade_rate * (w_target - w_cur)
            if np.abs(w_target - w_cur).sum() < self.config.no_trade_band:
                w_target = w_cur
        fc = self.config.friction
        t = self._t
        p_t = self.prices[t]

        nav_pre = self._nav_at(t)
        trade = self._solve_trade(w_target, nav_pre, t)
        costs, increment = self.compute_trade_costs(trade, t)
        self.cash -= float(trade.sum()) + costs.total
        self.holdings = self.holdings + trade / p_t
        self.impact_state = self.impact_state + increment
        nav_post_trade = self._nav_at(t)

        # ---- hold over (t, t+1]
        t1 = t + 1
        short_notional = float(np.sum(np.maximum(-self.holdings * p_t, 0.0)))
        financing = self.cash * self._cash_daily - short_notional * self._borrow_daily
        self.cash += financing
        self.impact_state *= self._impact_decay
        self._t = t1
        self._steps += 1

        nav_next = self._nav_at(t1)
        port_ret = nav_next / nav_pre - 1.0 if nav_pre > _EPS else -1.0
        self.peak_nav = max(self.peak_nav, nav_next)
        dd = drawdown(nav_next, self.peak_nav)
        turnover = float(np.abs(trade).sum() / nav_pre) if nav_pre > _EPS else 0.0
        rb = self.reward_fn(port_ret, dd, turnover)

        cost_frac = costs.total / nav_pre if nav_pre > _EPS else 0.0
        self._ret_window.append(float(np.clip(port_ret, -1.0, 10.0)))
        self._ep_returns.append(float(port_ret))
        self._ep_turnover.append(turnover)
        self._ep_costs.append(float(cost_frac))
        self._ep_rewards.append(rb.reward)
        self._ep_navs.append(nav_next)

        terminated = bool(nav_next <= self.config.bankruptcy_threshold * self.config.initial_nav)
        ep_len = self.config.episode_length
        truncated = bool(t1 >= self.end_index - 1 or (ep_len is not None and self._steps >= ep_len))
        if terminated:
            truncated = False

        info = self._base_info()
        info.update(
            {
                "portfolio_return": float(port_ret),
                "nav_pre_trade": nav_pre,
                "nav_post_trade": nav_post_trade,
                "turnover": turnover,
                "cost": float(cost_frac),
                "costs": costs.as_dict(),
                "financing": float(financing),
                "target_weights": w_target.copy(),
                "reward_return_term": rb.return_term,
                "cvar": rb.cvar,
                "cvar_penalty": rb.cvar_penalty,
                "drawdown_penalty": rb.drawdown_penalty,
                "turnover_penalty": rb.turnover_penalty,
                "lambda_cvar": self.reward_fn.config.lambda_cvar,
                "lambda_drawdown": self.reward_fn.config.lambda_drawdown,
            }
        )
        if terminated or truncated:
            info["episode_stats"] = self.episode_stats()
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), rb.reward, terminated, truncated, info

    def set_reward_lambdas(self, lambda_cvar: float | None = None, lambda_drawdown: float | None = None) -> None:
        """Update the reward's penalty multipliers (called by the Lagrangian callback)."""
        self.reward_fn.set_lambdas(lambda_cvar, lambda_drawdown)

    def observation(self) -> np.ndarray:
        """Current observation (same as returned by ``reset`` / ``step``)."""
        return self._get_obs()

    # ------------------------------------------------------------ helpers
    def _solve_trade(self, w_target: np.ndarray, nav_pre: float, t: int, n_iter: int = 4) -> np.ndarray:
        """Size the trade so that post-cost weights match the target.

        Costs are paid from the book, so sizing on the pre-trade NAV would leave
        ``w * NAV_pre / (NAV_pre - cost) > w`` (implicit leverage). A short fixed-point
        iteration on the post-cost NAV fixes this; it converges in 2-3 iterations
        because costs are a small fraction of NAV.
        """
        current = self.holdings * self.prices[t]
        if nav_pre <= _EPS:  # insolvent book: liquidate only
            return self._cap_participation(-current, t)
        nav_eff = nav_pre
        trade = self._cap_participation(w_target * nav_eff - current, t)
        for _ in range(n_iter):
            costs, _ = self.compute_trade_costs(trade, t)
            nav_new = max(nav_pre - costs.total, _EPS)
            if abs(nav_new - nav_eff) <= 1e-12 * nav_pre:
                break
            nav_eff = nav_new
            trade = self._cap_participation(w_target * nav_eff - current, t)
        return trade

    def _cap_participation(self, trade: np.ndarray, t: int) -> np.ndarray:
        cap = self.config.friction.max_participation
        if cap is None:
            return trade
        limit = cap * self.adv[t]
        return np.clip(trade, -limit, limit)

    def _portfolio_vol(self) -> float:
        if len(self._ret_window) < 2:
            return 0.0
        return float(np.std(self._ret_window, ddof=1) * np.sqrt(self.config.periods_per_year))

    def _get_obs(self) -> np.ndarray:
        L, t = self.config.lookback, self._t
        window = self.features[t - L + 1 : t + 1].reshape(-1)
        nav = self.nav
        if nav > _EPS:
            w = self.holdings * self.prices[t] / nav
            cash_w = self.cash / nav
        else:
            w = np.zeros(self.n_assets)
            cash_w = 0.0
        port = np.concatenate([w, [cash_w, self.current_drawdown, self._portfolio_vol()]])
        obs = np.nan_to_num(np.concatenate([window, port]), nan=0.0, posinf=OBS_BOUND, neginf=-OBS_BOUND)
        return np.clip(obs, -OBS_BOUND, OBS_BOUND).astype(np.float32)

    def _base_info(self) -> dict[str, Any]:
        nav = self.nav
        w = self.current_weights
        info: dict[str, Any] = {
            "t": self._t,
            "nav": nav,
            "drawdown": self.current_drawdown,
            "weights": w,
            "cash_weight": float(self.cash / nav) if nav > _EPS else 0.0,
            "gross_exposure": float(np.abs(w).sum()),
            "net_exposure": float(w.sum()),
            "portfolio_vol": self._portfolio_vol(),
        }
        if self.dates is not None:
            info["date"] = self.dates[self._t]
        return info

    def episode_stats(self) -> dict[str, float]:
        """Summary statistics of the episode so far."""
        r = np.asarray(self._ep_returns, dtype=np.float64)
        P = self.config.periods_per_year
        navs = np.asarray(self._ep_navs)
        peak = np.maximum.accumulate(navs)
        mdd = float(np.max(1.0 - navs / np.maximum(peak, _EPS))) if navs.size else 0.0
        vol = float(np.std(r, ddof=1) * np.sqrt(P)) if r.size > 1 else 0.0
        mean = float(np.mean(r)) if r.size else 0.0
        return {
            "length": float(r.size),
            "total_return": float(navs[-1] / navs[0] - 1.0) if navs.size else 0.0,
            "ann_vol": vol,
            "sharpe": float(mean / (np.std(r, ddof=1) + _EPS) * np.sqrt(P)) if r.size > 1 else 0.0,
            "max_drawdown": mdd,
            "avg_turnover": float(np.mean(self._ep_turnover)) if self._ep_turnover else 0.0,
            "total_cost": float(np.sum(self._ep_costs)),
            "total_reward": float(np.sum(self._ep_rewards)),
        }

    def render(self) -> None:
        """Print a one-line summary of the current state."""
        label = self.dates[self._t] if self.dates is not None else self._t
        w = ", ".join(f"{n}={x:+.2f}" for n, x in zip(self.asset_names, self.current_weights))
        print(f"[{label}] NAV={self.nav:,.0f} DD={self.current_drawdown:.2%} | {w}")


def make_portfolio_env(
    features: np.ndarray,
    prices: np.ndarray,
    volumes: np.ndarray,
    cfg: Mapping[str, Any],
    start_index: int,
    end_index: int,
    overrides: Mapping[str, Any] | None = None,
    reward_overrides: Mapping[str, Any] | None = None,
    asset_names: Sequence[str] | None = None,
    dates: Sequence[Any] | None = None,
) -> PortfolioEnv:
    """Build a :class:`PortfolioEnv` from the full configuration dictionary.

    Args:
        features: ``(T, N, F)`` feature tensor.
        prices: ``(T, N)`` close prices.
        volumes: ``(T, N)`` volumes.
        cfg: Full config (uses ``environment`` and ``reward`` sections).
        start_index: First decision period.
        end_index: Exclusive end period.
        overrides: Overrides for the ``environment`` section (e.g. no random start).
        reward_overrides: Overrides for the ``reward`` section (e.g. ablation lambdas).
        asset_names: Asset identifiers.
        dates: Timestamps.

    Returns:
        The environment.
    """
    env_cfg = dict(cfg["environment"])
    env_cfg.update(overrides or {})
    rew_cfg = dict(cfg["reward"])
    rew_cfg.update(reward_overrides or {})
    return PortfolioEnv(
        features=features,
        prices=prices,
        volumes=volumes,
        config=EnvConfig.from_dict(env_cfg),
        reward_config=RewardConfig.from_dict(rew_cfg),
        start_index=start_index,
        end_index=end_index,
        asset_names=asset_names,
        dates=dates,
    )
