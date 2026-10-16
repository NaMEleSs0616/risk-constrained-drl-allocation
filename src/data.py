"""Market data containers, CSV loading and a regime-switching synthetic market generator."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.utils import get_logger, resolve_path

logger = get_logger(__name__)


@dataclass(frozen=True)
class MarketData:
    """Aligned OHLCV panel for a universe of assets.

    All arrays have shape ``(T, N)``: ``T`` periods by ``N`` assets.

    Attributes:
        dates: Period timestamps of length ``T``.
        assets: Asset identifiers of length ``N``.
        open: Opening prices.
        high: High prices.
        low: Low prices.
        close: Closing prices (used for valuation and execution reference).
        volume: Traded volume in shares.
        regimes: Optional ground-truth regime label per period (synthetic data only;
            ``0`` = calm, ``1`` = stress). Never used by agents - for analysis only.
    """

    dates: pd.DatetimeIndex
    assets: list[str]
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    regimes: np.ndarray | None = None

    def __post_init__(self) -> None:
        shape = self.close.shape
        if self.close.ndim != 2:
            raise ValueError(f"close must be 2-D (T, N); got shape {shape}")
        for name in ("open", "high", "low", "volume"):
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} shape {getattr(self, name).shape} != close shape {shape}")
        if len(self.dates) != shape[0] or len(self.assets) != shape[1]:
            raise ValueError("dates / assets lengths do not match the price panel")
        if not np.all(np.isfinite(self.close)) or np.any(self.close <= 0):
            raise ValueError("close prices must be finite and strictly positive")
        if np.any(self.volume < 0) or not np.all(np.isfinite(self.volume)):
            raise ValueError("volume must be finite and non-negative")
        if self.regimes is not None and len(self.regimes) != shape[0]:
            raise ValueError("regimes length must equal the number of periods")

    @property
    def n_periods(self) -> int:
        """Number of time periods ``T``."""
        return int(self.close.shape[0])

    @property
    def n_assets(self) -> int:
        """Number of assets ``N``."""
        return int(self.close.shape[1])

    @property
    def dollar_volume(self) -> np.ndarray:
        """Traded notional per period, ``close * volume``."""
        return self.close * self.volume

    def slice(self, start: int, end: int) -> "MarketData":
        """Return the sub-panel ``[start, end)``.

        Args:
            start: First period (inclusive).
            end: Last period (exclusive).

        Returns:
            A new :class:`MarketData` restricted to the range.
        """
        sl = slice(start, end)
        return MarketData(
            dates=self.dates[sl],
            assets=list(self.assets),
            open=self.open[sl],
            high=self.high[sl],
            low=self.low[sl],
            close=self.close[sl],
            volume=self.volume[sl],
            regimes=None if self.regimes is None else self.regimes[sl],
        )


def _uniform_per_asset(rng: np.random.Generator, bounds: Sequence[float] | float, n: int) -> np.ndarray:
    """Sample ``n`` values uniformly from ``bounds`` (or broadcast a scalar)."""
    if np.isscalar(bounds):
        return np.full(n, float(bounds))  # type: ignore[arg-type]
    lo, hi = float(bounds[0]), float(bounds[1])  # type: ignore[index]
    return rng.uniform(lo, hi, size=n)


def generate_synthetic_market(
    n_assets: int = 5,
    n_periods: int = 3000,
    seed: int = 0,
    start_date: str = "2008-01-01",
    asset_names: Sequence[str] | None = None,
    annual_drift: Sequence[float] | float = (0.02, 0.10),
    annual_vol: Sequence[float] | float = (0.08, 0.35),
    market_beta: float = 0.55,
    stress_beta: float = 0.85,
    stress_entry_prob: float = 0.008,
    stress_persistence: float = 0.97,
    stress_vol_multiplier: float = 2.5,
    stress_annual_drift: float = -0.35,
    signal_persistence: float = 0.97,
    signal_strength: float = 0.06,
    tail_df: float = 4.0,
    base_dollar_volume: Sequence[float] | float = (5.0e7, 5.0e8),
    periods_per_year: int = 252,
) -> MarketData:
    """Generate a regime-switching, fat-tailed, factor-correlated OHLCV panel.

    The generator mixes (i) a two-state Markov regime (calm / stress) that raises
    volatility and correlation and turns drift negative, (ii) a one-factor
    correlation structure, (iii) Student-t innovations rescaled to unit variance, and
    (iv) a persistent AR(1) latent alpha per asset so that trend features carry
    exploitable but noisy information. Volume co-moves with absolute returns, and
    OHLC bars are consistent (``low <= min(open, close) <= max(open, close) <= high``).

    Args:
        n_assets: Number of assets.
        n_periods: Number of periods.
        seed: RNG seed.
        start_date: First business day.
        asset_names: Optional identifiers (defaults to ``ASSET_i``).
        annual_drift: Scalar or ``[lo, hi]`` range for per-asset annual drift.
        annual_vol: Scalar or ``[lo, hi]`` range for per-asset annual volatility.
        market_beta: Factor loading in the calm regime.
        stress_beta: Factor loading in the stress regime.
        stress_entry_prob: Per-period probability of entering stress.
        stress_persistence: Per-period probability of staying in stress.
        stress_vol_multiplier: Volatility multiplier in stress.
        stress_annual_drift: Annual drift applied in stress.
        signal_persistence: AR(1) coefficient of latent alpha.
        signal_strength: Latent alpha scale in units of daily sigma.
        tail_df: Student-t degrees of freedom (> 2).
        base_dollar_volume: Scalar or range for average daily dollar volume.
        periods_per_year: Annualisation factor.

    Returns:
        Synthetic :class:`MarketData`.
    """
    if tail_df <= 2:
        raise ValueError("tail_df must exceed 2 for finite variance")
    rng = np.random.default_rng(seed)
    names = list(asset_names) if asset_names else [f"ASSET_{i}" for i in range(n_assets)]
    if len(names) != n_assets:
        raise ValueError("asset_names length must equal n_assets")

    mu = _uniform_per_asset(rng, annual_drift, n_assets) / periods_per_year
    sigma = _uniform_per_asset(rng, annual_vol, n_assets) / np.sqrt(periods_per_year)
    adv_dollar = _uniform_per_asset(rng, base_dollar_volume, n_assets)
    t_scale = np.sqrt((tail_df - 2.0) / tail_df)  # rescale Student-t to unit variance

    regime = np.zeros(n_periods, dtype=np.int8)
    for t in range(1, n_periods):
        if regime[t - 1] == 0:
            regime[t] = int(rng.random() < stress_entry_prob)
        else:
            regime[t] = int(rng.random() < stress_persistence)

    signal = np.zeros((n_periods, n_assets))
    innov_scale = np.sqrt(1.0 - signal_persistence**2)
    for t in range(1, n_periods):
        signal[t] = signal_persistence * signal[t - 1] + innov_scale * rng.standard_normal(n_assets)

    factor = rng.standard_t(tail_df, size=n_periods) * t_scale
    idio = rng.standard_t(tail_df, size=(n_periods, n_assets)) * t_scale

    stress = regime[:, None].astype(float)
    beta = market_beta + (stress_beta - market_beta) * stress
    shock = beta * factor[:, None] + np.sqrt(1.0 - beta**2) * idio
    vol_t = sigma[None, :] * (1.0 + (stress_vol_multiplier - 1.0) * stress)
    drift_t = mu[None, :] * (1.0 - stress) + (stress_annual_drift / periods_per_year) * stress
    # Alpha at t is driven by the latent state at t-1 (known only through past prices).
    alpha_t = signal_strength * sigma[None, :] * np.vstack([np.zeros((1, n_assets)), signal[:-1]])
    log_ret = drift_t + alpha_t - 0.5 * vol_t**2 + vol_t * shock
    log_ret[0] = 0.0

    p0 = rng.uniform(20.0, 200.0, size=n_assets)
    close = p0[None, :] * np.exp(np.cumsum(log_ret, axis=0))
    prev_close = np.vstack([close[:1], close[:-1]])
    open_ = prev_close * np.exp(0.15 * vol_t * rng.standard_normal((n_periods, n_assets)))
    wick = np.abs(rng.standard_normal((n_periods, n_assets))) * 0.5 * vol_t
    high = np.maximum(open_, close) * np.exp(wick)
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.standard_normal((n_periods, n_assets))) * 0.5 * vol_t)

    activity = np.exp(0.25 * rng.standard_normal((n_periods, n_assets)) + 0.35 * np.abs(shock))
    activity /= activity.mean(axis=0, keepdims=True)
    volume = adv_dollar[None, :] * activity / close

    dates = pd.bdate_range(start=start_date, periods=n_periods)
    return MarketData(
        dates=dates, assets=names, open=open_, high=high, low=low, close=close, volume=volume,
        regimes=regime.astype(np.int8),
    )


def load_csv_market(path: str | Path, assets: Sequence[str] | None = None) -> MarketData:
    """Load a long-format OHLCV CSV into an aligned :class:`MarketData` panel.

    Expected columns (case-insensitive): ``date``, ``ticker``, ``close``, ``volume`` and
    optionally ``open``, ``high``, ``low``. Prices are forward-filled across missing
    periods (never back-filled, which would leak future information); leading periods
    where any asset has no price yet are dropped.

    Args:
        path: CSV file path.
        assets: Optional subset of tickers to keep (in this order).

    Returns:
        The aligned market panel.

    Raises:
        ValueError: If required columns or requested tickers are missing.
    """
    df = pd.read_csv(resolve_path(path))
    df.columns = [c.strip().lower() for c in df.columns]
    required = {"date", "ticker", "close", "volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    df["date"] = pd.to_datetime(df["date"])
    if assets:
        unknown = set(assets) - set(df["ticker"].unique())
        if unknown:
            raise ValueError(f"Tickers not found in CSV: {sorted(unknown)}")
        df = df[df["ticker"].isin(assets)]
    tickers = list(assets) if assets else sorted(df["ticker"].unique())

    def pivot(col: str) -> pd.DataFrame:
        src = col if col in df.columns else "close"
        return df.pivot_table(index="date", columns="ticker", values=src, aggfunc="last")[tickers]

    close = pivot("close").sort_index().ffill()
    valid = close.notna().all(axis=1)
    if not valid.any():
        raise ValueError("No period has prices for every requested asset")
    first = valid.idxmax()
    close = close.loc[first:]
    frames = {}
    for col in ("open", "high", "low"):
        frames[col] = pivot(col).sort_index().ffill().loc[first:].fillna(close)
    volume = pivot("volume").sort_index().loc[first:].fillna(0.0).clip(lower=0.0)
    high = np.maximum.reduce([frames["high"].to_numpy(), frames["open"].to_numpy(), close.to_numpy()])
    low = np.minimum.reduce([frames["low"].to_numpy(), frames["open"].to_numpy(), close.to_numpy()])
    logger.info("Loaded %d periods x %d assets from %s", len(close), len(tickers), path)
    return MarketData(
        dates=pd.DatetimeIndex(close.index),
        assets=tickers,
        open=frames["open"].to_numpy(dtype=float),
        high=high.astype(float),
        low=low.astype(float),
        close=close.to_numpy(dtype=float),
        volume=volume.to_numpy(dtype=float),
    )


def regimes_from_periods(dates: pd.DatetimeIndex, periods: Sequence[Sequence[str]]) -> np.ndarray:
    """Label each date 1 if it falls inside any ``[start, end]`` stress period, else 0.

    Args:
        dates: Period timestamps.
        periods: Inclusive ``[start, end]`` date strings.

    Returns:
        ``int8`` array of regime labels.
    """
    lab = np.zeros(len(dates), dtype=np.int8)
    for start, end in periods:
        lab[(dates >= pd.Timestamp(start)) & (dates <= pd.Timestamp(end))] = 1
    return lab


def load_market_data(cfg: Mapping[str, Any], seed: int = 0) -> MarketData:
    """Build the market panel described by the ``data`` config section.

    Args:
        cfg: Full configuration dictionary (uses ``cfg["data"]``).
        seed: Seed for the synthetic generator.

    Returns:
        The market panel.

    Raises:
        ValueError: On an unknown data source or missing CSV path.
    """
    data_cfg = cfg["data"]
    source = data_cfg.get("source", "synthetic")
    assets = data_cfg.get("assets")
    if source == "csv":
        if not data_cfg.get("csv_path"):
            raise ValueError("data.csv_path must be set when data.source == 'csv'")
        market = load_csv_market(data_cfg["csv_path"], assets)
        if data_cfg.get("start_date") and data_cfg.get("trim_start", False):
            first = int(np.searchsorted(market.dates, pd.Timestamp(data_cfg["start_date"])))
            market = market.slice(first, market.n_periods)
        periods = data_cfg.get("stress_periods")
        if periods:
            market = MarketData(market.dates, market.assets, market.open, market.high, market.low,
                                market.close, market.volume, regimes_from_periods(market.dates, periods))
        return market
    if source == "synthetic":
        syn = dict(data_cfg.get("synthetic", {}))
        n_assets = len(assets) if assets else int(syn.pop("n_assets", 5))
        syn.pop("n_assets", None)
        return generate_synthetic_market(
            n_assets=n_assets,
            seed=seed,
            start_date=data_cfg.get("start_date", "2008-01-01"),
            asset_names=assets,
            periods_per_year=int(cfg.get("environment", {}).get("periods_per_year", 252)),
            **syn,
        )
    raise ValueError(f"Unknown data.source '{source}' (expected 'synthetic' or 'csv')")
