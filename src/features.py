"""Causal multi-timeframe technical features and order-flow imbalance proxies.

Every feature value at period ``t`` is a function of data observed at or before the
close of ``t`` only. Normalisation is a rolling (or expanding) z-score that likewise
uses only past and current values, so the full-history feature tensor can be computed
once and sliced for walk-forward folds without look-ahead leakage.

Order-flow imbalance (OFI) is not directly observable from daily bars, so two
standard proxies are provided:

* **Bulk Volume Classification** (Easley, Lopez de Prado & O'Hara, 2012): the buy
  fraction of a bar's volume is ``Phi(r_t / sigma_t)``; signed volume is
  ``V_t * (2 * Phi(r_t / sigma_t) - 1)``.
* **Close Location Value**: ``((C - L) - (H - C)) / (H - L)`` weights volume by where
  the bar closed inside its range (accumulation / distribution pressure).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr

from src.data import MarketData

_EPS = 1e-12


@dataclass(frozen=True)
class FeatureSet:
    """Normalised feature tensor aligned with a :class:`MarketData` panel.

    Attributes:
        values: Array of shape ``(T, N, F)`` (float32). Rows before ``warmup`` are zero.
        names: Feature names of length ``F``.
        warmup: First period index at which every feature is fully defined.
    """

    values: np.ndarray
    names: list[str]
    warmup: int

    @property
    def n_features(self) -> int:
        """Number of features per asset ``F``."""
        return int(self.values.shape[2])


def _rolling_zscore(
    df: pd.DataFrame, window: int, min_periods: int, mode: str
) -> pd.DataFrame:
    """Causal z-score using rolling or expanding moments up to and including ``t``."""
    if mode == "expanding":
        roll = df.expanding(min_periods=min_periods)
    elif mode == "rolling":
        roll = df.rolling(window=window, min_periods=min_periods)
    else:
        raise ValueError(f"Unknown normalization '{mode}' (expected 'rolling' or 'expanding')")
    mean = roll.mean()
    std = roll.std()
    z = (df - mean) / std.where(std > _EPS)
    return z


def order_flow_imbalance_bvc(
    close: pd.DataFrame, volume: pd.DataFrame, sigma_window: int, window: int
) -> pd.DataFrame:
    """Bulk-volume-classification order-flow imbalance over ``window`` periods.

    Args:
        close: Close prices ``(T, N)``.
        volume: Volumes ``(T, N)``.
        sigma_window: Window of the return volatility used to standardise returns.
        window: Aggregation window for the imbalance ratio.

    Returns:
        Imbalance ratio in ``[-1, 1]``: net signed volume / total volume.
    """
    log_ret = np.log(close).diff()
    sigma = log_ret.rolling(sigma_window, min_periods=max(2, sigma_window // 2)).std()
    buy_frac = pd.DataFrame(
        ndtr((log_ret / sigma.where(sigma > _EPS)).to_numpy()), index=close.index, columns=close.columns
    )
    signed = volume * (2.0 * buy_frac - 1.0)
    num = signed.rolling(window, min_periods=window).sum()
    den = volume.rolling(window, min_periods=window).sum()
    return num / den.where(den > _EPS)


def order_flow_imbalance_clv(
    high: pd.DataFrame, low: pd.DataFrame, close: pd.DataFrame, volume: pd.DataFrame, window: int
) -> pd.DataFrame:
    """Close-location-value weighted volume imbalance over ``window`` periods.

    Args:
        high: High prices.
        low: Low prices.
        close: Close prices.
        volume: Volumes.
        window: Aggregation window.

    Returns:
        Imbalance ratio in ``[-1, 1]``.
    """
    rng = (high - low).where((high - low) > _EPS)
    clv = (((close - low) - (high - close)) / rng).fillna(0.0)
    num = (clv * volume).rolling(window, min_periods=window).sum()
    den = volume.rolling(window, min_periods=window).sum()
    return num / den.where(den > _EPS)


def raw_features(data: MarketData, windows: Sequence[int], ofi_sigma_window: int) -> dict[str, pd.DataFrame]:
    """Compute un-normalised features keyed by name, each a ``(T, N)`` frame.

    Args:
        data: Market panel.
        windows: Look-back windows (multi-timeframe).
        ofi_sigma_window: Volatility window for BVC.

    Returns:
        Ordered mapping ``name -> DataFrame``.
    """
    idx = data.dates
    cols = data.assets
    close = pd.DataFrame(data.close, index=idx, columns=cols)
    high = pd.DataFrame(data.high, index=idx, columns=cols)
    low = pd.DataFrame(data.low, index=idx, columns=cols)
    volume = pd.DataFrame(data.volume, index=idx, columns=cols)
    log_close = np.log(close)
    log_ret = log_close.diff()

    feats: dict[str, pd.DataFrame] = {
        "ret_1": log_ret,
        "range_1": np.log(high / low),  # Parkinson-style intrabar range
    }
    for w in windows:
        feats[f"mom_{w}"] = log_close - log_close.shift(w)
        feats[f"vol_{w}"] = log_ret.rolling(w, min_periods=w).std()
        feats[f"trend_{w}"] = close / close.rolling(w, min_periods=w).mean() - 1.0
        feats[f"volu_{w}"] = np.log((volume + 1.0) / (volume.rolling(w, min_periods=w).mean() + 1.0))
        feats[f"ofi_bvc_{w}"] = order_flow_imbalance_bvc(close, volume, ofi_sigma_window, w)
        feats[f"ofi_clv_{w}"] = order_flow_imbalance_clv(high, low, close, volume, w)
    return feats


def compute_features(data: MarketData, cfg: Mapping[str, Any]) -> FeatureSet:
    """Build the normalised ``(T, N, F)`` feature tensor.

    Args:
        data: Market panel.
        cfg: The ``features`` config section (or a full config containing it).

    Returns:
        The :class:`FeatureSet`.
    """
    fcfg = cfg.get("features", cfg)
    windows = [int(w) for w in fcfg.get("windows", [5, 21, 63])]
    z_window = int(fcfg.get("zscore_window", 252))
    z_min = int(fcfg.get("zscore_min_periods", 63))
    mode = str(fcfg.get("normalization", "rolling"))
    clip = float(fcfg.get("clip", 5.0))

    feats = raw_features(data, windows, int(fcfg.get("ofi_sigma_window", 21)))
    names = list(feats.keys())
    stacked = []
    for name in names:
        z = _rolling_zscore(feats[name].replace([np.inf, -np.inf], np.nan), z_window, z_min, mode)
        stacked.append(z.to_numpy(dtype=np.float64))
    values = np.stack(stacked, axis=-1)  # (T, N, F)

    finite_rows = np.isfinite(values).all(axis=(1, 2))
    if not finite_rows.any():
        raise ValueError(
            "No period has fully-defined features; increase data length or reduce windows "
            f"(T={data.n_periods}, max window={max(windows)}, zscore_min_periods={z_min})"
        )
    warmup = int(np.argmax(finite_rows))
    # Sporadic NaNs after warm-up (e.g. zero-variance windows) carry no information -> 0.
    values = np.nan_to_num(values, nan=0.0, posinf=clip, neginf=-clip)
    values = np.clip(values, -clip, clip)
    values[:warmup] = 0.0
    return FeatureSet(values=values.astype(np.float32), names=names, warmup=warmup)
