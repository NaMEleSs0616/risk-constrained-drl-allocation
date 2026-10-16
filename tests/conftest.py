"""Shared fixtures: small deterministic markets and environment builders."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pytest

from src.environment import EnvConfig, FrictionConfig, PortfolioEnv
from src.reward import RewardConfig

N_ASSETS = 3
N_FEATURES = 4
LOOKBACK = 5
T = 80


def make_prices(t: int = T, n: int = N_ASSETS, seed: int = 0) -> np.ndarray:
    """Random-walk prices starting at 100."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.01, size=(t, n))
    rets[0] = 0.0
    return 100.0 * np.exp(np.cumsum(rets, axis=0))


@pytest.fixture
def market_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(features, prices, volumes)`` for a small market."""
    rng = np.random.default_rng(1)
    features = rng.normal(size=(T, N_ASSETS, N_FEATURES)).astype(np.float32)
    prices = make_prices()
    volumes = np.full((T, N_ASSETS), 1.0e6)  # ~1e8 dollar ADV
    return features, prices, volumes


@pytest.fixture
def make_env(market_arrays) -> Callable[..., PortfolioEnv]:
    """Factory for environments with configurable frictions / config fields."""

    def _make(
        friction: dict[str, Any] | None = None,
        reward: dict[str, Any] | None = None,
        prices: np.ndarray | None = None,
        start_index: int | None = None,
        **env_kwargs: Any,
    ) -> PortfolioEnv:
        features, base_prices, volumes = market_arrays
        fr = {
            "commission_bps": 0.0,
            "slippage_bps": 0.0,
            "temp_impact_coef": 0.0,
            "perm_impact_coef": 0.0,
            "max_participation": None,
            "short_borrow_bps_annual": 0.0,
        }
        fr.update(friction or {})
        kwargs: dict[str, Any] = {
            "initial_nav": 1.0e6,
            "lookback": LOOKBACK,
            "episode_length": None,
            "random_start": False,
            "friction": FrictionConfig(**fr),
        }
        kwargs.update(env_kwargs)
        return PortfolioEnv(
            features,
            base_prices if prices is None else prices,
            volumes,
            config=EnvConfig(**kwargs),
            reward_config=RewardConfig(**{"min_periods": 2, **(reward or {})}),
            start_index=start_index,
        )

    return _make
