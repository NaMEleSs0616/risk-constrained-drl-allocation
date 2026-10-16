"""Unit tests for CVaR, drawdown and the risk-adjusted reward."""

from __future__ import annotations

import numpy as np
import pytest

from src.reward import (
    RewardConfig,
    RiskAdjustedReward,
    drawdown,
    drawdown_series,
    historical_cvar,
    historical_var,
    max_drawdown,
)


# --------------------------------------------------------------------- VaR / CVaR
def test_cvar_known_value_uniform_losses() -> None:
    # Losses 1..100 -> 95% VaR = 95, CVaR = mean of the worst 5 = 98.
    returns = -np.arange(1, 101, dtype=float)
    assert historical_var(returns, 0.95) == pytest.approx(95.0)
    assert historical_cvar(returns, 0.95) == pytest.approx(98.0)


def test_cvar_fractional_tail_atom() -> None:
    # n=10, alpha=0.95 -> tail mass 0.5 observation: CVaR equals the worst loss.
    returns = -np.arange(1, 11, dtype=float) / 100
    assert historical_cvar(returns, 0.95) == pytest.approx(0.10)
    # alpha=0.85 -> tail of 1.5 obs: (0.10 + 0.5 * 0.09) / 1.5
    assert historical_cvar(returns, 0.85) == pytest.approx((0.10 + 0.5 * 0.09) / 1.5)


def test_cvar_matches_tail_mean_on_large_sample() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, 200_000)
    losses = np.sort(-r)[::-1]
    tail_mean = losses[: int(0.05 * r.size)].mean()
    assert historical_cvar(r, 0.95) == pytest.approx(tail_mean, rel=1e-3)
    # Normal closed form: sigma * phi(z) / (1 - alpha)
    assert historical_cvar(r, 0.95) == pytest.approx(0.01 * 2.0627, rel=0.02)


@pytest.mark.parametrize("seed", range(10))
def test_cvar_dominates_var_and_mean_loss(seed: int) -> None:
    r = np.random.default_rng(seed).standard_t(3, 500) * 0.01
    cvar, var = historical_cvar(r, 0.95), historical_var(r, 0.95)
    assert cvar >= var - 1e-15
    assert cvar >= -r.mean()


def test_cvar_monotone_in_alpha() -> None:
    r = np.random.default_rng(3).normal(0, 0.02, 1000)
    vals = [historical_cvar(r, a) for a in (0.5, 0.8, 0.9, 0.95, 0.99)]
    assert all(b >= a - 1e-12 for a, b in zip(vals, vals[1:]))


def test_cvar_insufficient_and_nonfinite_data() -> None:
    assert historical_cvar([], 0.95) == 0.0
    assert historical_cvar([-0.1, 0.05], 0.95, min_periods=5) == 0.0
    assert historical_cvar([np.nan, np.inf, -np.inf], 0.95) == 0.0
    clean = historical_cvar([-0.05, 0.01, 0.02, -0.01], 0.75)
    dirty = historical_cvar([-0.05, np.nan, 0.01, 0.02, np.inf, -0.01], 0.75)
    assert clean == pytest.approx(dirty)


def test_invalid_alpha() -> None:
    with pytest.raises(ValueError):
        historical_var([0.1], 1.0)
    with pytest.raises(ValueError):
        RewardConfig(alpha=0.0)


# --------------------------------------------------------------------- drawdown
def test_drawdown_scalar() -> None:
    assert drawdown(80.0, 100.0) == pytest.approx(0.2)
    assert drawdown(120.0, 100.0) == 0.0
    assert drawdown(-5.0, 100.0) == 1.0
    assert drawdown(np.nan, 100.0) == 1.0
    assert drawdown(50.0, 0.0) == 1.0


def test_drawdown_series_and_max() -> None:
    nav = np.array([100, 110, 99, 121, 60.5, 130])
    dd = drawdown_series(nav)
    np.testing.assert_allclose(dd, [0, 0, 0.1, 0, 0.5, 0], atol=1e-12)
    assert max_drawdown(nav) == pytest.approx(0.5)
    assert max_drawdown([]) == 0.0
    assert max_drawdown([1.0, 0.0]) == pytest.approx(1.0)


# --------------------------------------------------------------------- reward
def test_reward_formula_simple_returns() -> None:
    cfg = RewardConfig(alpha=0.5, window=4, min_periods=1, lambda_cvar=0.5, lambda_drawdown=2.0,
                       return_type="simple", reward_scale=1.0, clip=100.0)
    fn = RiskAdjustedReward(cfg)
    for r in (0.01, -0.02, 0.03):
        fn(r, 0.0)
    out = fn(-0.04, 0.1)
    expected_cvar = historical_cvar([0.01, -0.02, 0.03, -0.04], 0.5)
    assert out.cvar == pytest.approx(expected_cvar)
    assert out.reward == pytest.approx(-0.04 - 0.5 * expected_cvar - 2.0 * 0.1**2)
    assert out.drawdown_penalty == pytest.approx(0.02)


def test_reward_without_penalties_is_scaled_log_return() -> None:
    fn = RiskAdjustedReward(RewardConfig(lambda_cvar=0.0, lambda_drawdown=0.0, reward_scale=100.0,
                                         return_type="log", min_periods=1))
    out = fn(0.01, 0.3)
    assert out.reward == pytest.approx(100.0 * np.log(1.01))


def test_cvar_floor_zero_prevents_bonus() -> None:
    fn = RiskAdjustedReward(RewardConfig(min_periods=1, lambda_cvar=1.0, lambda_drawdown=0.0,
                                         return_type="simple", reward_scale=1.0))
    for _ in range(10):
        out = fn(0.01, 0.0)
    assert out.cvar == 0.0 and out.reward == pytest.approx(0.01)


def test_cvar_window_rolls() -> None:
    fn = RiskAdjustedReward(RewardConfig(window=3, min_periods=1))
    for r in (-0.5, 0.01, 0.01, 0.01):
        fn(r, 0.0)
    assert fn.returns_history.tolist() == [0.01, 0.01, 0.01]
    fn.reset()
    assert fn.returns_history.size == 0


@pytest.mark.parametrize("ret", [-1.0, -0.999999, -5.0, np.nan, -np.inf, np.inf, 1e9])
@pytest.mark.parametrize("return_type", ["log", "simple"])
def test_reward_finite_under_extreme_moves(ret: float, return_type: str) -> None:
    cfg = RewardConfig(min_periods=1, lambda_cvar=1.0, lambda_drawdown=1.0, return_type=return_type)
    fn = RiskAdjustedReward(cfg)
    fn(-0.2, 0.2)
    out = fn(ret, 1.0 if not np.isfinite(ret) or ret <= -1 else 0.5)
    assert np.isfinite(out.reward)
    assert -cfg.clip <= out.reward <= cfg.clip
    assert np.isfinite(out.cvar) and 0.0 <= out.drawdown <= 1.0


def test_reward_penalties_monotone_in_lambdas() -> None:
    returns = np.random.default_rng(0).normal(0, 0.02, 60)
    dds = np.linspace(0, 0.3, 60)

    def total(l1: float, l2: float) -> float:
        fn = RiskAdjustedReward(RewardConfig(lambda_cvar=l1, lambda_drawdown=l2, min_periods=5, clip=1e6))
        return sum(fn(r, d).reward for r, d in zip(returns, dds))

    assert total(0, 0) > total(1, 0) > total(1, 1)
    assert total(0, 0) > total(0, 1)


def test_turnover_penalty_and_lambda_setter() -> None:
    cfg = RewardConfig(lambda_cvar=0.0, lambda_drawdown=0.0, lambda_turnover=0.001, return_type="simple",
                       reward_scale=1.0, min_periods=1)
    fn = RiskAdjustedReward(cfg)
    out = fn(0.01, 0.0, turnover=0.5)
    assert out.turnover_penalty == pytest.approx(0.0005)
    assert out.reward == pytest.approx(0.01 - 0.0005)
    fn.set_lambdas(lambda_cvar=2.0, lambda_drawdown=-1.0)
    assert fn.config.lambda_cvar == 2.0 and fn.config.lambda_drawdown == 0.0
    with pytest.raises(ValueError):
        RewardConfig(lambda_turnover=-0.1)
