"""Tests for walk-forward splits, performance metrics, baselines and features."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from src.backtest import run_episode, split_validation, walk_forward_splits
from src.baselines import (
    EqualWeightStrategy,
    HierarchicalRiskParityStrategy,
    hierarchical_risk_parity,
    MeanVarianceStrategy,
    RiskParityStrategy,
    equal_risk_contribution,
    ledoit_wolf_covariance,
)
from src.data import generate_synthetic_market
from src.features import compute_features
from src.metrics import compute_performance_metrics, safe_divide, summary_table, to_markdown_table


# --------------------------------------------------------------------- splits
@pytest.mark.parametrize("expanding", [False, True])
def test_walk_forward_folds_are_chronological(expanding: bool) -> None:
    folds = walk_forward_splits(2000, 100, 500, 200, embargo=5, expanding=expanding)
    assert len(folds) >= 5
    for f in folds:
        assert f.train_start >= 100
        assert f.train_end + 5 <= f.test_start < f.test_end <= 2000
    for a, b in zip(folds, folds[1:]):
        assert b.test_start >= a.test_end  # test windows never overlap
    if not expanding:
        assert all(f.train_end - f.train_start <= 500 for f in folds)


def test_end_anchored_folds() -> None:
    folds = walk_forward_splits(1500, 120, 600, 150, embargo=5, n_folds=3)
    assert [f.test_start for f in folds] == [1050, 1200, 1350]
    assert folds[-1].test_end == 1500
    assert all(f.train_end == f.test_start - 5 for f in folds)
    # evaluation ranges chain: fold k marks to market at the next fold's first decision period
    assert folds[0].evaluation_range(1500) == (1050, 1201)
    assert folds[-1].evaluation_range(1500) == (1350, 1500)


def test_walk_forward_raises_when_nothing_fits() -> None:
    with pytest.raises(ValueError):
        walk_forward_splits(300, 100, 500, 200)


def test_split_validation() -> None:
    tr, va = split_validation((100, 1100), 0.2)
    assert tr == (100, 900) and va == (900, 1100)
    assert split_validation((0, 100), 0.1) == ((0, 100), None)


# --------------------------------------------------------------------- metrics
def test_metrics_known_values() -> None:
    r = np.array([0.10, -0.05, 0.02, -0.10, 0.05])
    m = compute_performance_metrics(r, turnover=np.full(5, 0.2), costs=np.full(5, 1e-4),
                                    cost_dollars=np.full(5, 100.0), periods_per_year=252)
    equity = np.cumprod(1 + r)
    assert m["cumulative_return_pct"] == pytest.approx(100 * (equity[-1] - 1))
    assert m["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * math.sqrt(252))
    downside = math.sqrt(np.mean(np.minimum(r, 0) ** 2))
    assert m["sortino"] == pytest.approx(r.mean() / downside * math.sqrt(252))
    peak = np.maximum.accumulate(np.concatenate([[1.0], equity]))
    mdd = np.max(1 - np.concatenate([[1.0], equity]) / peak)
    assert m["max_drawdown_pct"] == pytest.approx(100 * mdd)
    assert m["calmar"] == pytest.approx(m["annualized_return_pct"] / 100 / mdd)
    assert m["avg_daily_turnover"] == pytest.approx(0.2)
    assert m["total_cost_pct"] == pytest.approx(0.05)
    assert m["total_cost_usd"] == pytest.approx(500.0)


def test_metrics_zero_volatility_and_empty() -> None:
    m = compute_performance_metrics(np.zeros(100))
    assert math.isnan(m["sharpe"]) and math.isnan(m["sortino"]) and math.isnan(m["calmar"])
    assert m["cumulative_return_pct"] == 0.0 and m["max_drawdown_pct"] == 0.0
    empty = compute_performance_metrics([])
    assert math.isnan(empty["sharpe"]) and empty["n_periods"] == 0


def test_metrics_nan_and_total_loss_are_handled() -> None:
    m = compute_performance_metrics([0.01, np.nan, -np.inf, 0.02])
    assert m["n_invalid_returns"] == 2.0 and math.isfinite(m["sharpe"])
    wiped = compute_performance_metrics([0.05, -1.0, 0.1])
    assert wiped["cumulative_return_pct"] == pytest.approx(-100.0)
    assert wiped["max_drawdown_pct"] == pytest.approx(100.0)
    assert wiped["annualized_return_pct"] == pytest.approx(-100.0)


def test_safe_divide() -> None:
    assert safe_divide(1.0, 0.0) != safe_divide(1.0, 0.0)  # NaN
    assert safe_divide(1.0, 0.0, default=0.0) == 0.0
    assert safe_divide(1.0, 4.0) == 0.25


def test_markdown_table_renders_nan() -> None:
    table = summary_table({"A": {"sharpe": 1.234, "sortino": float("nan")}}, ["sharpe", "sortino"])
    md = to_markdown_table(table, floatfmt=".2f")
    assert "1.23" in md and "n/a" in md and md.count("\n") == 2


# --------------------------------------------------------------------- baselines
def test_ledoit_wolf_is_spd() -> None:
    r = np.random.default_rng(0).normal(0, 0.01, (40, 8))
    cov = ledoit_wolf_covariance(r)
    assert np.allclose(cov, cov.T) and np.linalg.eigvalsh(cov).min() > 0


def test_erc_equalises_risk_contributions() -> None:
    rng = np.random.default_rng(1)
    a = rng.normal(size=(5, 5))
    cov = a @ a.T / 5 + np.diag(rng.uniform(0.1, 1.0, 5))
    w = equal_risk_contribution(cov)
    rc = w * (cov @ w)
    assert w.sum() == pytest.approx(1.0) and np.all(w > 0)
    np.testing.assert_allclose(rc / rc.sum(), np.full(5, 0.2), atol=1e-5)


def test_erc_diagonal_is_inverse_vol() -> None:
    vols = np.array([0.1, 0.2, 0.4])
    w = equal_risk_contribution(np.diag(vols**2))
    np.testing.assert_allclose(w, (1 / vols) / (1 / vols).sum(), atol=1e-6)


def test_hrp_diagonal_is_inverse_variance() -> None:
    var = np.array([0.01, 0.04, 0.09, 0.16])
    w = hierarchical_risk_parity(np.diag(var))
    np.testing.assert_allclose(w, (1 / var) / (1 / var).sum(), atol=1e-10)


def test_hrp_weights_valid_on_correlated_assets() -> None:
    rng = np.random.default_rng(4)
    f = rng.normal(size=(500, 1))
    rets = 0.01 * (np.hstack([f, f, -f, rng.normal(size=(500, 2))]) + 0.5 * rng.normal(size=(500, 5)))
    w = HierarchicalRiskParityStrategy().target_weights(rets)
    assert w.sum() == pytest.approx(1.0) and np.all(w > 0)


def test_mvo_long_only_fully_invested() -> None:
    rng = np.random.default_rng(2)
    rets = rng.normal([0.001, 0.0, -0.001], 0.01, (252, 3))
    w = MeanVarianceStrategy(risk_aversion=5.0, mean_shrinkage=0.0).target_weights(rets)
    assert w.sum() == pytest.approx(1.0, abs=1e-6) and np.all(w >= -1e-9)
    assert w[0] > w[2]


def test_baselines_run_through_env(make_env) -> None:
    for strat in (EqualWeightStrategy(rebalance_every=5), EqualWeightStrategy(rebalance_every=None),
                  MeanVarianceStrategy(min_history=10, rebalance_every=5),
                  RiskParityStrategy(min_history=10, rebalance_every=5),
                  HierarchicalRiskParityStrategy(min_history=10, rebalance_every=5)):
        env = make_env(friction={"commission_bps": 1.0}, start_index=20)
        rec = run_episode(env, weights_fn=strat)
        assert len(rec) == env.end_index - 1 - env.start_index
        assert np.all(np.isfinite(rec["return"])) and (rec["cost"] >= 0).all()
        assert rec["gross_exposure"].max() <= 1.0 + 1e-6
    # Buy-and-hold trades only on the first step.
    rec = run_episode(make_env(friction={"commission_bps": 1.0}), weights_fn=EqualWeightStrategy(None))
    assert (rec["turnover"].iloc[1:] < 1e-9).all() and rec["turnover"].iloc[0] > 0.9


# --------------------------------------------------------------------- features / data
def test_features_are_causal_and_finite() -> None:
    cfg = {"windows": [5, 21], "zscore_window": 60, "zscore_min_periods": 20, "clip": 5.0}
    data = generate_synthetic_market(n_assets=3, n_periods=300, seed=0)
    fs = compute_features(data, cfg)
    assert fs.values.shape == (300, 3, fs.n_features) and np.isfinite(fs.values).all()
    assert np.abs(fs.values).max() <= 5.0 and 0 < fs.warmup < 100
    # Perturbing the future leaves past features unchanged.
    cut = 200
    close = data.close.copy()
    close[cut:] *= 1.5
    from src.data import MarketData

    future = np.where(np.arange(300) >= cut, 1.5, 1.0)[:, None]
    shocked = MarketData(data.dates, data.assets, data.open, data.high * future, data.low, close,
                         data.volume * future**2)
    fs2 = compute_features(shocked, cfg)
    np.testing.assert_array_equal(fs.values[:cut], fs2.values[:cut])


def test_synthetic_market_is_valid() -> None:
    data = generate_synthetic_market(n_assets=4, n_periods=500, seed=1)
    assert (data.high >= np.maximum(data.open, data.close) - 1e-9).all()
    assert (data.low <= np.minimum(data.open, data.close) + 1e-9).all()
    assert isinstance(data.dates, pd.DatetimeIndex) and data.n_assets == 4
    assert data.regimes is not None and set(np.unique(data.regimes)) <= {0, 1}
    assert data.slice(10, 20).regimes.shape == (10,)


def test_tail_and_conditional_metrics() -> None:
    from src.metrics import conditional_metrics, tail_statistics

    r = np.array([0.01, -0.02, 0.03, -0.10, 0.02, 0.01, 0.0])
    t = tail_statistics(r, horizon=2)
    assert t["worst_day_pct"] == pytest.approx(-10.0)
    assert t["worst_2d_pct"] == pytest.approx(100 * ((1 - 0.10) * (1 + 0.02) - 1))
    mask = np.array([0, 1, 0, 1, 0, 0, 0], dtype=bool)
    c = conditional_metrics(r, mask)
    assert c["n"] == 2 and c["worst_day_pct"] == pytest.approx(-10.0) and c["hit_rate"] == 0.0
    assert conditional_metrics(r, np.zeros(7, bool))["n"] == 0


def test_sharpe_bootstrap() -> None:
    from src.metrics import sharpe_difference_bootstrap

    rng = np.random.default_rng(0)
    base = rng.normal(0.0003, 0.01, 1500)
    same = sharpe_difference_bootstrap(base, base, n_boot=200)
    assert same["diff"] == pytest.approx(0.0) and same["p_value"] == pytest.approx(1.0)
    better = sharpe_difference_bootstrap(base + 0.002, base, n_boot=500)
    assert better["diff"] > 0 and better["ci_low"] > 0 and better["p_value"] < 0.01


def test_regimes_from_periods() -> None:
    from src.data import regimes_from_periods

    dates = pd.bdate_range("2020-01-01", periods=10)
    lab = regimes_from_periods(dates, [["2020-01-03", "2020-01-07"]])
    assert lab.tolist() == [0, 0, 1, 1, 1, 0, 0, 0, 0, 0]


def test_vol_target_overlay_scales_down_only(make_env) -> None:
    from src.baselines import VolTargetOverlay

    env = make_env(start_index=40)
    env.reset(seed=0)
    full = lambda obs: env.weights_to_action(np.array([0.5, 0.3, 0.2]))  # noqa: E731
    tight = VolTargetOverlay(full, target_vol=0.01)(env)
    loose = VolTargetOverlay(full, target_vol=10.0)(env)
    assert tight.sum() < loose.sum() and np.all(tight >= 0)
    np.testing.assert_allclose(loose, env.action_to_weights(full(None)), atol=1e-9)


def test_lagrangian_multiplier_update() -> None:
    from src.train import LagrangianCallback

    cb = LagrangianCallback(cvar_target=0.02, drawdown_target=0.01, lr=0.5, lambda_max=3.0)
    assert cb.update([0.04, 0.005]) == pytest.approx([0.5, 0.0])  # CVaR 2x target; DD2 below target
    for _ in range(10):
        cb.update([0.04, None])
    assert cb.lambdas == pytest.approx([3.0, 0.0])  # clipped at lambda_max; None skips
    assert cb.update([0.0, 0.0]) == pytest.approx([2.5, 0.0])  # slack constraint relaxes the multiplier
