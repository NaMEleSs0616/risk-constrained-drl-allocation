"""Unit tests for environment state transitions and friction accounting."""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src.environment import (
    EnvConfig,
    FrictionConfig,
    PortfolioEnv,
    project_to_capped_simplex,
    project_to_l1_ball,
    project_to_simplex,
)
from tests.conftest import LOOKBACK, N_ASSETS, N_FEATURES, T

FULL_FRICTION = {
    "commission_bps": 5.0,
    "slippage_bps": 3.0,
    "temp_impact_coef": 0.2,
    "perm_impact_coef": 0.1,
    "perm_impact_halflife": 3.0,
    "max_participation": None,
    "short_borrow_bps_annual": 100.0,
}


# --------------------------------------------------------------------- spaces / reset
def test_gymnasium_api_compliance(make_env) -> None:
    check_env(make_env(episode_length=20, random_start=True), skip_render_check=True)


def test_observation_and_action_shapes(make_env) -> None:
    env = make_env()
    obs, info = env.reset(seed=0)
    assert obs.shape == (LOOKBACK * N_ASSETS * N_FEATURES + N_ASSETS + 3,)
    assert env.action_space.shape == (N_ASSETS + 1,)  # softmax includes a cash logit
    assert obs.dtype == np.float32 and np.all(np.isfinite(obs))
    layout = env.observation_layout
    assert layout == {"lookback": LOOKBACK, "n_assets": N_ASSETS, "n_features": N_FEATURES,
                      "portfolio_dim": N_ASSETS + 3}


def test_reset_restores_initial_state(make_env) -> None:
    env = make_env(friction=FULL_FRICTION)
    obs0, info0 = env.reset(seed=123)
    for _ in range(10):
        env.step(env.action_space.sample())
    assert env.nav != pytest.approx(info0["nav"])
    obs1, info1 = env.reset(seed=123)
    np.testing.assert_array_equal(obs0, obs1)
    assert info1["nav"] == pytest.approx(1.0e6)
    assert env.cash == pytest.approx(1.0e6)
    assert np.all(env.holdings == 0) and np.all(env.impact_state == 0)
    assert env.peak_nav == pytest.approx(1.0e6) and env.current_drawdown == 0.0
    assert env.reward_fn.returns_history.size == 0
    # Portfolio block of the observation: zero asset weights, full cash, no drawdown / vol.
    port = obs1[-(N_ASSETS + 3):]
    np.testing.assert_allclose(port, [0.0] * N_ASSETS + [1.0, 0.0, 0.0])


def test_random_start_is_seed_reproducible(make_env) -> None:
    env = make_env(episode_length=10, random_start=True)
    starts_a = [env.reset(seed=s)[1]["t"] for s in range(5)]
    starts_b = [env.reset(seed=s)[1]["t"] for s in range(5)]
    assert starts_a == starts_b
    assert all(env.start_index <= s <= env.end_index - 11 for s in starts_a)


def test_episode_truncates_at_length_and_range_end(make_env) -> None:
    env = make_env(episode_length=7)
    env.reset(seed=0)
    for i in range(7):
        _, _, term, trunc, info = env.step(env.action_space.sample())
        assert not term
        assert trunc == (i == 6)
    assert "episode_stats" in info
    env = make_env(episode_length=None)
    env.reset(seed=0)
    steps, done = 0, False
    while not done:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        done, steps = term or trunc, steps + 1
    assert steps == (T - 1) - (LOOKBACK - 1)
    with pytest.raises(RuntimeError):
        env.step(env.action_space.sample())


def test_invalid_range_rejected(market_arrays) -> None:
    features, prices, volumes = market_arrays
    with pytest.raises(ValueError):
        PortfolioEnv(features, prices, volumes, config=EnvConfig(execution="close"), start_index=0)
    with pytest.raises(ValueError):
        PortfolioEnv(features, prices, volumes, config=EnvConfig(execution="close"), start_index=T - 1, end_index=T)
    with pytest.raises(ValueError):  # next-open execution needs open prices
        PortfolioEnv(features, prices, volumes, config=EnvConfig(execution="next_open"))


# --------------------------------------------------------------------- action transforms
@pytest.mark.parametrize("seed", range(20))
def test_softmax_weights_long_only_sum_le_one(make_env, seed: int) -> None:
    env = make_env()
    a = np.random.default_rng(seed).uniform(-1, 1, env.action_space.shape)
    w = env.action_to_weights(a)
    assert np.all(w >= 0) and w.sum() <= 1.0 + 1e-12


@pytest.mark.parametrize("seed", range(20))
def test_projection_transforms_respect_constraints(make_env, seed: int) -> None:
    rng = np.random.default_rng(seed)
    lo = make_env(action_transform="projection", long_only=True, max_gross_leverage=0.8)
    w = lo.action_to_weights(rng.uniform(-1, 1, lo.action_space.shape))
    assert np.all(w >= -1e-12) and w.sum() <= 0.8 + 1e-9
    ls = make_env(action_transform="projection", long_only=False, max_gross_leverage=1.5)
    w = ls.action_to_weights(rng.uniform(-1, 1, ls.action_space.shape))
    assert np.abs(w).sum() <= 1.5 + 1e-9


def test_softmax_rejected_for_long_short(make_env) -> None:
    with pytest.raises(ValueError):
        make_env(long_only=False, action_transform="softmax")


def test_nan_action_is_sanitised(make_env) -> None:
    env = make_env()
    env.reset(seed=0)
    obs, reward, *_ = env.step(np.full(env.action_space.shape, np.nan))
    assert np.isfinite(reward) and np.all(np.isfinite(obs))


def test_projection_helpers() -> None:
    w = project_to_simplex(np.array([0.5, 0.9, -0.2]), 1.0)
    assert w.sum() == pytest.approx(1.0) and np.all(w >= 0)
    np.testing.assert_allclose(project_to_capped_simplex(np.array([0.2, 0.3]), 1.0), [0.2, 0.3])
    v = project_to_l1_ball(np.array([2.0, -2.0, 0.5]), 1.0)
    assert np.abs(v).sum() == pytest.approx(1.0)


def test_weights_to_action_round_trip(make_env) -> None:
    env = make_env()
    target = np.array([0.3, 0.2, 0.1])
    np.testing.assert_allclose(env.action_to_weights(env.weights_to_action(target)), target, atol=1e-6)


# --------------------------------------------------------------------- accounting
def test_frictionless_hold_tracks_asset_return(make_env, market_arrays) -> None:
    _, prices, _ = market_arrays
    env = make_env()
    env.reset(seed=0)
    t0 = env.t
    target = np.array([1.0, 0.0, 0.0])
    _, _, _, _, info = env.step_with_weights(target)
    expected = prices[t0 + 1, 0] / prices[t0, 0] - 1.0
    assert info["portfolio_return"] == pytest.approx(expected, rel=1e-10)
    assert info["costs"]["total"] == 0.0
    # Zero-trade step (hold drifted weights) incurs no turnover / cost.
    _, _, _, _, info = env.step_with_weights(env.current_weights)
    assert info["turnover"] == pytest.approx(0.0, abs=1e-12)
    assert info["costs"]["total"] == pytest.approx(0.0, abs=1e-9)


def test_commission_matches_closed_form(make_env) -> None:
    c_bps = 10.0
    env = make_env(friction={"commission_bps": c_bps})
    env.reset(seed=0)
    nav0 = env.nav
    w_prev = env.current_weights
    target = np.array([0.5, 0.3, 0.1])
    _, _, _, _, info = env.step_with_weights(target)
    # Cost_t = c * sum|dw| * NAV_t, where dw is the realised (post-cost-sizing) change.
    expected = c_bps * 1e-4 * info["turnover"] * nav0
    assert info["costs"]["commission"] == pytest.approx(expected, rel=1e-12)
    assert info["costs"]["total"] == pytest.approx(expected, rel=1e-12)
    # Trades are sized on post-cost NAV, so turnover is the target change up to O(cost).
    assert info["turnover"] == pytest.approx(np.abs(target - w_prev).sum(), rel=1e-3)
    # ... and the post-trade weights hit the target exactly (no implicit leverage).
    np.testing.assert_allclose(env.holdings * env.prices[env.t - 1] / info["nav_post_trade"], target, rtol=1e-9)


def test_accounting_identity_with_all_frictions(make_env) -> None:
    """NAV after trading equals NAV before minus total costs, every step."""
    env = make_env(friction=FULL_FRICTION, action_transform="projection", long_only=False,
                   max_gross_leverage=1.5)
    env.reset(seed=0)
    rng = np.random.default_rng(7)
    for _ in range(40):
        nav_before = env.nav
        cash_before, hold_before = env.cash, env.holdings.copy()
        p_t = env.prices[env.t]
        _, reward, term, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        assert info["nav_pre_trade"] == pytest.approx(nav_before, rel=1e-12)
        assert info["nav_post_trade"] == pytest.approx(nav_before - info["costs"]["total"], rel=1e-10)
        # Cash flow: cash_after_trade = cash_before - traded notional - costs.
        traded = (env.holdings - hold_before) * p_t
        cash_after_trade = env.cash - info["financing"]
        assert cash_after_trade == pytest.approx(cash_before - traded.sum() - info["costs"]["total"], rel=1e-9)
        comps = info["costs"]
        parts = sum(v for k, v in comps.items() if k != "total")
        assert comps["total"] == pytest.approx(parts)
        assert comps["commission"] >= 0 and comps["slippage"] >= 0
        assert comps["temporary_impact"] >= 0 and comps["permanent_impact"] >= 0
        # Marked NAV after the period = holdings at t+1 + cash.
        assert info["nav"] == pytest.approx(env.cash + env.holdings @ env.prices[env.t], rel=1e-12)
        assert np.isfinite(reward)
        if term or trunc:
            break


def test_round_trip_always_costs_money(make_env, market_arrays) -> None:
    """Buying then fully selling on a flat price path loses exactly the friction costs."""
    features, _, volumes = market_arrays
    flat = np.full((T, N_ASSETS), 100.0)
    env = make_env(friction=FULL_FRICTION, prices=flat)
    env.reset(seed=0)
    total_cost = 0.0
    _, _, _, _, info = env.step_with_weights(np.array([0.6, 0.3, 0.1]))
    total_cost += info["costs"]["total"]
    _, _, _, _, info = env.step_with_weights(np.zeros(N_ASSETS))
    total_cost += info["costs"]["total"]
    assert total_cost > 0
    assert env.nav == pytest.approx(1.0e6 - total_cost, rel=1e-10)


def test_square_root_impact_scaling(make_env) -> None:
    """Temporary impact cost scales as |q|^1.5."""
    env = make_env(friction={"temp_impact_coef": 0.3})
    env.reset(seed=0)
    q = np.array([1.0e5, 0.0, 0.0])
    c1, _ = env.compute_trade_costs(q)
    c4, _ = env.compute_trade_costs(4.0 * q)
    assert c4.temporary_impact / c1.temporary_impact == pytest.approx(8.0, rel=1e-10)


def test_permanent_impact_state_and_decay(make_env) -> None:
    env = make_env(friction={"perm_impact_coef": 0.2, "perm_impact_halflife": 2.0})
    env.reset(seed=0)
    env.step_with_weights(np.array([0.5, 0.0, 0.0]))
    s1 = env.impact_state.copy()
    assert s1[0] > 0 and s1[1] == 0
    env.step_with_weights(env.current_weights)  # no trade -> pure decay
    np.testing.assert_allclose(env.impact_state, s1 * 0.5 ** (1 / 2.0), rtol=1e-6, atol=1e-12)


def test_participation_cap_limits_trade(make_env, market_arrays) -> None:
    _, _, volumes = market_arrays
    env = make_env(friction={"max_participation": 0.001})
    env.reset(seed=0)
    _, _, _, _, info = env.step_with_weights(np.array([1.0, 0.0, 0.0]))
    cap = 0.001 * env.adv[env.t - 1, 0]
    traded_dollars = info["turnover"] * info["nav_pre_trade"]
    assert traded_dollars == pytest.approx(cap, rel=1e-9)


def test_short_borrow_fee_charged(make_env) -> None:
    flat = np.full((T, N_ASSETS), 100.0)
    env = make_env(friction={"short_borrow_bps_annual": 252.0}, prices=flat,
                   action_transform="projection", long_only=False)
    env.reset(seed=0)
    _, _, _, _, info = env.step_with_weights(np.array([-0.5, 0.0, 0.0]))
    # 252 bps/yr = 1 bp/day on 0.5 * NAV short notional
    assert info["financing"] == pytest.approx(-0.5 * 1.0e6 * 1e-4, rel=1e-9)


# --------------------------------------------------------------------- robustness
def test_extreme_crash_no_nan(make_env, market_arrays) -> None:
    _, prices, _ = market_arrays
    crashed = prices.copy()
    crashed[LOOKBACK + 2:] *= 1e-4  # -99.99% overnight
    env = make_env(friction=FULL_FRICTION, prices=crashed, reward={"lambda_cvar": 1.0, "lambda_drawdown": 1.0})
    env.reset(seed=0)
    rewards = []
    done = False
    while not done:
        obs, r, term, trunc, info = env.step_with_weights(np.array([0.34, 0.33, 0.33]))
        rewards.append(r)
        assert np.all(np.isfinite(obs)) and np.isfinite(r)
        done = term or trunc
    assert term, "NAV collapse should trigger bankruptcy termination"
    assert min(rewards) >= -env.reward_fn.config.clip


def test_leveraged_short_squeeze_no_nan(make_env, market_arrays) -> None:
    _, prices, _ = market_arrays
    squeezed = prices.copy()
    squeezed[LOOKBACK + 1:, 0] *= 50.0  # +4900% on a short position -> NAV < 0
    env = make_env(prices=squeezed, action_transform="projection", long_only=False, max_gross_leverage=2.0)
    env.reset(seed=0)
    env.step_with_weights(np.array([-2.0, 0.0, 0.0]))
    obs, r, term, trunc, info = env.step_with_weights(np.array([-2.0, 0.0, 0.0]))
    assert np.isfinite(r) and np.all(np.isfinite(obs))
    assert term


def test_observation_has_no_lookahead(make_env, market_arrays) -> None:
    """Changing future data must not change the current observation."""
    features, prices, volumes = market_arrays
    env_a = make_env()
    obs_a, _ = env_a.reset(seed=0, options={"start_index": 30})
    future = features.copy()
    future[31:] += 100.0
    env_b = PortfolioEnv(future, prices, volumes, config=env_a.config, reward_config=env_a.reward_fn.config)
    obs_b, _ = env_b.reset(seed=0, options={"start_index": 30})
    np.testing.assert_array_equal(obs_a, obs_b)


def test_partial_adjustment_and_no_trade_band(make_env) -> None:
    env = make_env(trade_rate=0.25)
    env.reset(seed=0)
    w0 = env.current_weights.copy()  # all cash -> zeros
    target = np.array([0.4, 0.4, 0.0])
    _, _, _, _, info = env.step_with_weights(target)
    np.testing.assert_allclose(info["target_weights"], w0 + 0.25 * (target - w0))
    banded = make_env(trade_rate=1.0, no_trade_band=0.5)
    banded.reset(seed=0)
    _, _, _, _, info = banded.step_with_weights(np.array([0.1, 0.1, 0.1]))  # L1 change 0.3 < band
    assert info["turnover"] == 0.0 and info["costs"]["total"] == 0.0
    with pytest.raises(ValueError):
        make_env(trade_rate=0.0)


# --------------------------------------------------------------------- execution timing
def test_execution_next_open_timeline(make_env, market_arrays) -> None:
    """Decide at close t, fill at open t+1, mark at close t+1; old book carried overnight."""
    _, prices, _ = market_arrays
    env = make_env(execution="next_open")
    env.reset(seed=0)
    t, opens = env.t, env.open_prices
    target = np.array([0.5, 0.3, 0.0])
    _, _, _, _, info = env.step_with_weights(target)  # from all cash: overnight return is 0
    np.testing.assert_allclose(info["execution_prices"], opens[t + 1])
    # Shares are fixed at the decision close (w * NAV / close_t) and filled at the open of t+1.
    expected = float(np.sum(target * (prices[t + 1] - opens[t + 1]) / prices[t]))
    assert info["portfolio_return"] == pytest.approx(expected, rel=1e-10)
    # Holding (None = no order): zero turnover/cost, and the return is the held book's close-to-close move.
    h, cash, nav0, t2 = env.holdings.copy(), env.cash, env.nav, env.t
    _, _, _, _, info = env.step_with_weights(None)
    assert info["turnover"] == 0.0 and info["costs"]["total"] == 0.0
    assert info["portfolio_return"] == pytest.approx((cash + h @ prices[t2 + 1]) / nav0 - 1.0, rel=1e-12)


def test_execution_next_open_accounting(make_env) -> None:
    env = make_env(execution="next_open", friction=FULL_FRICTION)
    env.reset(seed=0)
    rng = np.random.default_rng(3)
    for _ in range(25):
        nav_t = env.nav
        _, _, term, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        assert info["nav_decision"] == pytest.approx(nav_t, rel=1e-12)
        assert info["nav_post_trade"] == pytest.approx(info["nav_pre_trade"] - info["costs"]["total"], rel=1e-10)
        assert info["nav"] == pytest.approx(env.cash + env.holdings @ env.prices[env.t], rel=1e-12)
        if term or trunc:
            break


def test_decision_ignores_information_unavailable_at_order_time() -> None:
    """Perturbing every input from t+1 on (open, high, low, close, volume) leaves day t's
    observation and decision unchanged; only the fill price can differ."""
    from src.data import MarketData, generate_synthetic_market
    from src.features import compute_features

    cfg = {"windows": [5, 21], "zscore_window": 60, "zscore_min_periods": 20}
    data = generate_synthetic_market(n_assets=3, n_periods=200, seed=4)
    t = 120
    k = np.where(np.arange(200) > t, 1.0, 0.0)[:, None]
    shocked = MarketData(data.dates, data.assets, data.open * (1 + 0.3 * k), data.high * (1 + 0.5 * k),
                         data.low * (1 - 0.2 * k), data.close * (1 + 0.4 * k), data.volume * (1 + 2.0 * k))
    w_policy = np.random.default_rng(0).normal(size=4)

    def decide(md: MarketData) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        fs = compute_features(md, cfg)
        env = PortfolioEnv(fs.values, md.close, md.volume, config=EnvConfig(lookback=5, execution="next_open",
                           episode_length=None, random_start=False), open_prices=md.open, start_index=fs.warmup + 4)
        obs, _ = env.reset(options={"start_index": t})
        action = np.tanh(w_policy * obs[:4].mean())  # any deterministic function of the observation
        _, _, _, _, info = env.step(action)
        return obs, info["target_weights"], info["execution_prices"], info["order_shares"]

    obs_a, w_a, px_a, q_a = decide(data)
    obs_b, w_b, px_b, q_b = decide(shocked)
    np.testing.assert_array_equal(obs_a, obs_b)
    np.testing.assert_array_equal(w_a, w_b)
    np.testing.assert_array_equal(q_a, q_b)  # the order itself (shares) is fixed at submission time
    assert not np.allclose(px_a, px_b)  # the fill happens at t+1 and does see the shock


def test_buy_and_hold_does_not_retrade_under_next_open(make_env) -> None:
    from src.backtest import run_episode
    from src.baselines import EqualWeightStrategy

    rec = run_episode(make_env(execution="next_open", friction={"commission_bps": 1.0}),
                      weights_fn=EqualWeightStrategy(rebalance_every=None))
    assert rec["turnover"].iloc[0] > 0.9 and (rec["turnover"].iloc[1:] == 0).all()


def test_gross_equals_net_plus_cost_under_next_open(make_env) -> None:
    env = make_env(execution="next_open", friction=FULL_FRICTION)
    env.reset(seed=0)
    nav_t = env.nav
    _, _, _, _, info = env.step_with_weights(np.array([0.5, 0.3, 0.2]))
    gross_nav = env.nav + info["costs"]["total"] * 1.0  # cash earns 0, so costs reduce NAV one-for-one
    assert info["portfolio_return"] + info["cost"] == pytest.approx(gross_nav / nav_t - 1.0, rel=1e-9)


# --------------------------------------------------------------------- next-open verification suite
def _tiny_env(close, open_, commission_bps=0.0, slippage_bps=0.0, **kw):
    """Hand-built 2-asset market; features are irrelevant (zeros), lookback 1."""
    close = np.asarray(close, float)
    open_ = np.asarray(open_, float)
    T, N = close.shape
    fr = FrictionConfig(commission_bps=commission_bps, slippage_bps=slippage_bps, temp_impact_coef=0.0,
                        perm_impact_coef=0.0, max_participation=None, short_borrow_bps_annual=0.0)
    cfg = EnvConfig(initial_nav=1_000_000.0, lookback=1, episode_length=None, random_start=False,
                    execution="next_open", friction=fr, **kw)
    return PortfolioEnv(np.zeros((T, N, 1), np.float32), close, np.full((T, N), 1e9), config=cfg,
                        start_index=0, open_prices=open_)


def test_manual_two_asset_example() -> None:
    """Every number below is computed by hand (plain arithmetic), not by environment code."""
    close = [[100.0, 50.0], [102.0, 49.0], [101.0, 50.0]]
    open_ = [[100.0, 50.0], [101.0, 49.5], [102.5, 48.8]]
    env = _tiny_env(close, open_, commission_bps=10.0)
    env.reset()
    nav0, c = 1_000_000.0, 0.0010
    # Day 0 close: decide w = (0.6, 0.3). Post-cost sizing at decision prices:
    # NAV_eff = NAV0 / (1 + c * 0.9); shares = w * NAV_eff / close_0.
    nav_eff = nav0 / (1 + c * 0.9)
    sh = np.array([0.6 * nav_eff / 100.0, 0.3 * nav_eff / 50.0])
    # Fill at day-1 open: notional = shares * open_1; commission on executed notional.
    notional = sh * np.array([101.0, 49.5])
    commission = c * notional.sum()
    cash = nav0 - notional.sum() - commission
    nav1 = cash + sh @ np.array([102.0, 49.0])
    _, _, _, _, info = env.step_with_weights(np.array([0.6, 0.3]))
    np.testing.assert_allclose(info["order_shares"], sh, rtol=1e-12)
    assert info["costs"]["commission"] == pytest.approx(commission, rel=1e-12)
    assert env.cash == pytest.approx(cash, abs=1e-3)          # to a tenth of a cent
    assert info["nav"] == pytest.approx(nav1, abs=1e-3)
    assert info["portfolio_return"] == pytest.approx(nav1 / nav0 - 1.0, rel=1e-12)
    # Day 1 close: hold. Overnight to day-2 open then intraday to day-2 close; no order, no cost.
    nav_open2 = cash + sh @ np.array([102.5, 48.8])
    nav2 = cash + sh @ np.array([101.0, 50.0])
    _, _, _, _, info = env.step_with_weights(None)
    assert info["nav_pre_trade"] == pytest.approx(nav_open2, abs=1e-3)  # overnight move recognised
    assert info["nav"] == pytest.approx(nav2, abs=1e-3)
    assert info["costs"]["total"] == 0.0 and info["portfolio_return"] == pytest.approx(nav2 / nav1 - 1.0, rel=1e-12)


def test_overnight_gap_applies_to_existing_positions(make_env) -> None:
    env = make_env(execution="next_open")
    env.reset(seed=0)
    env.step_with_weights(np.array([0.5, 0.5, 0.0]))
    h, cash, t = env.holdings.copy(), env.cash, env.t
    _, _, _, _, info = env.step_with_weights(None)
    assert info["nav_pre_trade"] == pytest.approx(cash + h @ env.open_prices[t + 1], rel=1e-12)
    assert info["nav"] == pytest.approx(cash + h @ env.prices[t + 1], rel=1e-12)


def test_costs_charged_on_executed_quantities(make_env) -> None:
    env = make_env(execution="next_open", friction={"commission_bps": 7.0, "slippage_bps": 3.0})
    env.reset(seed=0)
    t = env.t
    _, _, _, _, info = env.step_with_weights(np.array([0.4, 0.3, 0.2]))
    executed = np.abs(info["order_shares"] * env.open_prices[t + 1]).sum()
    assert info["costs"]["commission"] == pytest.approx(7e-4 * executed, rel=1e-12)
    assert info["costs"]["slippage"] == pytest.approx(3e-4 * executed, rel=1e-12)
    assert info["turnover"] == pytest.approx(executed / info["nav_pre_trade"], rel=1e-12)


def test_cash_and_weights_consistent_under_gaps(make_env, market_arrays) -> None:
    _, prices, _ = market_arrays
    gapped_open = np.vstack([prices[:1], prices[:-1]]) * (1 + 0.03 * np.cos(np.arange(prices.shape[0])))[:, None]
    env = make_env(execution="next_open", friction=FULL_FRICTION, open_prices=gapped_open)
    env.reset(seed=0)
    rng = np.random.default_rng(5)
    for _ in range(40):
        _, _, term, trunc, info = env.step(rng.uniform(-1, 1, env.action_space.shape))
        w = info["weights"]
        assert info["nav"] == pytest.approx(env.cash + env.holdings @ env.prices[env.t], rel=1e-12)
        assert w.sum() + info["cash_weight"] == pytest.approx(1.0, abs=1e-12)
        assert np.all(env.holdings >= -1e-9)          # long-only book never goes short
        assert info["gross_exposure"] <= 1.0 + 0.10   # fixed-share orders: leverage bounded by the gap size
        if term or trunc:
            break
