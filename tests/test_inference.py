"""Tests for the two-level bootstrap, Holm correction, TOST and MDE."""

from __future__ import annotations

import numpy as np
import pytest

from src.inference import holm, minimum_detectable_effect, tost_equivalence, two_level_bootstrap


def _seeds(mu: float, sd: float, k: int, n: int = 1500, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 0.01, n)
    return [mu + common + rng.normal(0, sd, n) for _ in range(k)]


def test_identical_strategies_give_zero_difference() -> None:
    a = _seeds(0.0004, 0.002, 3)
    c = two_level_bootstrap(a, a, n_boot=300)
    assert c.estimate == pytest.approx(0.0) and c.p_value > 0.5 and c.ci_low < 0 < c.ci_high


def test_detects_large_shift_and_seed_noise_widens_interval() -> None:
    base = _seeds(0.0, 0.001, 1, seed=1)
    better = _seeds(0.0015, 0.001, 5, seed=1)
    c = two_level_bootstrap(better, base, n_boot=400)
    assert c.estimate > 0 and c.ci_low > 0 and c.p_value < 0.01
    noisy = [x + s for x, s in zip(better, np.linspace(-0.001, 0.001, 5))]  # seed-level dispersion
    c2 = two_level_bootstrap(noisy, base, n_boot=400)
    assert (c2.ci_high - c2.ci_low) > (c.ci_high - c.ci_low)


def test_other_metrics_run_and_turnover_requires_series() -> None:
    a, b = _seeds(0.0003, 0.002, 3, seed=2), _seeds(0.0, 0.0, 1, seed=3)
    to_a = [np.full(1500, 0.4)] * 3
    to_b = [np.full(1500, 0.01)]
    for m in ("ann_return_pct", "maxdd_pct", "cvar95_pct"):
        assert np.isfinite(two_level_bootstrap(a, b, m, n_boot=100).estimate)
    t = two_level_bootstrap(a, b, "turnover_pct_day", to_a, to_b, n_boot=100)
    assert t.estimate == pytest.approx(39.0)


def test_holm_matches_hand_computation() -> None:
    adj = holm([0.01, 0.04, 0.03, 0.20])
    assert adj == pytest.approx([0.04, 0.09, 0.09, 0.20])


def test_tost_and_mde() -> None:
    a = _seeds(0.0002, 0.0005, 5, seed=4)
    c = two_level_bootstrap(a, a[:1], n_boot=400)
    assert tost_equivalence(c, margin=1.0) == "equivalent"
    assert tost_equivalence(c, margin=1e-6) in ("inconclusive", "different")
    assert minimum_detectable_effect(0.1) == pytest.approx(0.2802, abs=1e-3)
