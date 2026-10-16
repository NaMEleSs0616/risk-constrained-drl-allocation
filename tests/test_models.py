"""Tests for the policy feature extractor and PPO wiring."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from stable_baselines3 import PPO

from src.models import RiskAwareFeatureExtractor, build_policy_kwargs


@pytest.mark.parametrize("encoder", ["cnn", "lstm", "mlp"])
@pytest.mark.parametrize("use_attention", [True, False])
def test_extractor_output_shape(make_env, encoder: str, use_attention: bool) -> None:
    env = make_env()
    ext = RiskAwareFeatureExtractor(
        env.observation_space, **env.observation_layout, encoder=encoder, hidden_dim=16,
        n_heads=4, use_attention=use_attention, features_dim=24,
    )
    obs = torch.as_tensor(np.stack([env.reset(seed=s)[0] for s in range(3)]))
    out = ext(obs)
    assert out.shape == (3, 24) and torch.isfinite(out).all()


def test_cnn_encoder_is_causal(make_env) -> None:
    """Perturbing only the last time step must leave earlier-step-only encodings unchanged."""
    from src.models import TemporalEncoder

    enc = TemporalEncoder(n_features=4, lookback=10, hidden_dim=8, encoder="cnn")
    x = torch.randn(2, 10, 4)
    full = enc.net(x.transpose(1, 2))
    x2 = x.clone()
    x2[:, -1] += 10.0
    full2 = enc.net(x2.transpose(1, 2))
    torch.testing.assert_close(full[:, :, :-1], full2[:, :, :-1])


def test_layout_mismatch_raises(make_env) -> None:
    env = make_env()
    layout = dict(env.observation_layout, n_features=env.n_features + 1)
    with pytest.raises(ValueError):
        RiskAwareFeatureExtractor(env.observation_space, **layout)


def test_ppo_learns_one_iteration(make_env) -> None:
    env = make_env(episode_length=16, random_start=True, friction={"commission_bps": 1.0})
    cfg = {"encoder": "cnn", "hidden_dim": 8, "n_heads": 2, "features_dim": 16,
           "net_arch": {"pi": [16], "vf": [16]}}
    model = PPO("MlpPolicy", env, policy_kwargs=build_policy_kwargs(cfg, env.observation_layout),
                n_steps=32, batch_size=16, n_epochs=1, seed=0, device="cpu")
    model.learn(total_timesteps=32)
    action, _ = model.predict(env.reset(seed=0)[0], deterministic=True)
    assert env.action_space.contains(action.astype(np.float32))
