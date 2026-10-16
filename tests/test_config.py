"""Tests for configuration loading and overrides."""

from __future__ import annotations

from src.environment import EnvConfig
from src.reward import RewardConfig
from src.utils import deep_update, load_config, parse_overrides, yaml_load


def test_scientific_notation_parses_as_float() -> None:
    # PyYAML's YAML 1.1 resolver would return the string "1.0e4" here.
    assert yaml_load("a: 1.0e4\nb: 3e-4\nc: 1e7\nd: '1e3'\ne: 5") == {"a": 1e4, "b": 3e-4, "c": 1e7, "d": "1e3", "e": 5}


def test_default_config_builds_typed_objects() -> None:
    cfg = load_config()
    env_cfg = EnvConfig.from_dict(cfg["environment"])
    assert isinstance(env_cfg.initial_nav, float) and isinstance(env_cfg.friction.min_adv, float)
    assert isinstance(cfg["ppo"]["learning_rate"], float)
    RewardConfig.from_dict(cfg["reward"])
    variants = cfg["experiments"]["variants"]
    assert {k: (v["lambda_cvar"], v["lambda_drawdown"]) for k, v in variants.items()} == {
        "A_vanilla": (0.0, 0.0),
        "B_cvar_only": (1.0, 0.0),
        "C_drawdown_only": (0.0, 1.0),
        "D_full": (1.0, 1.0),
        "E_full_calibrated": (0.05, 0.5),
    }


def test_overrides_merge_nested_values() -> None:
    over = parse_overrides(["ppo.learning_rate=1e-4", "environment.friction.commission_bps=2", "model.net_arch.pi=[32]"])
    cfg = deep_update(load_config(), over)
    assert cfg["ppo"]["learning_rate"] == 1e-4
    assert cfg["environment"]["friction"]["commission_bps"] == 2
    assert cfg["environment"]["friction"]["slippage_bps"] == 2.0  # untouched sibling kept
    assert cfg["model"]["net_arch"] == {"pi": [32], "vf": [128, 64]}
