"""Shared utilities: configuration loading, overrides, seeding and logging."""

from __future__ import annotations

import copy
import logging
import random
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "hyperparameters.yaml"

#: Overrides for a fast end-to-end smoke run (``--smoke``): tiny data and training budgets.
SMOKE_OVERRIDES: dict[str, Any] = {
    "data": {"synthetic": {"n_periods": 1200}},
    "features": {"zscore_window": 126, "zscore_min_periods": 42},
    "ppo": {"n_envs": 2, "n_steps": 128, "batch_size": 64, "n_epochs": 2, "total_timesteps": 512},
    "training": {"eval_freq": 256, "checkpoint_freq": 512, "logger": "tensorboard"},
    "backtest": {"train_window": 400, "test_window": 120, "step": 120, "timesteps_per_fold": 512},
    "experiments": {"train_window": 400, "test_window": 120, "timesteps": 512, "seeds": [0, 1], "n_workers": 2},
}


class _ConfigLoader(yaml.SafeLoader):
    """SafeLoader that also parses ``1e-4`` / ``1.0e7`` (YAML 1.1 requires a dot and a sign)."""


_ConfigLoader.add_implicit_resolver(
    "tag:yaml.org,2002:float",
    re.compile(
        r"""^(?:[-+]?(?:[0-9][0-9_]*)\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\.[0-9_]+(?:[eE][-+]?[0-9]+)?
        |[-+]?\.(?:inf|Inf|INF)
        |\.(?:nan|NaN|NAN))$""",
        re.X,
    ),
    list("-+0123456789."),
)


def yaml_load(text: str) -> Any:
    """Parse YAML text with scientific-notation-aware float resolution."""
    return yaml.load(text, Loader=_ConfigLoader)  # noqa: S506 - SafeLoader subclass


def deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge ``updates`` into a copy of ``base``.

    Args:
        base: Original nested dictionary (left untouched).
        updates: Values to merge; nested mappings are merged key by key.

    Returns:
        A new dictionary containing the merged configuration.
    """
    merged = copy.deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = deep_update(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def parse_overrides(pairs: Iterable[str]) -> dict[str, Any]:
    """Parse ``section.key=value`` strings into a nested override dictionary.

    Values are parsed with YAML so that ``1e-4``, ``true``, ``[1, 2]`` and ``null``
    become the expected Python types.

    Args:
        pairs: Strings such as ``"ppo.learning_rate=1e-4"``.

    Returns:
        Nested dictionary suitable for :func:`deep_update`.

    Raises:
        ValueError: If a pair does not contain ``=``.
    """
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must have the form key.subkey=value")
        dotted, raw = pair.split("=", 1)
        node = overrides
        keys = dotted.strip().split(".")
        for key in keys[:-1]:
            node = node.setdefault(key, {})
        node[keys[-1]] = yaml_load(raw)
    return overrides


def load_config(
    path: str | Path | None = None, overrides: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Load the YAML configuration and apply optional overrides.

    Args:
        path: Path to the YAML file; defaults to ``config/hyperparameters.yaml``.
        overrides: Nested overrides merged on top of the file contents.

    Returns:
        The configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file does not exist.
    """
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml_load(fh.read()) or {}
    if overrides:
        cfg = deep_update(cfg, overrides)
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    """Write a configuration dictionary to YAML.

    Args:
        cfg: Configuration to persist.
        path: Destination file path; parent directories are created.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False)


def set_global_seed(seed: int) -> None:
    """Seed Python, NumPy and (if installed) PyTorch RNGs.

    Args:
        seed: Seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:  # pragma: no cover - torch is a hard dependency in practice
        pass


def configure_logging(level: int | str = logging.INFO) -> None:
    """Configure root logging with a concise, timestamped format.

    Args:
        level: Logging level name or number.
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module logger.

    Args:
        name: Logger name (usually ``__name__``).

    Returns:
        The logger instance.
    """
    return logging.getLogger(name)


def resolve_path(path: str | Path, root: Path = PROJECT_ROOT) -> Path:
    """Resolve a possibly-relative path against the project root.

    Args:
        path: Absolute or project-relative path.
        root: Base directory for relative paths.

    Returns:
        Absolute path.
    """
    p = Path(path)
    return p if p.is_absolute() else root / p
