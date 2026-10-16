"""Policy feature extractor: shared per-asset temporal encoder + cross-asset attention.

The flat observation produced by :class:`src.environment.PortfolioEnv` is un-flattened
into a ``(B, L, N, F)`` feature window and a ``(B, P)`` portfolio-state vector.

1. Each asset's ``(L, F)`` history is encoded by the *same* temporal encoder (causal
   dilated 1D CNN, LSTM or MLP), giving parameter sharing across assets.
2. A learned asset-identity embedding is added (assets are not exchangeable).
3. Optional multi-head self-attention mixes information across assets
   (correlation / relative-value structure).
4. Asset embeddings are concatenated with an embedding of the portfolio state
   (weights, drawdown, realised vol) and fused into the features consumed by the
   PPO actor and critic heads.
"""

from __future__ import annotations

from typing import Any, Mapping

import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {"tanh": nn.Tanh, "relu": nn.ReLU, "gelu": nn.GELU}


class CausalConv1d(nn.Module):
    """1D convolution with left padding so output ``t`` depends only on inputs ``<= t``."""

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int) -> None:
        """Initialise the layer.

        Args:
            in_ch: Input channels.
            out_ch: Output channels.
            kernel_size: Kernel width.
            dilation: Dilation factor.
        """
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the convolution to ``(B, C, L)`` input."""
        return self.conv(nn.functional.pad(x, (self.pad, 0)))


class TemporalEncoder(nn.Module):
    """Encode a ``(B, L, F)`` sequence into a ``(B, H)`` embedding."""

    def __init__(
        self,
        n_features: int,
        lookback: int,
        hidden_dim: int = 32,
        encoder: str = "cnn",
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
    ) -> None:
        """Initialise the encoder.

        Args:
            n_features: Input features per step ``F``.
            lookback: Sequence length ``L``.
            hidden_dim: Output embedding size ``H``.
            encoder: ``"cnn"`` (causal dilated TCN), ``"lstm"`` or ``"mlp"``.
            n_layers: Number of conv / LSTM / MLP layers.
            kernel_size: Conv kernel width (CNN only).
            dropout: Dropout probability.

        Raises:
            ValueError: On an unknown encoder type.
        """
        super().__init__()
        self.kind = encoder
        if encoder == "cnn":
            layers: list[nn.Module] = []
            ch = n_features
            for i in range(n_layers):
                layers += [CausalConv1d(ch, hidden_dim, kernel_size, dilation=2**i), nn.GELU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
                ch = hidden_dim
            self.net = nn.Sequential(*layers)
            self.skip = nn.Conv1d(n_features, hidden_dim, 1)
        elif encoder == "lstm":
            self.net = nn.LSTM(
                n_features,
                hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
        elif encoder == "mlp":
            dims = [lookback * n_features] + [hidden_dim] * n_layers
            layers = []
            for a, b in zip(dims[:-1], dims[1:]):
                layers += [nn.Linear(a, b), nn.GELU()]
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))
            self.net = nn.Sequential(*layers)
        else:
            raise ValueError(f"Unknown encoder '{encoder}' (expected cnn | lstm | mlp)")
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ``(B, L, F)`` -> ``(B, H)``."""
        if self.kind == "cnn":
            xc = x.transpose(1, 2)  # (B, F, L)
            h = self.net(xc) + self.skip(xc)
            out = h[:, :, -1]  # causal: last step summarises the whole window
        elif self.kind == "lstm":
            _, (h_n, _) = self.net(x)
            out = h_n[-1]
        else:
            out = self.net(x.flatten(1))
        return self.norm(out)


class CrossAssetAttention(nn.Module):
    """Pre-norm transformer block over the asset dimension."""

    def __init__(self, dim: int, n_heads: int, dropout: float = 0.0) -> None:
        """Initialise the block.

        Args:
            dim: Embedding size (must be divisible by ``n_heads``).
            n_heads: Attention heads.
            dropout: Dropout probability.
        """
        super().__init__()
        if dim % n_heads != 0:
            raise ValueError(f"hidden_dim={dim} must be divisible by n_heads={n_heads}")
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention to ``(B, N, H)``."""
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        return x + self.ff(self.norm2(x))


class RiskAwareFeatureExtractor(BaseFeaturesExtractor):
    """SB3 features extractor combining market history and portfolio state."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        lookback: int,
        n_assets: int,
        n_features: int,
        portfolio_dim: int,
        encoder: str = "cnn",
        hidden_dim: int = 32,
        n_layers: int = 2,
        kernel_size: int = 3,
        dropout: float = 0.0,
        use_attention: bool = True,
        n_heads: int = 4,
        features_dim: int = 128,
    ) -> None:
        """Initialise the extractor.

        Args:
            observation_space: Flat observation space of the environment.
            lookback: Feature window length ``L``.
            n_assets: Number of assets ``N``.
            n_features: Features per asset ``F``.
            portfolio_dim: Size of the trailing portfolio-state block ``P``.
            encoder: Temporal encoder type.
            hidden_dim: Per-asset embedding size.
            n_layers: Encoder depth.
            kernel_size: CNN kernel width.
            dropout: Dropout probability.
            use_attention: Whether to add cross-asset self-attention.
            n_heads: Attention heads.
            features_dim: Output dimension fed to the actor / critic heads.

        Raises:
            ValueError: If the observation size does not match the layout.
        """
        super().__init__(observation_space, features_dim)
        expected = lookback * n_assets * n_features + portfolio_dim
        if observation_space.shape[0] != expected:
            raise ValueError(f"observation dim {observation_space.shape[0]} != layout dim {expected}")
        self.L, self.N, self.F, self.P = lookback, n_assets, n_features, portfolio_dim
        self.market_dim = lookback * n_assets * n_features

        self.encoder = TemporalEncoder(n_features, lookback, hidden_dim, encoder, n_layers, kernel_size, dropout)
        self.asset_embedding = nn.Embedding(n_assets, hidden_dim)
        self.attention = CrossAssetAttention(hidden_dim, n_heads, dropout) if use_attention else None
        port_dim = max(16, hidden_dim)
        self.portfolio_mlp = nn.Sequential(nn.Linear(portfolio_dim, port_dim), nn.GELU())
        self.fusion = nn.Sequential(
            nn.Linear(n_assets * hidden_dim + port_dim, features_dim),
            nn.GELU(),
            nn.LayerNorm(features_dim),
        )
        self.register_buffer("_asset_ids", torch.arange(n_assets), persistent=False)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Map flat observations ``(B, D)`` to features ``(B, features_dim)``."""
        B = observations.shape[0]
        market = observations[:, : self.market_dim].reshape(B, self.L, self.N, self.F)
        port = observations[:, self.market_dim :]
        seq = market.permute(0, 2, 1, 3).reshape(B * self.N, self.L, self.F)
        h = self.encoder(seq).reshape(B, self.N, -1)
        h = h + self.asset_embedding(self._asset_ids)[None]
        if self.attention is not None:
            h = self.attention(h)
        fused = torch.cat([h.flatten(1), self.portfolio_mlp(port)], dim=1)
        return self.fusion(fused)


def build_policy_kwargs(model_cfg: Mapping[str, Any], layout: Mapping[str, int]) -> dict[str, Any]:
    """Assemble ``policy_kwargs`` for SB3's ``PPO("MlpPolicy", ...)``.

    Args:
        model_cfg: The ``model`` config section.
        layout: ``PortfolioEnv.observation_layout``.

    Returns:
        Keyword arguments for the actor-critic policy.
    """
    activation = str(model_cfg.get("activation", "tanh")).lower()
    if activation not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{activation}'")
    net_arch = model_cfg.get("net_arch", {"pi": [128, 64], "vf": [128, 64]})
    return {
        "features_extractor_class": RiskAwareFeatureExtractor,
        "features_extractor_kwargs": {
            **layout,
            "encoder": model_cfg.get("encoder", "cnn"),
            "hidden_dim": int(model_cfg.get("hidden_dim", 32)),
            "n_layers": int(model_cfg.get("n_layers", 2)),
            "kernel_size": int(model_cfg.get("kernel_size", 3)),
            "dropout": float(model_cfg.get("dropout", 0.0)),
            "use_attention": bool(model_cfg.get("use_attention", True)),
            "n_heads": int(model_cfg.get("n_heads", 4)),
            "features_dim": int(model_cfg.get("features_dim", 128)),
        },
        "net_arch": dict(pi=list(net_arch["pi"]), vf=list(net_arch["vf"])),
        "activation_fn": _ACTIVATIONS[activation],
        "log_std_init": float(model_cfg.get("log_std_init", -0.5)),
        "ortho_init": bool(model_cfg.get("ortho_init", True)),
        "share_features_extractor": bool(model_cfg.get("share_features_extractor", True)),
    }
