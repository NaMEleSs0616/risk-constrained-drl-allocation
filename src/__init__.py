"""Risk-constrained dynamic portfolio allocation with PPO.

Modules:
    data:        Market data containers, CSV loading and a regime-switching synthetic generator.
    features:    Causal multi-timeframe technical features and order-flow imbalance proxies.
    reward:      CVaR / drawdown risk measures and the risk-adjusted reward.
    environment: Gymnasium portfolio environment with commission, slippage and sqrt market impact.
    models:      Per-asset temporal encoder + cross-asset attention feature extractor for SB3.
    train:       PPO training with TensorBoard / W&B logging and checkpointing.
    metrics:     Performance metrics and table formatting.
    baselines:   Equal-weight, Ledoit-Wolf mean-variance and equal-risk-contribution strategies.
    backtest:    Walk-forward out-of-sample evaluation.
"""

__version__ = "0.1.0"
