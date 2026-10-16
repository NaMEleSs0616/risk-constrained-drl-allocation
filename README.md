# Risk-Constrained Dynamic Portfolio Allocation with PPO

A modular framework for training PPO agents (Stable-Baselines3) to allocate a multi-asset portfolio. The reward is penalised by tail risk (CVaR and drawdown), and the simulator charges commissions, slippage, and square-root market impact.

```
config/hyperparameters.yaml   all settings (universe, features, frictions, reward, model, PPO, splits, experiments)
src/
  data.py          MarketData container, CSV loader, regime-switching fat-tailed synthetic generator
  features.py      causal multi-timeframe features + order-flow imbalance (BVC & CLV proxies)
  reward.py        historical VaR/CVaR (Rockafellar-Uryasev), drawdown, RiskAdjustedReward
  environment.py   Gymnasium env: weights projection, commission/slippage/temp+perm sqrt impact
  models.py        shared per-asset temporal encoder (causal TCN / LSTM / MLP) + cross-asset attention
  train.py         PPO training, VecNormalize, TensorBoard/W&B risk metrics, checkpoints, val selection
  metrics.py       Sharpe/Sortino/MaxDD/Calmar/turnover/costs with NaN- and zero-division-safe handling
  baselines.py     1/N (rebalanced & buy-and-hold), Ledoit-Wolf MVO, Equal Risk Contribution
  backtest.py      walk-forward folds, deterministic OOS runners, stitching, equity plots
scripts/run_experiments.py    lambda ablation (models A-E) x seeds + baselines, 3-fold rolling walk-forward (parallel)
scripts/checkpoint_analysis.py selection-free check: every agent at 50k/100k/150k steps + untrained-policy control
scripts/make_figures.py        paper figures from outputs/  ->  paper/figures/
scripts/build_paper.py         fills paper tables from outputs/ CSVs  ->  paper/paper.html
main.py                       CLI: train | evaluate | backtest | experiments
tests/                        pytest suite (env accounting, reward/CVaR, metrics, baselines, model)
```

## Quick start

```bash
pip install -r requirements.txt
pytest                                   # full test suite
python main.py experiments --smoke       # ~30 s end-to-end pipeline check
python scripts/run_experiments.py        # full ablation study -> outputs/
tensorboard --logdir outputs/models      # training curves
```

Other commands:

```bash
python main.py train                                  # train on train split, select on val split
python main.py evaluate                               # test split vs. baselines
python main.py backtest                               # expanding walk-forward, re-train per fold
python main.py train --set ppo.total_timesteps=1e6 model.encoder=lstm reward.lambda_cvar=0.1
python scripts/run_experiments.py --variants A_vanilla D_full --timesteps 300000 --n-folds 3
```

To use real data, set `data.source: csv` and `data.csv_path`. The CSV must be in long format with columns `date, ticker, close, volume`; `open`, `high` and `low` are optional.

## Design notes

**Timing and look-ahead.** The agent observes features up to the close of `t`, trades at the close of `t` and is marked at `t+1`. Features use only causal rolling windows, including the z-score normalisation, so one feature tensor can be shared across folds. Walk-forward folds enforce `train_end + embargo <= test_start`. Tests check that changing future prices or features never changes the current observation.

**Frictions.** For each trade of `q` dollars the env charges:

- commission `c|q|` and half-spread `s|q|`;
- temporary impact `η σ √(|q|/ADV) |q|`;
- permanent impact: the trade shifts a per-asset impact state by `γ σ √(|q|/ADV)`. The trade pays half of that shift, and later trades pay or recoup the remaining shift as it decays with a half-life.

Positions are marked at un-impacted prices, so impact never shows up as a paper gain. Trades are sized on the NAV after costs, so post-trade weights match the target exactly with no hidden leverage. An optional participation cap limits each trade to a fraction of ADV. Every step enforces `NAV_after_trade = NAV_before − total_cost`.

**Reward.** `R_t = scale · (r_t − λ₁·CVaR_α(window) − λ₂·D_t²)`. Inputs are sanitised and the result clipped, so the reward stays finite for −100% moves, short squeezes (NAV < 0) and NaN or inf inputs.

**Lambda scale in the ablation.** CVaR here is a *daily* loss, while the return term is a single day's return. In a 30k-step check run, model D (`λ₁ = λ₂ = 1`) averaged a CVaR penalty of about 0.023 and a drawdown penalty of about 0.004 per step, against a mean daily return of about 0.0007. Its episode reward was about −700, compared with +26 for model A. Because the penalties dominate the return by roughly 30×, the gradient pushes B and D toward cash. At 30k steps their cash weight had only moved from 16.5% to 18%, so the size of that shift over a full training run is not yet measured. If you want the penalised models to trade off risk against return, try `λ₁ ∈ [0.02, 0.1]` and `λ₂ ∈ [0.1, 1]` (the defaults in the `reward` section).
