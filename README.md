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
  baselines.py     1/N (rebalanced & buy-and-hold), Ledoit-Wolf MVO, ERC, HRP, vol-target overlay
  inference.py     two-level (seed x block) bootstrap, Holm, TOST equivalence, minimum detectable effect
  backtest.py      walk-forward folds, deterministic OOS runners, stitching, equity plots
scripts/run_experiments.py    lambda ablation (models A-E) x seeds + baselines, 3-fold rolling walk-forward (parallel)
scripts/checkpoint_analysis.py selection-free check: every agent at 50k/100k/150k steps + untrained-policy control
scripts/make_figures.py        paper figures from outputs/  ->  paper/figures/
scripts/build_paper.py         fills paper tables from results CSVs, numbers tables/figures -> paper/paper.html
scripts/download_data.py       daily adjusted OHLCV for the 10-ETF universe (Yahoo Finance) -> data/
scripts/run_real_experiments.py  14-fold warm-started walk-forward study on ETFs (A, D, E, T, RCPO, overlay)
scripts/extra_analysis.py      gross-of-cost, tracking, timing-vs-tilt, de-risking, exposure-matched 1/N, RCPO paths
scripts/compare_studies.py     paired tests: follow-up variants vs their original-run counterparts
scripts/build_latex.py         HTML paper -> paper/latex/main.tex (+ figures); compile with `tectonic main.tex`
scripts/inference_report.py    pre-specified two-level bootstrap tests (Holm, TOST, MDE) -> inference_*.csv
scripts/sensitivity_analysis.py evaluation-only robustness: costs, impact, fill delay, starting book, reward clipping
scripts/warmstart_control.py   warm-started chain vs from-scratch retraining on selected folds
scripts/run_robustness.sh      simulated-market replications (next-open) + warm-start control
scripts/synthetic_robustness_summary.py  summary and tests across the simulated markets
scripts/make_manifest.py       release manifest: data hash + SHA-256 of every committed output
main.py                       CLI: train | evaluate | backtest | experiments
tests/                        pytest suite (env accounting, reward/CVaR, metrics, baselines, model)
```

## Quick start

```bash
pip install -r requirements-lock.txt     # exact versions used for v6 (or requirements.txt)
pytest
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

## Headline results (version 6, tradable next-open execution)

On 10 liquid multi-asset ETFs over 14 walk-forward years (Oct 2012 to Oct 2026), net of commission, spread and square-root impact, 5 training seeds per design, decide at the close of `t` and fill at the open of `t+1`. The five primary tests were fixed in [`ANALYSIS_PLAN.md`](ANALYSIS_PLAN.md) before the rerun (Holm-corrected two-level bootstrap):

| ID | Comparison (net Sharpe) | Estimate [95% CI] | Verdict |
|----|-------------------------|-------------------|---------|
| P1 | penalised E − vanilla A | +0.06 [−0.12, 0.26] | not significant, inconclusive |
| P2 | smoothed S − vanilla A | +0.49 [0.27, 0.74] | significant |
| P3 | smoothed + penalised SE − 1/N | −0.08 [−0.23, 0.06] | not significant, not equivalent |
| P4 | vanilla A − 1/N | −0.53 [−0.79, −0.32] | significant |
| P5 | gross A − gross 1/N (timing edge) | −0.09 [−0.30, 0.12] | not significant: **timing claim withdrawn** |

- Tail-risk penalties lower CVaR and turnover but not the Sharpe ratio; 1/N de-levered to the same cash matches their drawdowns. No agent de-risks when volatility rises.
- Partial-adjustment execution is the only design choice with a significant effect.
- The untrained policy (equal weight on assets and cash) is as good as any trained agent.
- **Correction:** versions 1–5 filled orders at the same close the features used, which is not tradable. That convention created the apparent pre-cost timing edge they reported. The legacy outputs (`outputs/`, `outputs_real/`, `outputs_real_followup/`) and paper (`paper/legacy/`) are kept, with configs pinned to `execution: close`.

Robustness (`outputs_real_nextopen/sensitivity_*.csv`, `warmstart_control/`, `outputs_syn_nextopen/`): costs ×0–×3, no or tripled impact, an extra session of fill delay, an inherited starting book, warm-start vs from-scratch training, four independent simulated markets including a bear market, and reward-clipping frequency. None changes a conclusion: no agent beats 1/N even with every cost removed, from-scratch training is no better than warm-started training (Sharpe −0.12 [−0.38, 0.19]), and in every simulated market PPO-S is indistinguishable from 1/N and the effect of the penalties changes sign. The write-up is `paper/paper.html`, with a PDF in `paper/latex/`.

## Design notes

**Timing and look-ahead.** `environment.execution` selects the fill convention. `next_open` (default for the v6 studies): the agent observes features up to the close of `t`; the order is a vector of share quantities fixed at that close (NAV, weights, volatility and ADV known at `t`), filled at the open of `t+1`; the old book carries the overnight gap; the position is marked at the close of `t+1`. `next_close` adds another session of delay. `close` is the legacy same-close fill, which is not tradable and is kept only to reproduce versions 1–5. Features use only causal rolling windows, and walk-forward folds enforce `train_end + embargo <= test_start`. Tests check that perturbing any data after `t` leaves the observation, the target weights and the order at `t` unchanged, plus a hand-computed two-asset example, overnight-gap, executed-quantity cost and cash/weight invariants.

**Frictions.** For each trade of `q` dollars the env charges:

- commission `c|q|` and half-spread `s|q|`;
- temporary impact `η σ √(|q|/ADV) |q|`;
- permanent impact: the trade shifts a per-asset impact state by `γ σ √(|q|/ADV)`. The trade pays half of that shift, and later trades pay or recoup the remaining shift as it decays with a half-life.

Positions are marked at un-impacted prices, so impact never shows up as a paper gain. Trades are sized on the NAV after costs, so post-trade weights match the target exactly with no hidden leverage. An optional participation cap limits each trade to a fraction of ADV. Every step enforces `NAV_after_trade = NAV_before − total_cost`.

**Reward.** `R_t = scale · (r_t − λ₁·CVaR_α(window) − λ₂·D_t²)`. Inputs are sanitised and the result clipped, so the reward stays finite for −100% moves, short squeezes (NAV < 0) and NaN or inf inputs.

**Lambda scale in the ablation.** CVaR here is a *daily* loss, while the return term is a single day's return. In a 30k-step check run, model D (`λ₁ = λ₂ = 1`) averaged a CVaR penalty of about 0.023 and a drawdown penalty of about 0.004 per step, against a mean daily return of about 0.0007. Its episode reward was about −700, compared with +26 for model A. Because the penalties dominate the return by roughly 30×, the gradient pushes B and D toward cash. At 30k steps their cash weight had only moved from 16.5% to 18%, so the size of that shift over a full training run is not yet measured. If you want the penalised models to trade off risk against return, try `λ₁ ∈ [0.02, 0.1]` and `λ₂ ∈ [0.1, 1]` (the defaults in the `reward` section).
