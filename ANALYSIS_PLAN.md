# Pre-specified analysis plan: next-open rerun

Written and committed **before** any result of `config/real_etf_nextopen.yaml`
(`outputs_real_nextopen/`) was inspected. The git commit timestamp is the record.

## Why a rerun

An execution-timing audit found that every earlier experiment (v1–v5 of the paper)
filled orders at the close of day *t* while the features used that same close,
range and volume. That convention cannot be traded. The rerun decides after the
close of *t* and fills at the open of *t+1* (`execution: next_open`), with the old
book carried overnight. Costs use volatility and ADV known at the decision time.
Folds, dates, data, features, frictions and PPO hyper-parameters are unchanged.

## Strategies

PPO variants, each with 5 training seeds (0–4):
- A: vanilla;
- E: calibrated penalties, λ₁ = 0.05 and λ₂ = 0.5;
- S: vanilla with smoothed execution (κ = 0.1, band 1%);
- SE: smoothed with calibrated penalties.

Baselines:
- 1/N, monthly and daily;
- 1/N buy-and-hold;
- ERC, monthly and daily;
- HRP;
- Ledoit–Wolf MVO;
- the untrained policy.

## Estimand and inference

- **Estimand.** For a PPO variant, the expected out-of-sample metric of a newly
  trained agent, estimated as the mean over seeds of each seed's metric on the
  stitched 14-fold track. Baselines are deterministic and treated as one "seed".
- **Two-level bootstrap** (B = 2,000; block length 21 days):
  1. resample each variant's seeds with replacement;
  2. resample days with circular blocks, using the same indices for every series;
  3. recompute the metric difference.

  CIs are percentile intervals. p-values are two-sided, from the bootstrap
  distribution centred under the null.
- **Metrics:** annualised Sharpe (primary); annualised net return, maximum drawdown,
  daily CVaR₉₅ and mean daily turnover (secondary).

## Primary family (Holm, family-wise α = 0.05), net Sharpe unless stated

| ID | Comparison | Question |
|----|------------|----------|
| P1 | E − A | Do calibrated tail-risk penalties change net performance? |
| P2 | S − A | Does smoothed execution change net performance? |
| P3 | SE − 1/N (monthly) | Does the best-designed PPO differ from 1/N? |
| P4 | A − 1/N (monthly) | Does vanilla PPO differ from 1/N? |
| P5 | gross A − gross 1/N | Is there a pre-cost timing edge (the v5 headline claim)? |

All other comparisons are secondary: they are reported with CIs but not used for claims.

## Equivalence

Two strategies are called **equivalent** in Sharpe only if the 90% two-level
bootstrap CI of the difference lies inside **±0.15** (TOST at α = 0.05). If the
CI includes 0 but extends beyond ±0.15, the result is **inconclusive**, never "parity".

## Power

For each comparison, report the minimum detectable effect at 80% power,
MDE = (z₀.₉₇₅ + z₀.₈) · SE_boot ≈ 2.80 · SE_boot.

## Decision rule for the paper

If P5 is not significant after Holm correction, the claim that PPO learns a
pre-cost timing signal is withdrawn. The paper then reports the same-close result
only as an artifact of the non-tradable convention.

## Amendment 1, before any result (rerun restarted)

The first launch was stopped after about 30 minutes, with no results inspected. A
follow-up audit found that orders were converted from weights into share quantities
using the open of *t+1*, a price not available when a market-on-open order is submitted.
Orders are now **share quantities fixed at the decision close** (weights, NAV, volatility
and ADV up to *t*). Only the fill price, and so the realised notional and cost, uses the
open of *t+1*. The regression test asserts that observation, target weights **and order
shares** at *t* are unchanged when all data from *t+1* onwards is perturbed. Partial models
from the aborted launch are kept in `outputs_real_nextopen_aborted/` and are not used.

## Look-ahead checklist for the rerun

| Pathway | Status |
|---|---|
| Features (close, range, volume and order flow of day *t*) | Known at the decision; causal rolling z-scores (tested) |
| Order generation | Fixed at the close of *t* from information up to *t* (tested) |
| Fill | Open of *t+1*; old book carried overnight; hold days send no order (tested) |
| Cost inputs (volatility, ADV, impact state) | Up to *t* |
| Train/test boundaries | Training uses prices up to *train_end − 1*; 5-day embargo; warm starts use only earlier folds |
| Reward normalisation | Statistics from training only; observations not normalised |
| Baselines and overlay | Estimated on closes up to *t*; same fill convention |
| Dividend and split adjustment | Features are scale-invariant (tested), so later adjustments cannot leak in |

**Known limitations** (disclosed; not decision leaks):
- Dollar ADV uses back-adjusted prices, which understate early-sample ADV. This
  overstates impact costs early on, conservatively and mostly against high-turnover
  strategies.
- The ETF universe was chosen in 2026 from funds that still exist (survivorship
  and selection).
- The 2012–2026 test period informed design choices in earlier versions, so it
  is not an untouched hold-out. This plan fixes only the analysis.
- Stress-episode labels are defined with hindsight but used only for analysis.
