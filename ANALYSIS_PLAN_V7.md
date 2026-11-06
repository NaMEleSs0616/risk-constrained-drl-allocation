# Pre-specified analysis plan 2: second universe and risk-sensitive RL baseline

Written and committed **before** the sector data was downloaded and before any run of
`config/sector_nextopen.yaml` or `config/real_etf_nextopen_cvar.yaml`. The git commit
timestamp is the record. Everything not stated here is unchanged from `ANALYSIS_PLAN.md`
(next-open execution, frictions, PPO settings, estimand, two-level bootstrap with
B = 2,000 and 21-day blocks, TOST margin ±0.15 Sharpe, MDE = 2.80 × SE).

## Why

Review of version 6 asked for (i) evidence beyond one ten-ETF universe and one test
period, and (ii) a recognised risk-sensitive RL method instead of only reward penalties
on realised daily history.

## Second universe

- The nine original Select Sector SPDR ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY),
  daily adjusted OHLCV from Yahoo Finance from December 1998 (`data/sector_daily.csv`,
  not redistributed; hash recorded in the release manifest).
- 14 rolling folds (1,008-day training window, 5-day embargo, 252-day test window),
  **start-anchored**: the first test window begins as soon as a full training window is
  available (about 2003), so the test period (about 2003–2017) includes the 2008 crisis
  and was never used as a test period in any earlier version.
- Same chain protocol, budgets, five seeds, baselines and controls as the ETF study.
- Variants A, E, S, SE (as in the ETF study) and SC (below).

## Risk-sensitive RL baseline (SC)

PPO with smoothed execution (as S) whose objective is the episode-level mean-CVaR
`J = E[G] + β · CVaR_α(G)`, `G` the 252-day episode log return, optimised with the CVaR
policy gradient of Tamar et al. (2015) in terminal-reward form
(`src.reward.EpisodeCVaRObjective`): each training episode receives the terminal reward
`β / (1 − α) · min(G − ν, 0)`, `ν` the running empirical VaR of the last 256 training
episodes (shared across environments; switched on after 16 episodes). Parameters fixed
here, not tuned: **β = 1, α = 0.9**, no per-step penalties. SC is run on both universes
(ETF: `outputs_real_nextopen_cvar/`, same folds, seeds and test dates as
`outputs_real_nextopen/`).

## Family A: replication on the sector universe (Holm over A1–A5)

Same definitions as P1–P5 of `ANALYSIS_PLAN.md`, on the sector universe, net Sharpe
unless stated:

| ID | Comparison |
|----|------------|
| A1 | E − A |
| A2 | S − A |
| A3 | SE − 1/N (monthly) |
| A4 | A − 1/N (monthly) |
| A5 | gross A − gross 1/N |

**Replication rule.** The ETF conclusions are called *replicated* if A2 is significantly
positive, A4 significantly negative, and none of A1, A3, A5 is significantly positive.
Any other pattern is reported as a failure to replicate, with the specific difference.

## Family B: the risk-sensitive baseline (Holm over B1–B6)

| ID | Universe | Comparison | Question |
|----|----------|------------|----------|
| B1 | ETF | SC − S, Sharpe | Does a CVaR objective change risk-adjusted return? |
| B2 | ETF | SC − S, daily CVaR₉₅ | Does it lower daily tail risk? |
| B3 | ETF | SC − 1/N, Sharpe | Does it differ from 1/N? |
| B4 | Sectors | SC − S, Sharpe | as B1 |
| B5 | Sectors | SC − S, daily CVaR₉₅ | as B2 |
| B6 | Sectors | SC − 1/N, Sharpe | as B3 |

**Decision rule.** The paper's claim that "tail-risk shaping does not make PPO better
than 1/N" is extended to the risk-sensitive objective unless B3 or B6 is significantly
positive. A CVaR reduction (B2/B5) is attributed to learned risk management only if SC
also beats the exposure-matched 1/N control on daily CVaR and shows a positive
correlation between its cash weight and trailing volatility (secondary, reported with
intervals).

## Secondary (uncorrected, reported with intervals)

Other metrics for every pair above; the exposure-matched control, timing-versus-tilt
decomposition, de-risking diagnostics and regime-conditional statistics (stress episodes
listed in the config) for the sector universe; SC − SE on both universes.
