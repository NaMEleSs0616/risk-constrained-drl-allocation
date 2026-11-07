## Out-of-sample performance (stitched walk-forward test windows; PPO = mean ± std over seeds)

| Strategy                                                 | Cum. Return (%) | Sharpe        | Sortino       | MaxDD (%)    | Calmar        | Avg Daily Turnover | Total Cost (% NAV) | Total Cost ($)   |
|----------------------------------------------------------|-----------------|---------------|---------------|--------------|---------------|--------------------|--------------------|------------------|
| PPO-S Smoothed execution (l1=0, l2=0)                    | 1.14 ± 3.42     | 0.089 ± 0.061 | 0.139 ± 0.098 | 18.97 ± 1.36 | 0.016 ± 0.046 | 0.0380 ± 0.0034    | 1.460 ± 0.115      | 148,294 ± 11,355 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | 3.36 ± 3.33     | 0.129 ± 0.080 | 0.205 ± 0.133 | 14.07 ± 2.36 | 0.055 ± 0.049 | 0.0257 ± 0.0025    | 0.914 ± 0.084      | 93,020 ± 9,049   |
| Equal Weight (1/N)                                       | -5.99           | -0.012        | -0.018        | 31.32        | -0.049        | 0.0052             | 0.242              | 24,321           |
| Buy & Hold (1/N)                                         | -5.86           | -0.007        | -0.011        | 31.50        | -0.048        | 0.0040             | 0.199              | 19,867           |
| MVO (Ledoit-Wolf)                                        | -26.86          | -0.437        | -0.640        | 39.89        | -0.189        | 0.0222             | 1.131              | 111,020          |
| Risk Parity (ERC)                                        | -10.83          | -0.121        | -0.183        | 31.76        | -0.089        | 0.0053             | 0.238              | 23,911           |
| HRP                                                      | -14.28          | -0.222        | -0.328        | 31.05        | -0.122        | 0.0087             | 0.379              | 38,073           |
| Untrained policy (1/(N+1) incl. cash)                    | -3.94           | -0.005        | -0.008        | 26.40        | -0.038        | 0.0082             | 0.322              | 32,574           |

### Sharpe ratio by fold (mean over seeds)

| Strategy                                                 | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|----------------------------------------------------------|-------:|-------:|-------:|-------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |  0.528 | -0.158 |  0.195 | -0.492 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |  0.659 | -0.210 |  0.020 | -0.449 |
| Equal Weight (1/N)                                       |  0.724 | -0.295 | -0.297 | -0.627 |
| Buy & Hold (1/N)                                         |  0.731 | -0.308 | -0.320 | -0.588 |
| MVO (Ledoit-Wolf)                                        |  0.513 | -1.337 | -0.534 | -1.108 |
| Risk Parity (ERC)                                        |  0.681 | -0.426 | -0.342 | -0.928 |
| HRP                                                      |  0.596 | -0.509 | -0.458 | -1.040 |
| Untrained policy (1/(N+1) incl. cash)                    |  0.727 | -0.303 | -0.269 | -0.618 |

### Regime-conditional statistics (annualised mean %, daily CVaR95 %, worst day %)

| Strategy                                                 | ann_mean_pct [calm] | ann_mean_pct [stress] | cvar_95_pct [calm] | cvar_95_pct [stress] | worst_day_pct [calm] | worst_day_pct [stress] |
|----------------------------------------------------------|--------------------:|----------------------:|-------------------:|---------------------:|---------------------:|-----------------------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |               -5.22 |                 21.72 |               1.17 |                 2.86 |                -2.18 |                  -5.90 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |               -3.13 |                 15.62 |               0.80 |                 2.02 |                -1.73 |                  -4.98 |
| Equal Weight (1/N)                                       |               -8.53 |                 26.87 |               1.46 |                 3.53 |                -2.91 |                  -7.27 |
| Buy & Hold (1/N)                                         |               -8.47 |                 27.03 |               1.48 |                 3.58 |                -2.92 |                  -7.35 |
| MVO (Ledoit-Wolf)                                        |              -10.42 |                  5.52 |               1.48 |                 3.38 |                -3.39 |                  -7.23 |
| Risk Parity (ERC)                                        |               -8.34 |                 19.52 |               1.31 |                 3.23 |                -2.55 |                  -6.71 |
| HRP                                                      |               -7.74 |                 12.52 |               1.23 |                 2.94 |                -2.68 |                  -6.07 |
| Untrained policy (1/(N+1) incl. cash)                    |               -7.08 |                 22.69 |               1.22 |                 2.93 |                -2.42 |                  -6.06 |

### Sharpe-difference tests (paired circular block bootstrap, block=21, B=2000)

| strategy                                                 | versus             |   diff | ci_low | ci_high | p_value |
|----------------------------------------------------------|--------------------|-------:|-------:|--------:|--------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    | Equal Weight (1/N) |  0.099 | -0.133 |   0.348 |   0.417 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | Equal Weight (1/N) |  0.147 | -0.025 |   0.353 |   0.128 |
| Buy & Hold (1/N)                                         | Equal Weight (1/N) |  0.005 | -0.025 |   0.042 |   0.770 |
| MVO (Ledoit-Wolf)                                        | Equal Weight (1/N) | -0.425 | -0.819 |   0.003 |   0.040 |
| Risk Parity (ERC)                                        | Equal Weight (1/N) | -0.109 | -0.216 |  -0.001 |   0.048 |
| HRP                                                      | Equal Weight (1/N) | -0.210 | -0.430 |   0.027 |   0.073 |
| Untrained policy (1/(N+1) incl. cash)                    | Equal Weight (1/N) |  0.007 | -0.006 |   0.019 |   0.270 |

### Allocation behaviour

| strategy                                                 | avg_gross_exposure | avg_cash_weight | min_gross_exposure | avg_effective_n | max_single_weight | avg_daily_turnover |
|----------------------------------------------------------|-------------------:|----------------:|-------------------:|----------------:|------------------:|-------------------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |              0.711 |           0.289 |              0.052 |           3.893 |             0.694 |              0.038 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |              0.548 |           0.452 |              0.038 |           4.270 |             0.323 |              0.026 |
| Equal Weight (1/N)                                       |              1.000 |           0.000 |              0.999 |           4.996 |             0.230 |              0.005 |
| Buy & Hold (1/N)                                         |              1.000 |           0.000 |              0.999 |           4.956 |             0.288 |              0.004 |
| MVO (Ledoit-Wolf)                                        |              0.996 |           0.004 |              0.818 |           2.113 |             1.000 |              0.022 |
| Risk Parity (ERC)                                        |              1.000 |           0.000 |              1.000 |           4.611 |             0.291 |              0.005 |
| HRP                                                      |              1.000 |           0.000 |              1.000 |           3.730 |             0.429 |              0.009 |
| Untrained policy (1/(N+1) incl. cash)                    |              0.833 |           0.167 |              0.823 |           5.000 |             0.180 |              0.008 |

### Folds

- Fold 0: train 2011-10-03 -> 2015-08-12, test 2015-08-20 -> 2016-08-08
- Fold 1: train 2012-09-19 -> 2016-07-29, test 2016-08-08 -> 2017-07-26
- Fold 2: train 2013-09-06 -> 2017-07-18, test 2017-07-26 -> 2018-07-13
- Fold 3: train 2014-08-26 -> 2018-07-05, test 2018-07-13 -> 2019-07-01
