## Out-of-sample performance (stitched walk-forward test windows; PPO = mean ± std over seeds)

| Strategy                                                 | Cum. Return (%) | Sharpe         | Sortino        | MaxDD (%)     | Calmar         | Avg Daily Turnover | Total Cost (% NAV) | Total Cost ($)  |
|----------------------------------------------------------|-----------------|----------------|----------------|---------------|----------------|--------------------|--------------------|-----------------|
| PPO-S Smoothed execution (l1=0, l2=0)                    | -51.52 ± 6.78   | -0.660 ± 0.094 | -0.807 ± 0.105 | 60.42 ± 6.21  | -0.275 ± 0.020 | 0.0435 ± 0.0025    | 1.712 ± 0.117      | 162,690 ± 7,100 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | -33.53 ± 10.90  | -0.609 ± 0.086 | -0.733 ± 0.128 | 40.22 ± 12.02 | -0.241 ± 0.020 | 0.0251 ± 0.0085    | 0.910 ± 0.308      | 89,361 ± 29,262 |
| Equal Weight (1/N)                                       | -49.18          | -0.559         | -0.686         | 57.94         | -0.269         | 0.0057             | 0.283              | 27,992          |
| Buy & Hold (1/N)                                         | -45.57          | -0.513         | -0.634         | 54.81         | -0.258         | 0.0040             | 0.220              | 21,970          |
| MVO (Ledoit-Wolf)                                        | -15.77          | -0.082         | -0.107         | 44.51         | -0.094         | 0.0142             | 0.713              | 73,680          |
| Risk Parity (ERC)                                        | -42.24          | -0.502         | -0.614         | 51.86         | -0.247         | 0.0060             | 0.289              | 28,406          |
| HRP                                                      | -33.95          | -0.401         | -0.491         | 46.95         | -0.210         | 0.0078             | 0.372              | 36,752          |
| Untrained policy (1/(N+1) incl. cash)                    | -41.61          | -0.548         | -0.673         | 50.42         | -0.250         | 0.0098             | 0.395              | 38,789          |

### Sharpe ratio by fold (mean over seeds)

| Strategy                                                 | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|----------------------------------------------------------|-------:|-------:|-------:|-------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    | -0.996 |  0.960 | -0.952 | -0.942 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | -0.754 |  0.900 | -0.901 | -1.228 |
| Equal Weight (1/N)                                       | -0.965 |  1.214 | -0.909 | -0.890 |
| Buy & Hold (1/N)                                         | -0.763 |  1.166 | -0.962 | -0.913 |
| MVO (Ledoit-Wolf)                                        |  0.258 |  1.391 | -0.743 | -1.056 |
| Risk Parity (ERC)                                        | -0.698 |  1.104 | -0.827 | -0.953 |
| HRP                                                      | -0.358 |  0.976 | -0.755 | -0.925 |
| Untrained policy (1/(N+1) incl. cash)                    | -0.971 |  1.234 | -0.889 | -0.866 |

### Regime-conditional statistics (annualised mean %, daily CVaR95 %, worst day %)

| Strategy                                                 | ann_mean_pct [calm] | ann_mean_pct [stress] | cvar_95_pct [calm] | cvar_95_pct [stress] | worst_day_pct [calm] | worst_day_pct [stress] |
|----------------------------------------------------------|--------------------:|----------------------:|-------------------:|---------------------:|---------------------:|-----------------------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |               -3.84 |                -59.49 |               1.52 |                 7.90 |                -3.17 |                 -23.73 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |               -0.92 |                -40.86 |               0.93 |                 5.09 |                -1.90 |                 -16.23 |
| Equal Weight (1/N)                                       |                0.06 |                -66.50 |               1.57 |                 8.57 |                -3.38 |                 -23.35 |
| Buy & Hold (1/N)                                         |                0.76 |                -61.90 |               1.56 |                 8.26 |                -3.38 |                 -21.99 |
| MVO (Ledoit-Wolf)                                        |               12.03 |                -54.69 |               1.71 |                 6.85 |                -3.34 |                 -17.03 |
| Risk Parity (ERC)                                        |                2.09 |                -61.74 |               1.39 |                 7.86 |                -3.17 |                 -21.00 |
| HRP                                                      |                4.90 |                -58.25 |               1.27 |                 7.24 |                -2.93 |                 -19.31 |
| Untrained policy (1/(N+1) incl. cash)                    |                0.13 |                -54.76 |               1.31 |                 7.14 |                -2.81 |                 -19.41 |

### Sharpe-difference tests (paired circular block bootstrap, block=21, B=2000)

| strategy                                                 | versus             |   diff | ci_low | ci_high | p_value |
|----------------------------------------------------------|--------------------|-------:|-------:|--------:|--------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    | Equal Weight (1/N) | -0.107 | -0.333 |   0.089 |   0.322 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | Equal Weight (1/N) | -0.071 | -0.278 |   0.139 |   0.498 |
| Buy & Hold (1/N)                                         | Equal Weight (1/N) |  0.046 | -0.029 |   0.124 |   0.231 |
| MVO (Ledoit-Wolf)                                        | Equal Weight (1/N) |  0.477 |  0.061 |   0.940 |   0.033 |
| Risk Parity (ERC)                                        | Equal Weight (1/N) |  0.057 | -0.043 |   0.180 |   0.302 |
| HRP                                                      | Equal Weight (1/N) |  0.158 | -0.066 |   0.440 |   0.203 |
| Untrained policy (1/(N+1) incl. cash)                    | Equal Weight (1/N) |  0.011 | -0.006 |   0.028 |   0.189 |

### Allocation behaviour

| strategy                                                 | avg_gross_exposure | avg_cash_weight | min_gross_exposure | avg_effective_n | max_single_weight | avg_daily_turnover |
|----------------------------------------------------------|-------------------:|----------------:|-------------------:|----------------:|------------------:|-------------------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |              0.815 |           0.185 |              0.079 |           3.804 |             0.684 |              0.044 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |              0.562 |           0.438 |              0.043 |           4.068 |             0.381 |              0.025 |
| Equal Weight (1/N)                                       |              1.000 |          -0.000 |              0.999 |           4.993 |             0.243 |              0.006 |
| Buy & Hold (1/N)                                         |              1.000 |          -0.000 |              0.999 |           4.901 |             0.311 |              0.004 |
| MVO (Ledoit-Wolf)                                        |              1.000 |           0.000 |              0.997 |           2.162 |             1.000 |              0.014 |
| Risk Parity (ERC)                                        |              1.000 |          -0.000 |              0.999 |           4.619 |             0.320 |              0.006 |
| HRP                                                      |              1.000 |          -0.000 |              0.999 |           3.771 |             0.466 |              0.008 |
| Untrained policy (1/(N+1) incl. cash)                    |              0.833 |           0.167 |              0.793 |           4.999 |             0.185 |              0.010 |

### Folds

- Fold 0: train 2011-10-03 -> 2015-08-12, test 2015-08-20 -> 2016-08-08
- Fold 1: train 2012-09-19 -> 2016-07-29, test 2016-08-08 -> 2017-07-26
- Fold 2: train 2013-09-06 -> 2017-07-18, test 2017-07-26 -> 2018-07-13
- Fold 3: train 2014-08-26 -> 2018-07-05, test 2018-07-13 -> 2019-07-01
