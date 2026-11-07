## Out-of-sample performance (stitched walk-forward test windows; PPO = mean ± std over seeds)

| Strategy                                                 | Cum. Return (%) | Sharpe         | Sortino        | MaxDD (%)    | Calmar         | Avg Daily Turnover | Total Cost (% NAV) | Total Cost ($)   |
|----------------------------------------------------------|-----------------|----------------|----------------|--------------|----------------|--------------------|--------------------|------------------|
| PPO-S Smoothed execution (l1=0, l2=0)                    | -46.45 ± 9.77   | -0.565 ± 0.170 | -0.728 ± 0.216 | 51.97 ± 8.00 | -0.278 ± 0.033 | 0.0410 ± 0.0007    | 1.857 ± 0.035      | 181,316 ± 7,989  |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | -26.93 ± 10.92  | -0.296 ± 0.059 | -0.381 ± 0.076 | 36.95 ± 6.05 | -0.202 ± 0.061 | 0.0286 ± 0.0063    | 1.126 ± 0.268      | 113,129 ± 23,893 |
| Equal Weight (1/N)                                       | -39.57          | -0.274         | -0.356         | 51.58        | -0.230         | 0.0063             | 0.411              | 41,218           |
| Buy & Hold (1/N)                                         | -34.55          | -0.214         | -0.280         | 50.78        | -0.198         | 0.0040             | 0.313              | 31,344           |
| MVO (Ledoit-Wolf)                                        | -7.85           | 0.037          | 0.050          | 42.74        | -0.047         | 0.0157             | 0.882              | 83,911           |
| Risk Parity (ERC)                                        | -34.60          | -0.276         | -0.359         | 45.59        | -0.221         | 0.0062             | 0.367              | 36,799           |
| HRP                                                      | -32.44          | -0.324         | -0.422         | 40.79        | -0.229         | 0.0067             | 0.355              | 35,321           |
| Untrained policy (1/(N+1) incl. cash)                    | -32.41          | -0.271         | -0.353         | 44.88        | -0.208         | 0.0124             | 0.577              | 57,631           |

### Sharpe ratio by fold (mean over seeds)

| Strategy                                                 | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|----------------------------------------------------------|-------:|-------:|-------:|-------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    | -1.178 |  0.534 | -0.329 | -1.563 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | -1.202 |  0.922 |  0.001 | -0.942 |
| Equal Weight (1/N)                                       | -1.150 |  1.025 |  0.271 | -1.404 |
| Buy & Hold (1/N)                                         | -1.216 |  1.025 |  0.467 | -1.291 |
| MVO (Ledoit-Wolf)                                        | -0.873 |  2.094 |  0.185 | -1.700 |
| Risk Parity (ERC)                                        | -1.114 |  1.117 |  0.189 | -1.559 |
| HRP                                                      | -1.083 |  1.172 |  0.010 | -1.726 |
| Untrained policy (1/(N+1) incl. cash)                    | -1.131 |  1.029 |  0.259 | -1.407 |

### Regime-conditional statistics (annualised mean %, daily CVaR95 %, worst day %)

| Strategy                                                 | ann_mean_pct [calm] | ann_mean_pct [stress] | cvar_95_pct [calm] | cvar_95_pct [stress] | worst_day_pct [calm] | worst_day_pct [stress] |
|----------------------------------------------------------|--------------------:|----------------------:|-------------------:|---------------------:|---------------------:|-----------------------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |               -2.79 |                -71.24 |               2.43 |                 7.99 |                -6.15 |                 -13.68 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |                3.50 |                -59.17 |               1.84 |                 6.98 |                -5.12 |                 -15.08 |
| Equal Weight (1/N)                                       |                4.45 |                -78.76 |               2.56 |                10.51 |                -6.10 |                 -22.31 |
| Buy & Hold (1/N)                                         |                6.45 |                -77.64 |               2.53 |                10.22 |                -6.14 |                 -20.92 |
| MVO (Ledoit-Wolf)                                        |               13.74 |                -71.38 |               2.59 |                 7.25 |                -5.51 |                 -13.01 |
| Risk Parity (ERC)                                        |                4.09 |                -70.47 |               2.29 |                 9.25 |                -5.46 |                 -19.47 |
| HRP                                                      |                3.15 |                -65.79 |               2.07 |                 7.93 |                -4.63 |                 -16.52 |
| Untrained policy (1/(N+1) incl. cash)                    |                3.54 |                -64.21 |               2.13 |                 8.77 |                -5.02 |                 -18.50 |

### Sharpe-difference tests (paired circular block bootstrap, block=21, B=2000)

| strategy                                                 | versus             |   diff | ci_low | ci_high | p_value |
|----------------------------------------------------------|--------------------|-------:|-------:|--------:|--------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    | Equal Weight (1/N) | -0.300 | -0.617 |  -0.005 |   0.056 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | Equal Weight (1/N) | -0.033 | -0.224 |   0.180 |   0.754 |
| Buy & Hold (1/N)                                         | Equal Weight (1/N) |  0.060 | -0.022 |   0.148 |   0.154 |
| MVO (Ledoit-Wolf)                                        | Equal Weight (1/N) |  0.311 | -0.249 |   0.908 |   0.283 |
| Risk Parity (ERC)                                        | Equal Weight (1/N) | -0.002 | -0.114 |   0.100 |   0.975 |
| HRP                                                      | Equal Weight (1/N) | -0.050 | -0.333 |   0.205 |   0.717 |
| Untrained policy (1/(N+1) incl. cash)                    | Equal Weight (1/N) |  0.003 | -0.024 |   0.028 |   0.816 |

### Allocation behaviour

| strategy                                                 | avg_gross_exposure | avg_cash_weight | min_gross_exposure | avg_effective_n | max_single_weight | avg_daily_turnover |
|----------------------------------------------------------|-------------------:|----------------:|-------------------:|----------------:|------------------:|-------------------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |              0.788 |           0.212 |              0.064 |           3.738 |             0.667 |              0.041 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |              0.690 |           0.310 |              0.039 |           4.185 |             0.446 |              0.029 |
| Equal Weight (1/N)                                       |              1.000 |          -0.000 |              0.999 |           4.987 |             0.262 |              0.006 |
| Buy & Hold (1/N)                                         |              1.001 |          -0.001 |              0.999 |           4.776 |             0.384 |              0.004 |
| MVO (Ledoit-Wolf)                                        |              1.000 |           0.000 |              0.997 |           1.905 |             1.000 |              0.016 |
| Risk Parity (ERC)                                        |              1.000 |          -0.000 |              1.000 |           4.444 |             0.351 |              0.006 |
| HRP                                                      |              1.000 |          -0.000 |              1.000 |           3.142 |             0.554 |              0.007 |
| Untrained policy (1/(N+1) incl. cash)                    |              0.833 |           0.167 |              0.795 |           4.999 |             0.191 |              0.012 |

### Folds

- Fold 0: train 2011-10-03 -> 2015-08-12, test 2015-08-20 -> 2016-08-08
- Fold 1: train 2012-09-19 -> 2016-07-29, test 2016-08-08 -> 2017-07-26
- Fold 2: train 2013-09-06 -> 2017-07-18, test 2017-07-26 -> 2018-07-13
- Fold 3: train 2014-08-26 -> 2018-07-05, test 2018-07-13 -> 2019-07-01
