## Out-of-sample performance (stitched walk-forward test windows; PPO = mean ± std over seeds)

| Strategy                                                 | Cum. Return (%) | Sharpe         | Sortino        | MaxDD (%)     | Calmar         | Avg Daily Turnover | Total Cost (% NAV) | Total Cost ($)   |
|----------------------------------------------------------|-----------------|----------------|----------------|---------------|----------------|--------------------|--------------------|------------------|
| PPO-S Smoothed execution (l1=0, l2=0)                    | -39.75 ± 11.18  | -0.434 ± 0.151 | -0.639 ± 0.183 | 48.54 ± 10.47 | -0.245 ± 0.032 | 0.0398 ± 0.0017    | 1.643 ± 0.057      | 161,833 ± 274    |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | -53.93 ± 14.05  | -0.792 ± 0.142 | -1.086 ± 0.153 | 60.53 ± 12.53 | -0.293 ± 0.044 | 0.0336 ± 0.0066    | 1.311 ± 0.268      | 122,701 ± 23,167 |
| Equal Weight (1/N)                                       | -62.81          | -0.686         | -0.937         | 70.24         | -0.312         | 0.0065             | 0.382              | 37,434           |
| Buy & Hold (1/N)                                         | -62.07          | -0.688         | -0.936         | 69.26         | -0.311         | 0.0040             | 0.272              | 27,244           |
| MVO (Ledoit-Wolf)                                        | -61.45          | -0.769         | -1.025         | 67.42         | -0.315         | 0.0186             | 1.123              | 102,655          |
| Risk Parity (ERC)                                        | -60.99          | -0.735         | -0.990         | 67.94         | -0.309         | 0.0065             | 0.369              | 36,009           |
| HRP                                                      | -58.77          | -0.774         | -1.029         | 65.69         | -0.303         | 0.0089             | 0.478              | 46,000           |
| Untrained policy (1/(N+1) incl. cash)                    | -55.41          | -0.695         | -0.947         | 63.25         | -0.289         | 0.0124             | 0.539              | 51,588           |

### Sharpe ratio by fold (mean over seeds)

| Strategy                                                 | Fold 0 | Fold 1 | Fold 2 | Fold 3 |
|----------------------------------------------------------|-------:|-------:|-------:|-------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    | -0.993 |  0.581 | -0.410 | -0.903 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | -0.974 |  0.079 | -0.719 | -1.438 |
| Equal Weight (1/N)                                       | -0.736 |  0.090 | -0.839 | -1.117 |
| Buy & Hold (1/N)                                         | -0.717 |  0.199 | -0.889 | -1.252 |
| MVO (Ledoit-Wolf)                                        | -0.572 | -0.479 | -1.466 | -0.621 |
| Risk Parity (ERC)                                        | -0.743 | -0.139 | -0.875 | -1.088 |
| HRP                                                      | -0.712 | -0.579 | -0.843 | -0.973 |
| Untrained policy (1/(N+1) incl. cash)                    | -0.760 |  0.061 | -0.826 | -1.111 |

### Regime-conditional statistics (annualised mean %, daily CVaR95 %, worst day %)

| Strategy                                                 | ann_mean_pct [calm] | ann_mean_pct [stress] | cvar_95_pct [calm] | cvar_95_pct [stress] | worst_day_pct [calm] | worst_day_pct [stress] |
|----------------------------------------------------------|--------------------:|----------------------:|-------------------:|---------------------:|---------------------:|-----------------------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |               -4.70 |                -25.43 |               1.93 |                 5.06 |                -3.36 |                  -7.71 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |               -8.12 |                -43.59 |               1.95 |                 5.09 |                -4.37 |                  -7.63 |
| Equal Weight (1/N)                                       |              -10.03 |                -48.74 |               2.45 |                 7.25 |                -5.33 |                 -11.42 |
| Buy & Hold (1/N)                                         |               -9.12 |                -50.03 |               2.41 |                 7.10 |                -5.46 |                 -11.45 |
| MVO (Ledoit-Wolf)                                        |              -10.30 |                -47.99 |               2.39 |                 6.58 |                -4.19 |                 -10.56 |
| Risk Parity (ERC)                                        |               -8.25 |                -51.93 |               2.19 |                 6.76 |                -4.44 |                 -10.73 |
| HRP                                                      |               -5.24 |                -57.32 |               1.99 |                 6.29 |                -3.76 |                 -10.21 |
| Untrained policy (1/(N+1) incl. cash)                    |               -8.49 |                -41.01 |               2.05 |                 6.03 |                -4.42 |                  -9.48 |

### Sharpe-difference tests (paired circular block bootstrap, block=21, B=2000)

| strategy                                                 | versus             |   diff | ci_low | ci_high | p_value |
|----------------------------------------------------------|--------------------|-------:|-------:|--------:|--------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    | Equal Weight (1/N) |  0.239 | -0.085 |   0.543 |   0.130 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) | Equal Weight (1/N) | -0.132 | -0.315 |   0.032 |   0.141 |
| Buy & Hold (1/N)                                         | Equal Weight (1/N) | -0.002 | -0.097 |   0.086 |   0.968 |
| MVO (Ledoit-Wolf)                                        | Equal Weight (1/N) | -0.083 | -0.516 |   0.373 |   0.705 |
| Risk Parity (ERC)                                        | Equal Weight (1/N) | -0.049 | -0.152 |   0.056 |   0.357 |
| HRP                                                      | Equal Weight (1/N) | -0.088 | -0.313 |   0.165 |   0.457 |
| Untrained policy (1/(N+1) incl. cash)                    | Equal Weight (1/N) | -0.008 | -0.038 |   0.017 |   0.557 |

### Allocation behaviour

| strategy                                                 | avg_gross_exposure | avg_cash_weight | min_gross_exposure | avg_effective_n | max_single_weight | avg_daily_turnover |
|----------------------------------------------------------|-------------------:|----------------:|-------------------:|----------------:|------------------:|-------------------:|
| PPO-S Smoothed execution (l1=0, l2=0)                    |              0.711 |           0.289 |              0.066 |           3.989 |             0.629 |              0.040 |
| PPO-SE Smoothed + calibrated penalties (l1=0.05, l2=0.5) |              0.778 |           0.222 |              0.079 |           4.302 |             0.443 |              0.034 |
| Equal Weight (1/N)                                       |              1.000 |           0.000 |              0.998 |           4.985 |             0.273 |              0.007 |
| Buy & Hold (1/N)                                         |              0.999 |           0.001 |              0.997 |           4.813 |             0.332 |              0.004 |
| MVO (Ledoit-Wolf)                                        |              1.000 |          -0.000 |              0.998 |           2.415 |             0.775 |              0.019 |
| Risk Parity (ERC)                                        |              1.000 |           0.000 |              0.998 |           4.572 |             0.362 |              0.007 |
| HRP                                                      |              1.000 |           0.000 |              0.998 |           3.451 |             0.574 |              0.009 |
| Untrained policy (1/(N+1) incl. cash)                    |              0.833 |           0.167 |              0.816 |           4.999 |             0.223 |              0.012 |

### Folds

- Fold 0: train 2011-10-03 -> 2015-08-12, test 2015-08-20 -> 2016-08-08
- Fold 1: train 2012-09-19 -> 2016-07-29, test 2016-08-08 -> 2017-07-26
- Fold 2: train 2013-09-06 -> 2017-07-18, test 2017-07-26 -> 2018-07-13
- Fold 3: train 2014-08-26 -> 2018-07-05, test 2018-07-13 -> 2019-07-01
