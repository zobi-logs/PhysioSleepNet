# Linear Probe — V8 autonomic latent -> PPG-derived features

Ridge regression from V8's 32-dim autonomic latent to classical PPG-derived autonomic features. Higher R^2 means more of the target is decodable linearly from the latent.

## MESA

| Target | folds | R^2 (mean +- std) | min | max |
|---|---:|---|---|---|
| HR_mean | 15 | 0.297 +- 0.193 | -0.068 | 0.608 |
| HR_std | 15 | 0.495 +- 0.135 | 0.231 | 0.661 |
| RMSSD | 15 | 0.439 +- 0.156 | 0.161 | 0.646 |
| HF_power | 15 | 0.284 +- 0.103 | 0.080 | 0.427 |
| resp_rate | 15 | -0.034 +- 0.027 | -0.077 | 0.004 |

## CFS

| Target | folds | R^2 (mean +- std) | min | max |
|---|---:|---|---|---|
| HR_mean | 15 | 0.170 +- 0.212 | -0.158 | 0.461 |
| HR_std | 15 | 0.172 +- 0.103 | 0.051 | 0.383 |
| RMSSD | 15 | 0.158 +- 0.094 | 0.018 | 0.340 |
| HF_power | 15 | 0.042 +- 0.052 | -0.117 | 0.105 |
| resp_rate | 15 | -0.034 +- 0.060 | -0.113 | 0.065 |

