# PPG Sleep-Staging Benchmark — Locked Results Snapshot

_Generated: 2026-07-06T01:09:36.308819Z_

## Fold completion counts

| Model | MESA folds | CFS folds |
|---|---:|---:|
| `dca_sleep` | 5 | 5 |
| `deepsleepnet` | 5 | 5 |
| `insightsleepnet` | 5 | 5 |
| `plain_transformer` | 5 | 5 |
| `sleeppgnet` | 5 | 5 |
| `v8` | 15 | 15 |
| `v8_full_with_ultradian` | 5 | 5 |
| `v8_no_bottleneck` | 0 | 0 |
| `v8_no_ultradian` | 0 | 0 |
| `wang_dualstream` | 5 | 5 |

## MESA — primary comparison table

| Model | n | macro-F1 | κ | Deep F1 | REM F1 | acc |
|---|---:|---|---|---|---|---|
| `v8` | 15 | 0.7441 ± 0.0041 | 0.7128 ± 0.0053 | 0.5041 ± 0.0139 | 0.7730 ± 0.0040 | 0.8128 ± 0.0038 |
| `deepsleepnet` | 5 | 0.7140 ± 0.0030 | 0.6774 ± 0.0047 | 0.4636 ± 0.0122 | 0.7301 ± 0.0087 | 0.7906 ± 0.0033 |
| `sleeppgnet` | 5 | 0.7392 ± 0.0036 | 0.7045 ± 0.0073 | 0.4868 ± 0.0069 | 0.7821 ± 0.0097 | 0.8073 ± 0.0053 |
| `insightsleepnet` | 5 | 0.7216 ± 0.0055 | 0.6759 ± 0.0063 | 0.4895 ± 0.0113 | 0.7467 ± 0.0062 | 0.7887 ± 0.0042 |
| `wang_dualstream` | 5 | 0.7306 ± 0.0138 | 0.6823 ± 0.0257 | 0.4675 ± 0.0088 | 0.7963 ± 0.0171 | 0.7891 ± 0.0188 |
| `dca_sleep` | 5 | 0.7057 ± 0.0073 | 0.6653 ± 0.0102 | 0.4067 ± 0.0140 | 0.7699 ± 0.0083 | 0.7815 ± 0.0078 |

### MESA — V8 ablation

| Config | n | macro-F1 | κ | Deep F1 | REM F1 |
|---|---:|---|---|---|---|
| `v8` | 15 | 0.7441 ± 0.0041 | 0.7128 ± 0.0053 | 0.5041 ± 0.0139 | 0.7730 ± 0.0040 |
| `plain_transformer` | 5 | 0.7442 ± 0.0027 | 0.7133 ± 0.0038 | 0.5014 ± 0.0118 | 0.7759 ± 0.0038 |
| `v8_full_with_ultradian` | 5 | 0.7440 ± 0.0038 | 0.7099 ± 0.0052 | 0.5119 ± 0.0077 | 0.7692 ± 0.0044 |

### MESA — V8 vs baselines (paired stats)

| Comparison | Δκ (mean) | folds V8 wins | Wilcoxon p (n=5 paired) | MWU p (15 vs 5, greater) |
|---|---|---|---|---|
| V8 vs `deepsleepnet` | +0.0354 | 5 / 5 | 0.0431 | 0.0006 |
| V8 vs `sleeppgnet` | +0.0084 | 5 / 5 | 0.0431 | 0.0404 |
| V8 vs `insightsleepnet` | +0.0369 | 5 / 5 | 0.0431 | 0.0006 |
| V8 vs `wang_dualstream` | +0.0305 | 5 / 5 | 0.0431 | 0.0073 |
| V8 vs `dca_sleep` | +0.0475 | 5 / 5 | 0.0431 | 0.0006 |

## CFS — primary comparison table

| Model | n | macro-F1 | κ | Deep F1 | REM F1 | acc |
|---|---:|---|---|---|---|---|
| `v8` | 15 | 0.7135 ± 0.0129 | 0.6158 ± 0.0173 | 0.6385 ± 0.0203 | 0.6741 ± 0.0229 | 0.7317 ± 0.0127 |
| `deepsleepnet` | 5 | 0.6737 ± 0.0124 | 0.5670 ± 0.0173 | 0.6002 ± 0.0171 | 0.6134 ± 0.0229 | 0.6982 ± 0.0115 |
| `sleeppgnet` | 5 | 0.7003 ± 0.0131 | 0.6009 ± 0.0204 | 0.6033 ± 0.0267 | 0.6658 ± 0.0168 | 0.7224 ± 0.0149 |
| `insightsleepnet` | 5 | 0.6915 ± 0.0161 | 0.5808 ± 0.0236 | 0.6165 ± 0.0256 | 0.6619 ± 0.0095 | 0.7079 ± 0.0172 |
| `wang_dualstream` | 5 | 0.7185 ± 0.0148 | 0.6217 ± 0.0196 | 0.6232 ± 0.0112 | 0.6988 ± 0.0359 | 0.7347 ± 0.0153 |
| `dca_sleep` | 5 | 0.6572 ± 0.0101 | 0.5474 ± 0.0159 | 0.5374 ± 0.0206 | 0.6193 ± 0.0304 | 0.6841 ± 0.0118 |

### CFS — V8 ablation

| Config | n | macro-F1 | κ | Deep F1 | REM F1 |
|---|---:|---|---|---|---|
| `v8` | 15 | 0.7135 ± 0.0129 | 0.6158 ± 0.0173 | 0.6385 ± 0.0203 | 0.6741 ± 0.0229 |
| `plain_transformer` | 5 | 0.7158 ± 0.0164 | 0.6189 ± 0.0224 | 0.6371 ± 0.0292 | 0.6803 ± 0.0206 |
| `v8_full_with_ultradian` | 5 | 0.7021 ± 0.0147 | 0.6039 ± 0.0212 | 0.6233 ± 0.0156 | 0.6541 ± 0.0259 |

### CFS — V8 vs baselines (paired stats)

| Comparison | Δκ (mean) | folds V8 wins | Wilcoxon p (n=5 paired) | MWU p (15 vs 5, greater) |
|---|---|---|---|---|
| V8 vs `deepsleepnet` | +0.0488 | 5 / 5 | 0.0431 | 0.0006 |
| V8 vs `sleeppgnet` | +0.0148 | 4 / 5 | 0.0796 | 0.1282 |
| V8 vs `insightsleepnet` | +0.0349 | 5 / 5 | 0.0431 | 0.0073 |
| V8 vs `wang_dualstream` | -0.0060 | 2 / 5 | 0.8927 | 0.6365 |
| V8 vs `dca_sleep` | +0.0683 | 5 / 5 | 0.0431 | 0.0006 |
