
### Disjoint stream (20 tasks, class-IL)

| Method | n | Final acc, conf router (%) | Final acc, proto router (%) | Avg incr. acc (%) | Forgetting (pts) | Learners | Splits | Routing, conf (%) | Routing, proto (%) | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Predictive split (ours) [flat, θ=0.5] | 1 | 19.59 | 19.59 | 29.01 | 56.87 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 7500.00 |
| Predictive split (ours) [nested_2level, θ=0.5] | 1 | 17.92 | 17.92 | 27.19 | 57.11 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 7500.00 |
| Predictive split (ours) [θ=0.5] | 1 | 20.50 | 20.50 | 28.14 | 55.28 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 7500.00 |
| Reactive split (DEN-style) [δ=0.3] | 1 | 1.17 | 10.25 | 12.60 | 17.45 | 20.00 | 19.00 | 1.17 | 12.59 | 22.18 | 7500.00 |
| No splitting | 1 | 18.36 | 18.36 | 28.10 | 57.72 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 7500.00 |
| Single net + replay (Stage 2) | 1 | 18.17 | 18.17 | 26.94 | 53.82 | 0.00 | 0.00 | n/a | n/a | 1.11 | 6000.00 |

### Recurring stream (20 tasks, class-IL)

| Method | n | Final acc, conf router (%) | Final acc, proto router (%) | Avg incr. acc (%) | Forgetting (pts) | Learners | Splits | Routing, conf (%) | Routing, proto (%) | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Predictive split (ours) [θ=0.5] | 1 | 27.10 | 27.10 | 25.40 | 40.90 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 3750.00 |
| Reactive split (DEN-style) [δ=0.3] | 1 | 11.54 | 18.46 | 18.04 | 24.52 | 6.00 | 5.00 | 14.62 | 25.62 | 6.66 | 3750.00 |
| Single net + replay (Stage 2) | 1 | 24.98 | 24.98 | 26.85 | 34.90 | 0.00 | 0.00 | n/a | n/a | 1.11 | 3000.00 |

Preliminary, one seed per row unless n > 1. Training data is routed by label; both test-time routers are label-free. 'conf' picks the learner most confident over its own classes; 'proto' picks the nearest class centroid.