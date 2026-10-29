
### Disjoint stream (6 tasks, class-IL)

| Method | n | Final acc, conf router (%) | Final acc, proto router (%) | Avg incr. acc (%) | Forgetting (pts) | Learners | Splits | Routing, conf (%) | Routing, proto (%) | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Predictive split (ours) [flat, θ=0.1] | 1 | 3.33 | 3.33 | 8.66 | 22.40 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 2250.00 |
| Predictive split (ours) [nested_2level, θ=0.1] | 1 | 4.17 | 4.17 | 9.04 | 21.40 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 2250.00 |
| Predictive split (ours) [θ=0.0] | 1 | 3.50 | 3.50 | 9.26 | 22.40 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 2250.00 |
| Predictive split (ours) [θ=0.1] | 1 | 3.33 | 3.33 | 9.11 | 21.60 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 2250.00 |
| Predictive split (ours) [θ=0.3] | 1 | 3.33 | 3.33 | 9.03 | 21.60 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 2250.00 |
| Reactive split (DEN-style) [δ=0.1] | 1 | 3.33 | 3.33 | 9.00 | 21.80 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 2250.00 |
| No splitting | 1 | 3.33 | 3.33 | 9.11 | 21.40 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 2250.00 |

### Recurring stream (6 tasks, class-IL)

| Method | n | Final acc, conf router (%) | Final acc, proto router (%) | Avg incr. acc (%) | Forgetting (pts) | Learners | Splits | Routing, conf (%) | Routing, proto (%) | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Predictive split (ours) [θ=0.1] | 1 | 8.33 | 8.33 | 9.89 | 15.67 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 1125.00 |
| Reactive split (DEN-style) [δ=0.1] | 1 | 8.33 | 8.33 | 9.78 | 15.00 | 1.00 | 0.00 | 100.00 | 100.00 | 1.11 | 1125.00 |

Preliminary, one seed per row unless n > 1. Training data is routed by label; both test-time routers are label-free. 'conf' picks the learner most confident over its own classes; 'proto' picks the nearest class centroid.