
### Disjoint stream (6 tasks, class-IL)

| Method | n | Final acc (%) | Oracle acc (%) | Avg incr. acc (%) | Forgetting (pts) | Learners | Routing, all signals (%) | 'Mine' only | Referrals only | Shared space only | Likelihood ratio only | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Predictive split (ours) [flat, θ=0.1] | 1 | 7.00 | 7.33 | 8.90 | 17.00 | 2.00 | 93.83 | 30.00 | 69.00 | 92.67 | 85.17 | 2.22 | 2250.00 |
| Predictive split (ours) [nested_2level, θ=0.1] | 1 | 9.50 | 11.17 | 9.68 | 19.60 | 3.00 | 89.00 | 5.17 | 76.67 | 87.67 | 48.83 | 3.33 | 2250.00 |
| Predictive split (ours) [θ=0.0] | 1 | 11.67 | 15.67 | 9.37 | 15.80 | 4.00 | 80.33 | 25.00 | 46.67 | 79.50 | 47.83 | 4.44 | 2250.00 |
| Predictive split (ours) [θ=0.1] | 1 | 9.00 | 11.50 | 8.69 | 17.20 | 3.00 | 86.33 | 33.83 | 66.67 | 85.50 | 47.50 | 3.33 | 2250.00 |
| Predictive split (ours) [θ=0.3] | 1 | 9.17 | 11.50 | 9.02 | 18.00 | 3.00 | 88.17 | 7.83 | 76.67 | 88.00 | 44.17 | 3.33 | 2250.00 |
| Reactive split (DEN-style) [δ=0.1] | 1 | 6.00 | 6.00 | 8.45 | 21.20 | 1.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 1.11 | 2250.00 |
| No splitting | 1 | 7.00 | 7.00 | 8.72 | 21.60 | 1.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 1.11 | 2250.00 |

### Recurring stream (6 tasks, class-IL)

| Method | n | Final acc (%) | Oracle acc (%) | Avg incr. acc (%) | Forgetting (pts) | Learners | Routing, all signals (%) | 'Mine' only | Referrals only | Shared space only | Likelihood ratio only | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Predictive split (ours) [θ=0.1] | 1 | 8.33 | 8.33 | 10.22 | 18.67 | 1.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 1.11 | 1125.00 |
| Reactive split (DEN-style) [δ=0.1] | 1 | 8.33 | 8.33 | 10.58 | 19.33 | 1.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 1.11 | 1125.00 |

'Final acc' uses the soft router, which multiplies four label-free signals ('mine', referrals, a shared-space model in the root's slow levels, and a likelihood ratio) and then P(class) = Σ_learner P(learner) × P(class | learner). The single-signal routing columns show what each signal achieves alone. 'Oracle acc' sends every test image to its true owner: the ceiling routing is chasing. Routing = how often the most likely learner is the true owner. Both routers are label-free.