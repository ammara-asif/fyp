
### Disjoint stream (20 tasks, class-IL, seeds per row shown as n)

| Method | n | Final acc (%) | Avg incr. acc (%) | Forgetting (pts) | Modules | Routing acc (%) | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|
| Ours (plastic offspring) | 1 | 15.46 | 23.32 | 41.24 | 1.00 | 100.00 | 2.22 | 6900.00 |
| Frozen offspring | 1 | 2.11 | 10.01 | 13.67 | 3.00 | 16.65 | 4.44 | 3048.00 |
| Single net + replay (same memory) | 1 | 18.17 | 26.94 | 53.82 | 0.00 | n/a | 1.11 | 6000.00 |
| Ours, flat learner (no nesting) | 1 | 15.28 | 24.50 | 45.35 | 1.00 | 100.00 | 2.22 | 6600.00 |

### Recurring stream (20 tasks, class-IL, seeds per row shown as n)

| Method | n | Final acc (%) | Avg incr. acc (%) | Forgetting (pts) | Modules | Routing acc (%) | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|
| Ours (plastic offspring) | 1 | 23.40 | 24.35 | 34.62 | 1.00 | 100.00 | 2.22 | 4200.00 |
| Frozen offspring | 1 | 5.38 | 10.60 | 16.76 | 3.00 | 29.68 | 4.44 | 1980.00 |
| Single net + replay (same memory) | 1 | 24.98 | 26.85 | 34.90 | 0.00 | n/a | 1.11 | 3000.00 |

Preliminary: Stage 2 minimal system, no consolidation. Every method uses the same nested backbone, data order and epochs; the single-network baseline replays from a buffer holding the same number of raw images per class as the anchor sets.