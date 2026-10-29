
### Disjoint stream (6 tasks, class-IL, seeds per row shown as n)

| Method | n | Final acc (%) | Avg incr. acc (%) | Forgetting (pts) | Modules | Routing acc (%) | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|
| Ours (plastic offspring) | 1 | 5.17 | 10.59 | 1.00 | 2.00 | 88.50 | 3.33 | 1290.00 |
| Frozen offspring | 1 | 7.17 | 13.35 | 10.20 | 3.00 | 48.00 | 4.44 | 1500.00 |
| Single net + replay (same memory) | 1 | 3.33 | 9.47 | 22.00 | 0.00 | n/a | 1.11 | 1800.00 |
| Ours, flat learner (no nesting) | 1 | 6.00 | 9.15 | 0.60 | 2.00 | 78.17 | 3.33 | 1002.00 |

### Recurring stream (6 tasks, class-IL, seeds per row shown as n)

| Method | n | Final acc (%) | Avg incr. acc (%) | Forgetting (pts) | Modules | Routing acc (%) | Params (M) | Raw memory (KB) |
|---|---|---|---|---|---|---|---|---|
| Ours (plastic offspring) | 1 | 15.67 | 16.92 | 2.33 | 3.00 | 82.33 | 4.44 | 1692.00 |
| Frozen offspring | 1 | 16.00 | 16.39 | 6.67 | 3.00 | 74.00 | 4.44 | 945.00 |
| Single net + replay (same memory) | 1 | 7.67 | 10.11 | 16.00 | 0.00 | n/a | 1.11 | 900.00 |

Preliminary: Stage 2 minimal system, no consolidation. Every method uses the same nested backbone, data order and epochs; the single-network baseline replays from a buffer holding the same number of raw images per class as the anchor sets.