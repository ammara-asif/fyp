# FYP Progress Update: Stage 0 and Stage 1 Control

2026-10-15

## Summary

Gate 0 has passed, and the Stage 1 deflationary control (Flat vs Flat + layer-wise LR decay) is run and analysed. The nested learner is not built yet.

The headline: once both arms are tuned, **Flat + LLRD is indistinguishable from Flat**, both on accuracy and on per-level representation forgetting. Slowing the deep layers by 16x does not make them forget less.

A second finding changes how Gate 1 should be judged. With plain sequential training in class-IL, every old task ends at exactly 0% accuracy in every arm, so class-IL forgetting cannot separate methods. Linear probes show that most old-task knowledge survives in the features (66–79% on old tasks, chance 20%). So the forgetting sits mostly in the classifier head.

I need decisions on the questions below before building the nested variants.

## Setup

All runs use one fixed protocol, so arms differ only in the method under test.

| Item | Choice | Why |
| --- | --- | --- |
| Benchmark | Split-CIFAR-100, 20 tasks × 5 classes, class-IL (one 100-way head, no task labels) | As specified in the plan |
| Model | Reduced ResNet-18, width 20, 1.11M parameters | Standard for this benchmark (GEM / Mammoth), so numbers are comparable |
| Training | SGD, batch 32, no momentum, gradient-norm clip 1.0, 10 epochs per task | Clip added after EWC diverged; applied to every arm |
| Tuning data | Fixed stratified 10% slice of the training set (50 images/class) | Nothing is tuned on the test set |
| Final runs | Full training set, evaluated on the test set, 3 seeds | Seed fixes both initialisation and class order, so arms see identical data |
| Flat + LLRD | Body split into 3 depth groups (stem + layer1 / layer2 + layer3 / layer4) at lr ÷1, ÷4, ÷16; head in the fast group | The same split will be used by the nested learner |

Infrastructure: every run is a config with a stable ID. It is saved as a JSON record on Google Drive and logged to Weights & Biases, and runs resume after a Colab disconnect. Model weights are saved after every task for later analysis.

## Results

### Stage 0: EWC baseline (Gate 0 passed)

EWC (λ = 1, 1 epoch) ran end to end on 20 tasks, logged to W&B and saved its records. Two seeds agreed within about 0.5 points (final accuracy 2.65% / 3.14%, forgetting 53.9 / 53.4). Every old task ended at 0%, the expected class-IL failure of EWC. A first attempt with λ = 100 diverged to NaN, so a NaN guard and gradient clipping were added.

### Epoch sweep (Flat, validation split, seed 0)

Accuracy on the task just learned rose from about 50% to 68%, 76% and 82% at 1, 5, 10 and 20 epochs. Gains level off past 10, so **10 epochs per task** was chosen. A 10-epoch run takes about 8.5 minutes on a T4 GPU.

### Learning-rate grid (validation split, seed 0)

Each arm's base learning rate was tuned with the same budget. The selection rule was fixed in advance: highest average incremental accuracy (%). Each arm peaks at a different learning rate, and once tuned they tie.

| Base (fast-level) lr | 0.01 | 0.03 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Flat | 12.12 | 13.47 | **14.47** | 14.41 | 14.25 | – | – | – |
| Flat + LLRD | 9.94 | 12.30 | 13.63 | 13.66 | 14.05 | **14.47** | 14.13 | 14.24 |

LLRD needs a higher base rate because its deep levels run at ÷4 and ÷16. Differences under about 0.5 points are within seed noise.

### Control: Flat vs Flat + LLRD (test set, 3 seeds, mean ± std)

| Metric | Flat (lr 0.3*) | Flat + LLRD (lr 0.5) |
| --- | --- | --- |
| Final average accuracy (%) | 4.40 ± 0.18 | 4.21 ± 0.28 |
| Average incremental accuracy (%) | 14.02 ± 0.08 | 13.89 ± 0.34 |
| Accuracy on just-learned task (%) | 82.53 ± 0.48 | 81.89 ± 0.58 |
| Class-IL forgetting (points) | 82.24 ± 0.33 | 81.78 ± 0.32 |

\*Flat was run at lr 0.3 instead of the selected 0.1 by mistake. On validation, 0.1 and 0.3 are within noise; a re-run at 0.1 is pending.

The arms are indistinguishable. Old tasks end at exactly 0% in both, so forgetting equals peak accuracy.

### Checkpoint analysis (same 6 runs, from saved weights)

For each level, a fresh linear probe was trained on each old task's features, right after the task and again at the end. Probe forgetting is the drop between the two (5-way, chance 20%).

| Measure | Flat | Flat + LLRD |
| --- | --- | --- |
| Probe forgetting, layer1 (points) | 2.0 ± 0.4 | 1.3 ± 0.9 |
| Probe forgetting, layer3 (points) | 6.9 ± 0.6 | 7.0 ± 1.2 |
| Probe forgetting, layer4 (points) | 10.0 ± 0.3 | 10.0 ± 1.4 |
| Final probe accuracy, layer4 (%) | 77.7 ± 0.2 | 76.3 ± 1.4 |
| Task-IL final accuracy (%) | 33.7 ± 1.9 | 32.7 ± 3.4 |

## Main findings

1. **Class-IL forgetting cannot decide Gate 1 in this setting.** With plain sequential training, every old task ends at exactly 0% in every arm, so "forgetting" equals how well each task was learned. An arm that learns less will always look as if it forgets less.
2. **Most forgetting is in the classifier head, not the features.** Linear probes still reach 66–79% on old tasks at the end of training, and probe forgetting is only 2–10 points. This matches published probing work (Davari et al., 2022).
3. **Deeper layers forget more:** about 2, 7 and 10 points at layer1, layer3 and layer4, in both arms. This metric is non-zero, differs by level and is stable across seeds, so it can separate methods.
4. **LLRD does not protect its slow levels.** Layer4 trains about 10x slower under LLRD (0.5 ÷ 16 ≈ 0.031, vs 0.3), yet its probe forgetting is identical (10.0 vs 10.0). There are two possible explanations, neither tested yet. First, drift may be inherited from the fast shallow layers feeding it. Second, BatchNorm statistics drift regardless of learning rate.

For Gate 1, this sets the bar. A nested variant must show clearly lower probe forgetting in its slow levels than Flat + LLRD, beyond the seed spread. It must also learn new tasks as well (about 82% on the just-learned task).

## Questions for you before the nested learner

These change what the Stage 1 result means, so I would like them settled before building variants A–D.

1. **Gate 1 criterion.** The plan says "beats Flat + LLRD on forgetting", but class-IL forgetting cannot separate the arms here. Can Gate 1 be judged on **per-level linear-probe forgetting**, at matched accuracy on new tasks? And can the margin be fixed in advance, for example larger than the seed standard deviation over 5 seeds?
2. **Matched effective learning rate.** The plan requires it but does not define it. I tuned each arm's base learning rate separately, with the same search budget. Is that acceptable? Or should the arms share one rate, or match the parameter-weighted average rate?
3. **Classifier head and head bias.** The head sits in the fast level, and head bias drives old-task accuracy to 0% for every method. Should I:
   - (a) keep it as it is and rely on probes;
   - (b) test the head in the slow level; or
   - (c) add a standard fix to all arms equally, such as a loss masked to the current task's classes, or a nearest-class-mean head?
4. **Level ratios.** Run only (1, 4, 16) in Stage 1, or also (1, 16, 256), which the plan lists for Stage 3? Both would need matching in the nested arms.
5. **Seeds and budget.** 3 seeds for exploration and 5 for the final Gate 1 comparison? Each 10-epoch run costs about 8.5 GPU-minutes on a free T4.
6. **Baselines for the Stage 1 report.** Only EWC is built. Are LwF, ER, DER++, GDumb and a byte-matched replay baseline all needed now, or can some wait until after Gate 1?
7. **LLRD's slow levels.** LLRD's layer4 drifts as much as Flat's despite a 10x lower rate. Is a short diagnostic worth running before building nesting, such as frozen BatchNorm statistics, or slowing all upstream layers?

## Next steps and caveats

- [ ] Re-run Flat at the selected lr 0.1 (3 runs, about 30 GPU-minutes), then re-run the checkpoint analysis
- [ ] Settle the questions above
- [ ] Build nested variants A–D on the same three-level split, and compare them against the Flat + LLRD probe-forgetting bar

Caveats:
- Results so far use 3 seeds, so differences of about 1 point are not detectable.
- Tuning used a single seed.
- EWC's λ was not tuned, so the Stage 0 numbers are a pipeline check, not a baseline result.
