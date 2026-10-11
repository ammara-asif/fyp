# Autonomous Dynamic-Nesting Networks (FYP)

This project asks whether **nested learners** (networks whose levels update on different timescales) can **slow down module expansion** in continual learning.
- The research proposal is in [documents/proposal.md](documents/proposal.md).
- The staged plan (Stages 0–4, each ending in a decision gate) is in [documents/plan.md](documents/plan.md).

**Current status:** Stage 0 is done (Gate 0 passed). Stage 1 is in progress: the control arms are run and analysed, and the nested learner is not built yet.

## Repository layout

| Path | What it is |
|---|---|
| `src/fyp_infra.py` | Paths, stable run IDs (a hash of the config), resumable `run_sweep` with atomic result files |
| `src/stage1_runner.py` | Training runner: config defaults, models, strategies, LLRD learning-rate groups, train/val/test split, metrics, wandb logging, checkpoints |
| `src/stage1_analysis.py` | Post-hoc analysis of saved checkpoints: task-IL accuracy and per-level linear probes |
| `notebooks/*.ipynb` | Colab notebooks. **Generated** by `notebooks/_build_notebooks.py`, so edit that file, not the notebooks |
| `runs/` | Copies of executed notebooks, with their outputs |
| `tests/test_smoke.py` | Tiny CPU tests (synthetic data, wandb disabled). Not yet run |
| `documents/` | Proposal and plan |

## How to run (Google Colab)

1. Push this repo to GitHub, and set `REPO_URL` in the notebook's first cell.
2. Add the Colab secret `WANDB_API_KEY`. If the repo is private, also add `GITHUB_TOKEN`.
3. Choose a T4 GPU runtime. The training notebooks stop if there is no GPU; `stage1_analysis` runs on CPU.
4. Run `00_setup.ipynb` once. It mounts Drive, installs the dependencies and downloads CIFAR-100 to Drive.
5. Run the stage notebooks in order:

   `stage0_ewc` → `stage1_epoch_sweep` → `stage1_lr_grid` → `stage1_control` → `stage1_analysis`

Everything that must survive a Colab reset lives on Drive under `MyDrive/fyp_nested_cl/`:
- `data/`: CIFAR-100.
- `results/{run_id}.json`: the permanent record of each run, which also marks it as finished.
- `checkpoints/{run_id}/`: model weights after each task.
- `logs/`: error tracebacks from crashed runs.

Re-running a notebook skips finished runs and retries crashed ones. wandb project: `fyp-nested-cl`, one group per stage.

To change a notebook: edit `notebooks/_build_notebooks.py`, then run `python notebooks/_build_notebooks.py`.

## Setup and conventions

**Benchmark**
- Split-CIFAR-100, 20 tasks × 5 classes.
- Class-incremental: one 100-way output layer, no task labels.

**Model**
- Reduced ResNet-18 (width 20, 1.11M parameters).

**Training**
- SGD, batch 32, no momentum, gradient-norm clipping at 1.0.
- 10 epochs per task.

**Evaluation**
- Tuning uses a fixed, stratified 10% validation slice of the training set.
- Final runs train on the full training set and evaluate on the test set.

**Seeds**
- The seed controls both weight initialisation and class order.
- Arms with the same seed see identical data.

**Flat + LLRD**
- The body is split into three depth groups: stem + layer1 / layer2 + layer3 / layer4.
- The groups train at lr ÷ 1, ÷ 4 and ÷ 16 (deepest slowest, "slow-deep").
- The head is in the fast group.

**Metrics** (defined in the module docstrings; don't change them silently)
- From training runs: final average accuracy, average incremental accuracy, forgetting (best − final), and accuracy on the just-learned task.
- From `stage1_analysis`: task-IL accuracy and per-level linear-probe forgetting.

## Results so far

All numbers come from the result files on Drive. The executed notebooks are in `runs/`.

**Stage 0: EWC, 1 epoch, λ = 1, 2 seeds, test set.** Gate 0 passed.
- The pipeline runs end to end, logs to wandb and saves JSON results.
- The two seeds agree within about 0.5 points: final accuracy 2.65 / 3.14, forgetting 53.9 / 53.4.
- Every old task ends at 0%, the expected class-IL failure of EWC.
- An earlier attempt with λ = 100 diverged to NaN. That's why the NaN check and gradient clipping exist.

**Epoch sweep: Flat, seed 0, validation split.**
- Current-task accuracy is about 50 / 68 / 76 / 82% at 1 / 5 / 10 / 20 epochs, so the gains level off.
- **10 epochs chosen.**
- Cost on a T4 is about 8.5 minutes per 10-epoch run.

**Learning-rate grid: 10 epochs, seed 0, validation split.** Selection rule: highest average incremental accuracy.
- Flat is best at lr **0.1** (grid 0.01–0.3).
- LLRD is best at lr **0.5** (grid 0.01–0.9).
- Once each arm is tuned, they are tied (14.47 vs 14.47).

**Control: Flat (lr 0.3\*) vs Flat + LLRD (lr 0.5), 3 seeds, test set.**
- The two arms are indistinguishable on every accuracy metric. Forgetting: 82.2 ± 0.3 vs 81.8 ± 0.3.
- Old tasks end at exactly 0% in both arms. So class-IL forgetting here only measures how well each task was learned.

\* Flat was run at 0.3 instead of the selected 0.1, and has to be re-run at 0.1.

**Checkpoint analysis: the same 6 control runs.**

| | Flat | Flat + LLRD |
|---|---|---|
| Linear probe on old tasks, final model (5-way, chance 20%) | 66–79% | 66–79% |
| Probe forgetting, layer1 | 2.0 | 1.3 |
| Probe forgetting, layer3 | 6.9 | 7.0 |
| Probe forgetting, layer4 | 10.0 | 10.0 |
| Task-IL accuracy, final | ~33% | ~33% |

- Most forgetting happens in the **classifier head**, not in the features.
- Deeper layers forget more.
- **LLRD does not protect its slow levels**: layer4 trains about 10× slower than in Flat, yet its probe forgetting is the same.

## Open decisions (for the supervisor)

1. **Gate 1 criterion.** Class-IL forgetting can't separate the arms. The proposal is to judge Gate 1 on **per-level probe forgetting**, at matched accuracy on new tasks.
2. **"Matched effective learning rate."** Currently each arm's base lr is tuned separately on validation.
3. **Head placement and head bias.** Options: the slow level, a masked loss, or a nearest-class-mean head, applied equally to all arms.
4. **Learning-rate ratios.** (1, 4, 16) only, or also (1, 16, 256).
5. **Number of seeds.** 3 for exploration, 5 for final runs.
6. **Baseline scope.** LwF, ER, DER++, GDumb and replay are not built yet.

## Next steps

1. Re-run Flat at lr 0.1, then re-run the analysis.
2. Build the nested learner (variants A–D), sharing the same three-level split.
3. Compare the nested variants against the Flat + LLRD probe-forgetting bar above.
