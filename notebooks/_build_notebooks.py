"""Generates the Colab notebooks in this folder from plain-text cells.

Edit this file, then run `python notebooks/_build_notebooks.py`. Keeping the
source here means the notebooks diff cleanly and the shared setup cell is
identical in every notebook.
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

SETUP = r'''# ---- Setup: Drive + code + wandb (run this first, every session) ----
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO.git"   # <-- EDIT ONCE
BRANCH = "nested-b-d"   # the branch with the code this notebook needs (set back to "main" after merging)

import os, subprocess, sys
from google.colab import drive, userdata

# Drive must be mounted from the notebook kernel (never from a !python subprocess).
drive.mount("/content/drive")

# Code: fresh clone each session (Colab disk is wiped on reset), pull if it already exists.
# For a private repo, add a Colab secret GITHUB_TOKEN (a fine-grained read-only token).
CODE = "/content/fyp"
url = REPO_URL
try:
    tok = userdata.get("GITHUB_TOKEN")
    url = REPO_URL.replace("https://", f"https://{tok}@")
except Exception:
    pass
if not os.path.isdir(CODE):
    subprocess.run(["git", "clone", "-q", url, CODE], check=True)
# fetch + checkout + pull, so a clone left on another branch earlier in the session switches over
subprocess.run(["git", "-C", CODE, "fetch", "-q", "origin"], check=True)
subprocess.run(["git", "-C", CODE, "checkout", "-q", BRANCH], check=True)
subprocess.run(["git", "-C", CODE, "pull", "-q", "origin", BRANCH], check=True)
print(BRANCH, subprocess.check_output(["git", "-C", CODE, "log", "-1", "--oneline"]).decode())

# Dependencies (torch/torchvision come preinstalled with Colab's CUDA build; don't reinstall them).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{CODE}/requirements.txt"], check=True)

# wandb: key comes from the Colab secret WANDB_API_KEY (key icon in the left sidebar). Never hardcode it.
import wandb
wandb.login(key=userdata.get("WANDB_API_KEY"))

sys.path.insert(0, f"{CODE}/src")
# Python caches imported modules for the whole kernel session, so after a git pull
# a plain `import` would silently keep running the OLD code. Drop the cached copies.
_MINE = {"fyp_infra", "stage1_runner", "stage1_analysis", "stage1_drift_diagnostic",
         "stage1_structural", "nested_builder"}
for _m in list(sys.modules):
    if _m in _MINE or _m == "src" or _m.startswith("src."):   # src.* = the team's nested code
        sys.modules.pop(_m, None)
import torch
from fyp_infra import setup_paths, run_sweep, run_id, load_result
import stage1_runner as sr
paths = setup_paths()   # -> /content/drive/MyDrive/fyp_nested_cl/{data,results,logs}
print(paths)
# Hard stop for training: a CPU run is ~12x slower and its timings are useless for budgeting.
REQUIRE_GPU = True
if REQUIRE_GPU:
    assert torch.cuda.is_available(), "No GPU! Runtime -> Change runtime type -> T4 GPU, then restart."
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none (CPU runtime)")'''

SUMMARY = r'''import numpy as np, pandas as pd
rows = []
for cfg in CONFIGS:
    r = load_result(cfg, paths)
    if r is None:
        rows.append({**{k: cfg[k] for k in SHOW}, "status": "not finished"}); continue
    m = r["metrics"]
    rows.append({"arm": sr.arm_label(cfg), **{k: cfg[k] for k in SHOW}, "run_id": r["run_id"],
                 "final_avg_acc": m["final_avg_acc"], "avg_forgetting": m["avg_forgetting"],
                 "avg_incremental_acc": m["avg_incremental_acc"],
                 # mean of the diagonal: how well each task was learned when it was current.
                 # Report next to forgetting: forgetting = peak - final, so it grows with this.
                 "acc_just_learned": float(np.mean(np.diag(m["acc_matrix"]))),
                 "steps": m["optimizer_steps"], "n_params": m["n_params"], "minutes": r["wall_clock_s"] / 60})
df = pd.DataFrame(rows); df'''


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").splitlines(True)}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
            "source": s.strip("\n").splitlines(True)}


def write(name, cells):
    nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"provenance": []},
                       "kernelspec": {"name": "python3", "display_name": "Python 3"}}}
    with open(os.path.join(HERE, name), "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("wrote", name)


write("00_setup.ipynb", [
    md("""
# 00 — One-time Colab setup check

Run this once before the experiment notebooks. Runtime → Change runtime type → **T4 GPU**.

Before you start:
1. Push this repo to GitHub, and set `REPO_URL` in the cell below.
2. Colab sidebar → 🔑 **Secrets** → add `WANDB_API_KEY` (from wandb.ai/authorize) and turn on notebook access.
   If the repo is private, also add `GITHUB_TOKEN`.
"""),
    code(SETUP),
    md("## Download CIFAR-100 to Drive (once; later sessions reuse it)"),
    code(r'''from torchvision.datasets import CIFAR100
CIFAR100(paths["data"], train=True, download=True); CIFAR100(paths["data"], train=False, download=True)
print(os.listdir(paths["data"]))'''),
    md("## Tiny GPU smoke run (2 experiences, 1 epoch, wandb group `smoke`)"),
    code(r'''cfg = sr.make_cfg(stage="smoke", n_experiences=2, epochs=1)
out = run_sweep([cfg], sr.run_experiment, paths, force=True)
print(out[run_id(cfg)]["metrics"]["acc_matrix"])'''),
])

write("stage0_ewc.ipynb", [
    md("""
# Stage 0 — EWC on Split-CIFAR-100 (Gate 0)

Gate 0: an EWC baseline completes end to end and logs to wandb. This notebook also runs a **second seed**
for the reproducibility check. Results land in `MyDrive/fyp_nested_cl/results/{run_id}.json`.
If Colab disconnects, re-run all cells. Finished runs are skipped.
"""),
    code(SETUP),
    code(r'''EPOCHS = 1           # Gate 0 only needs the pipeline to work; real epochs come from the Stage 1 sweep
CONFIGS = [sr.make_cfg(stage="stage0", strategy="ewc", ewc_lambda=1.0, epochs=EPOCHS, seed=s,
                       eval_split="test")
           for s in (0, 1)]
SHOW = ["strategy", "seed", "epochs", "n_experiences"]
for c in CONFIGS: print(run_id(c), {k: c[k] for k in SHOW})'''),
    code("out = run_sweep(CONFIGS, sr.run_experiment, paths)"),
    md("## Summary + seed-to-seed check\nExpect high accuracy on the latest task, near-zero on old tasks (forgetting), and the two seeds within a few points of each other."),
    code(SUMMARY),
    code(r'''import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, len(CONFIGS), figsize=(6 * len(CONFIGS), 5))
for ax, cfg in zip(np.atleast_1d(axes), CONFIGS):
    r = load_result(cfg, paths)
    if r is None: continue
    im = ax.imshow(np.array(r["metrics"]["acc_matrix"]), vmin=0, vmax=100, cmap="viridis")
    ax.set(title=f"seed {cfg['seed']}  acc[i][j]", xlabel="eval on exp j", ylabel="after training exp i")
fig.colorbar(im, ax=axes); plt.show()'''),
])

write("stage1_epoch_sweep.ipynb", [
    md("""
# Stage 1 — Epoch sweep (decides epochs per task)

Flat ReducedResNet-18, Naive sequential training, 20 tasks, seed 0. For each epoch budget we record
final accuracy, forgetting and **GPU minutes**. Per-run cost × (arms × seeds) sizes the rest of Stage 1.
Pick the epoch count with your supervisor. Don't just take the highest accuracy, because it also has to fit the compute budget.
"""),
    code(SETUP),
    code(r'''EPOCH_GRID = [1, 5, 10, 20]   # edit if needed; 20 epochs × 20 tasks is the slow one, so run it last
CONFIGS = [sr.make_cfg(stage="stage1_epoch_sweep", strategy="naive", epochs=e, seed=0,
                       eval_split="val")   # tuning -> held-out validation slice, never test
           for e in EPOCH_GRID]
SHOW = ["epochs", "seed", "eval_split"]
for c in CONFIGS: print(run_id(c), {k: c[k] for k in SHOW})'''),
    code("out = run_sweep(CONFIGS, sr.run_experiment, paths)"),
    code(SUMMARY),
])

write("stage1_lr_grid.ipynb", [
    md("""
# Stage 1 — Learning-rate grid (validation split)

Tunes the base learning rate for each control arm: 10 epochs, seed 0, evaluated on the held-out
validation split (never the test set). `GRID` lists every lr tried per arm; finished runs are skipped,
so the table always shows each arm's whole grid.

Arms: `flat`, `flat_llrd` (slow-deep: layer4 slowest) and `flat_llrd_rev` (fast-deep: stem+layer1 slowest,
layer4 at the full lr). The reversed arm was added after the drift diagnostic showed that, under slow-deep,
layer4's drift is ~90% inherited from the faster layers below it.

**Selection rule (fixed before looking):** pick the lr with the highest `avg_incremental_acc`.
`final_avg_acc` is useless here (with Naive training every old task ends at 0%, so it is just last-task / 20),
and forgetting can't be the criterion because it rewards learning less.
If an arm's best lr is at the edge of its own grid, extend that arm's grid in that direction before choosing.

For both LLRD arms the tuned value is the **fast-level** lr; slower levels stay at lr/4 and lr/16.
Tuning each arm separately is option (c) from the "matched effective lr" question. If your supervisor
picks option (a) instead, LLRD just reuses Flat's best lr and you can skip the LLRD rows.

The Flat lr=0.03 point is the 10-epoch run from the epoch sweep (same config, so it is reused, not re-run).
"""),
    code(SETUP),
    code(r'''EPOCHS = 10
ARMS = {
    "flat":          dict(lr_schedule="flat"),
    "flat_llrd":     dict(lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="slow_deep"),
    "flat_llrd_rev": dict(lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="fast_deep"),
}
# Every lr tried per arm (already-finished runs are skipped). Extend a list if its best is at an edge.
GRID = {
    "flat":          [0.01, 0.03, 0.1, 0.2, 0.3],
    "flat_llrd":     [0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9],
    "flat_llrd_rev": [0.03, 0.1, 0.2, 0.3, 0.5],   # new: 5 runs, ~45 GPU-min (estimate)
}

def grid_cfg(arm, lr):
    # The flat/0.03 point already exists as the epoch-sweep run; keeping its stage name
    # gives the identical config -> identical run_id -> run_sweep reuses that result.
    stage = "stage1_epoch_sweep" if (arm == "flat" and lr == 0.03) else "stage1_lr_grid"
    return sr.make_cfg(stage=stage, strategy="naive", epochs=EPOCHS, seed=0, lr=lr,
                       eval_split="val", **ARMS[arm])

CONFIGS = [grid_cfg(arm, lr) for arm in ARMS for lr in GRID[arm]]
SHOW = ["lr", "epochs", "eval_split"]
for c in CONFIGS: print(run_id(c), sr.arm_label(c), {k: c[k] for k in SHOW})'''),
    code("out = run_sweep(CONFIGS, sr.run_experiment, paths)"),
    code(SUMMARY),
    code(r'''LABEL = {sr.arm_label(sr.make_cfg(epochs=EPOCHS, **a)): name for name, a in ARMS.items()}   # table label -> ARMS key
best = df.loc[df.groupby("arm")["avg_incremental_acc"].idxmax(), ["arm", "lr", "avg_incremental_acc", "acc_just_learned"]]
print(best)
for _, r in best.iterrows():
    lrs = GRID[LABEL[r["arm"]]]
    if r["lr"] in (min(lrs), max(lrs)):
        print(f"WARNING: best lr for {r['arm']} is at the edge of its grid ({r['lr']}); extend it before choosing.")'''),
])

write("stage1_control.ipynb", [
    md("""
# Stage 1 — Deflationary control: Flat vs Flat + LLRD (slow-deep and fast-deep)

The comparison that decides the project (plan §1.2). Both arms use the same network, data order (seed),
batch size and epochs, so parameter count and optimiser-step count are identical. The summary cell **asserts** this.
`flat_llrd` (slow-deep): shallow→deep at lr/1, lr/4, lr/16. `flat_llrd_rev` (fast-deep): shallow→deep at lr/16, lr/4, lr/1.
The head is in the fast level in both. An arm whose `LR` is still `None` is skipped, so you can run the others first.
The earlier Flat runs at lr 0.3 stay on Drive as a record; this notebook runs Flat at the selected lr 0.1.
Before running: settle the head placement and "matched effective lr" with your supervisor,
and fill `LR` from `stage1_lr_grid`. Checkpoints (~90 MB/run) go to `MyDrive/fyp_nested_cl/checkpoints/`.
Nesting arms A–D get added here after you approve the next step.
"""),
    code(SETUP),
    code(r'''EPOCHS = 10            # from the epoch sweep
SEEDS = [0, 1, 2]
# Base (fast-level) lr per arm, from stage1_lr_grid. Fill these in before running.
LR = {"flat": 0.1, "flat_llrd": 0.5, "flat_llrd_rev": None}   # rev: fill in from stage1_lr_grid
ARMS = {
    "flat":          dict(lr_schedule="flat"),
    "flat_llrd":     dict(lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="slow_deep"),
    "flat_llrd_rev": dict(lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="fast_deep"),
}
for name in ARMS:
    if LR[name] is None: print(f"skipping {name}: set LR['{name}'] from stage1_lr_grid first")
CONFIGS = [sr.make_cfg(stage="stage1_control", strategy="naive", epochs=EPOCHS, seed=s, lr=LR[name],
                       eval_split="test",          # final comparison -> real test set
                       save_checkpoints=True,      # weights after every task, for CKA / probes later
                       **arm)
           for s in SEEDS for name, arm in ARMS.items() if LR[name] is not None]
SHOW = ["lr", "seed", "epochs"]
for c in CONFIGS: print(run_id(c), sr.arm_label(c), {k: c[k] for k in SHOW})'''),
    code("out = run_sweep(CONFIGS, sr.run_experiment, paths)"),
    code(SUMMARY),
    code(r'''done = df[df.get("status") != "not finished"] if "status" in df else df
assert done["steps"].nunique() == 1 and done["n_params"].nunique() == 1, "steps or param count differ between arms. The comparison is NOT matched"
print(done.groupby(["arm", "lr"])[["final_avg_acc", "avg_forgetting", "avg_incremental_acc"]].agg(["mean", "std", "count"]))
print(done.groupby(["arm", "lr"])["acc_just_learned"].agg(["mean", "std"]))'''),
])

write("stage1_analysis.ipynb", [
    md("""
# Stage 1: Checkpoint analysis (task-IL accuracy + linear probes)

No training. This loads the per-task checkpoints saved by `stage1_control` and asks whether old-task
knowledge survives **inside** the network, even though class-IL accuracy on old tasks is 0% for every arm.

- **Task-IL accuracy:** classify each old task's test images using only that task's 5 classes.
  If this stays high, the features still separate old classes and the head bias causes the 0%.
- **Linear probe per level (layer1 / layer3 / layer4):** fit a fresh linear classifier on each level's
  features for task *j*, right after learning *j* and again at the end. **Probe forgetting** = the drop.
  The thesis predicts that slow (deep, under `slow_deep`) levels lose less. That is what LLRD vs Flat tests here.

Definitions are in `src/stage1_analysis.py`. Results go to `results/` (resumable, like training runs).
Runs fine on a **CPU runtime** (no training, only forward passes + small logistic regressions):
expect roughly 5-15 min per training run on CPU (estimate, not measured), a few minutes on GPU.
"""),
    code(SETUP.replace("REQUIRE_GPU = True", "REQUIRE_GPU = False   # analysis only: CPU runtime is fine")),
    code(r"""import stage1_analysis as sa
STAGES = ["stage1_control", "stage1_nested_control", "stage1_epochs_per_task"]   # old control, Flat/B/D via the nested path, steps-per-task test
TRAIN_RUNS = sa.list_runs(paths, stage=STAGES)
for r in TRAIN_RUNS:
    c = r["config"]; print(r["run_id"], sr.arm_label(c), "lr", c["lr"], "seed", c["seed"])
ACFGS = sa.analysis_configs(TRAIN_RUNS)"""),
    code("out = run_sweep(ACFGS, sa.analyze_run, paths)"),
    code(r"""import numpy as np, pandas as pd
rows = []
for a in ACFGS:
    r = load_result(a, paths)
    if r is None: continue
    m = r["metrics"]; c = m["train_config"]
    row = {"arm": sr.arm_label(c), "lr": c["lr"], "seed": c["seed"],
           "til_final_avg_acc": m["til_final_avg_acc"], "til_avg_forgetting": m["til_avg_forgetting"]}
    for l, name in enumerate(m["level_modules"]):
        row[f"probe_final/{name}"] = m["probe_final_avg"][l]
        row[f"probe_forgetting/{name}"] = m["probe_forgetting"][l]
    rows.append(row)
adf = pd.DataFrame(rows)
display(adf)
summary = adf.drop(columns="seed").groupby(["arm", "lr"]).agg(["mean", "std", "count"])
display(summary.T)"""),
    md("""
**How to read it.**
- `til_final_avg_acc` well above 20% (chance for 5-way) means the features still separate old classes.
- `probe_forgetting/<level>`: lower means that level kept more old-task information. If LLRD's slow levels
  (layer3, layer4) show clearly lower probe forgetting than Flat's, beyond the seed std, then slowing *does*
  protect knowledge and class-IL accuracy was hiding it. If not, LLRD really behaves like Flat.
"""),
    code(r"""import matplotlib.pyplot as plt
levels = [k.split("/")[1] for k in adf.columns if k.startswith("probe_forgetting/")]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for (arm, lr), g in adf.groupby(["arm", "lr"]):
    lab = f"{arm} (lr {lr})"
    for ax, key in zip(axes, ["probe_forgetting", "probe_final"]):
        vals = g[[f"{key}/{l}" for l in levels]]
        ax.errorbar(levels, vals.mean(), yerr=vals.std(), marker="o", capsize=4, label=lab)
axes[0].set(title="Linear-probe forgetting per level (lower = kept more)", ylabel="points")
axes[1].set(title="Final linear-probe accuracy on all tasks (5-way, chance 20%)", ylabel="%")
for ax in axes: ax.legend(); ax.grid(alpha=.3)
plt.show()"""),
])

write("stage1_drift_diagnostic.ipynb", [
    md("""
# Stage 1: Drift diagnostic. Where does a slow level's drift come from?

No training; CPU runtime is fine. Under Flat + LLRD, layer4 trains ~10x slower than under Flat, yet its
linear-probe forgetting is the same (~10 points). This notebook splits each level's drift into:

- **own**: only that level's weights (and its BN stats) changed, inputs held at the old checkpoint
- **inherited**: only the upstream levels changed, the level itself held at the old checkpoint
- **interaction**: what is left (the two changes are not simply additive)
- **BN recovery**: how much probe accuracy comes back when the final model's BatchNorm statistics are
  re-estimated on the old task's images (diagnostic only; it uses old data)

It does this by "stitching" checkpoints: old upstream + new level, and new upstream + old level.
Definitions are in `src/stage1_drift_diagnostic.py`. Rough cost: 10-20 min per training run on CPU (estimate).

**Predictions, written before running:**
1. Under LLRD, layer4's **own** drift is small and most of its drift is **inherited** from the faster levels above it.
2. Under Flat, own drift is a larger share than under LLRD.
3. If BN recovery is large in both arms, BatchNorm statistics are a shared drift source that no learning-rate
   schedule can fix, and every nested variant needs a BN policy for its slow levels.

Result of the first run (Flat at lr 0.3, slow-deep LLRD at lr 0.5): 1 held (layer4 drift ~90% inherited),
2 held weakly, 3 did not hold (BN recovery ~0).

**Predictions for `llrd_fast_deep`, written before its runs:**
4. The slow level (stem+layer1) has no upstream, so its drift is all own drift and stays small.
5. layer3's inherited drift is lower than under slow-deep, because the levels feeding it are now slower.
6. layer4 is now the fast level: its own drift rises, and its total probe forgetting stays at or above ~10 points.
"""),
    code(SETUP.replace("REQUIRE_GPU = True", "REQUIRE_GPU = False   # analysis only: CPU runtime is fine")),
    code(r"""import stage1_analysis as sa
import stage1_drift_diagnostic as dd
STAGES = ["stage1_control", "stage1_nested_control", "stage1_epochs_per_task"]   # old control, Flat/B/D via the nested path, steps-per-task test
TRAIN_RUNS = sa.list_runs(paths, stage=STAGES)
for r in TRAIN_RUNS:
    c = r["config"]; print(r["run_id"], sr.arm_label(c), "lr", c["lr"], "seed", c["seed"])
DCFGS = dd.diagnostic_configs(TRAIN_RUNS)"""),
    code("out = run_sweep(DCFGS, dd.diagnose_run, paths)"),
    code(r"""import numpy as np, pandas as pd
PARTS = ["total_drift", "own_drift", "inherited_drift", "interaction", "bn_recovery", "bn_procedure_effect"]
rows = []
for a in DCFGS:
    r = load_result(a, paths)
    if r is None: continue
    m = r["metrics"]; c = m["train_config"]
    for l, name in enumerate(m["level_modules"]):
        rows.append({"arm": sr.arm_label(c), "lr": c["lr"], "seed": c["seed"], "level": name,
                     **{p: m[p][l] for p in PARTS}})
ddf = pd.DataFrame(rows)
summary = ddf.drop(columns="seed").groupby(["arm", "lr", "level"]).agg(["mean", "std"]).round(2)
display(summary)"""),
    md("""
**Consistency check.** `total_drift` uses the same probe protocol on the same checkpoints as
`stage1_analysis`, so it should equal that notebook's `probe_forgetting` (difference ~0). A large difference
means something in the stitching code is wrong, and the decomposition should not be trusted.
"""),
    code(r"""diffs = []
for a in DCFGS:
    r = load_result(a, paths)
    ra = load_result({"analysis": sa.ANALYSIS_VERSION, "train_run_id": a["train_run_id"]}, paths)
    if r is None or ra is None: continue
    diffs += [abs(x - y) for x, y in zip(r["metrics"]["total_drift"], ra["metrics"]["probe_forgetting"])]
print(f"max |total_drift - probe_forgetting| = {max(diffs):.3f} points over {len(diffs)} values" if diffs
      else "no stage1_analysis results found to compare against")"""),
    code(r"""import matplotlib.pyplot as plt
levels = list(dict.fromkeys(ddf["level"]))
arms = list(dict.fromkeys(zip(ddf["arm"], ddf["lr"])))
comps = ["own_drift", "inherited_drift", "interaction"]
fig, axes = plt.subplots(1, len(arms) + 1, figsize=(6 * (len(arms) + 1), 4), sharey=True)
for ax, (arm, lr) in zip(axes, arms):
    g = ddf[(ddf["arm"] == arm) & (ddf["lr"] == lr)]
    w = 0.25
    for k, comp in enumerate(comps):
        stats = g.groupby("level")[comp].agg(["mean", "std"]).reindex(levels)
        ax.bar(np.arange(len(levels)) + (k - 1) * w, stats["mean"], w, yerr=stats["std"], capsize=3, label=comp)
    tot = g.groupby("level")["total_drift"].mean().reindex(levels)
    ax.scatter(np.arange(len(levels)), tot, color="black", zorder=3, label="total")
    ax.set(title=f"{arm} (lr {lr}): drift components", xticks=range(len(levels)), xticklabels=levels, ylabel="probe points")
    ax.axhline(0, color="grey", lw=.8); ax.legend(); ax.grid(alpha=.3, axis="y")
ax = axes[-1]
for k, (arm, lr) in enumerate(arms):
    g = ddf[(ddf["arm"] == arm) & (ddf["lr"] == lr)]
    stats = g.groupby("level")["bn_recovery"].agg(["mean", "std"]).reindex(levels)
    ax.bar(np.arange(len(levels)) + (k - 0.5) * 0.35, stats["mean"], 0.35, yerr=stats["std"], capsize=3, label=f"{arm} (lr {lr})")
ax.set(title="BN recovery (old-task BN stats put back)", xticks=range(len(levels)), xticklabels=levels)
ax.axhline(0, color="grey", lw=.8); ax.legend(); ax.grid(alpha=.3, axis="y")
plt.tight_layout(); plt.show()"""),
    md("""
**How to read it.**
- Compare `own_drift` with `inherited_drift` at layer3 and layer4, per arm. Prediction 1 holds if LLRD's
  layer4 bar is mostly *inherited*.
- `bn_recovery` close to `total_drift` means BN statistics explain most of the drift. Close to 0 means BN is not the cause.
  Only trust it if `bn_procedure_effect` is near 0.
- Differences smaller than the seed std (error bars) are not meaningful with 3 seeds.
"""),
])

NESTED_TABLE = r'''import numpy as np, pandas as pd

def table(configs):
    """One row per config: accuracy metrics, cost, and per-level means over tasks (nested arms)."""
    rows = []
    for cfg in configs:
        base = {"arm": sr.arm_label(cfg), "lr": cfg["lr"], "seed": cfg["seed"]}
        r = load_result(cfg, paths)
        if r is None:
            rows.append({**base, "status": "not finished"}); continue
        m = r["metrics"]
        row = {**base, "run_id": r["run_id"], "avg_incremental_acc": m["avg_incremental_acc"],
               "final_avg_acc": m["final_avg_acc"], "avg_forgetting": m["avg_forgetting"],
               # mean of the diagonal: accuracy on each task right after learning it (plasticity check)
               "acc_just_learned": float(np.mean(np.diag(m["acc_matrix"]))),
               "steps": m["optimizer_steps"], "n_params": m["n_params"],
               "minutes": r["wall_clock_s"] / 60, "peak_gpu_mb": m.get("peak_gpu_mem_mb")}
        log = m.get("level_log_per_exp") or []
        for k in (log[0] if log else {}):
            if k.startswith(("updates/", "rel_displacement/", "deployed_rel_displacement/", "penalty_ratio/")):
                row[k] = float(np.mean([x[k] for x in log]))
        rows.append(row)
    return pd.DataFrame(rows)'''

write("stage1_nested_bd.ipynb", [
    md("""
# Stage 1: nested variants B and D (through my runner)

Same benchmark, model size, data order, batch size, epochs, clipping and metrics as the Flat / LLRD control.
All three arms here use the team's ResNet class (`reduced_resnet18_blocks`, same architecture and, per the tests,
the same initial weights as my ResNet) and the team's `NestedLearner`, split into three levels:
`[stem, layer1] | [layer2, layer3] | [layer4, fc]`, `slow_side: input` (input-side level slowest).

- **Flat (`flat_nl`)**: variant B with all three ratios at 1, i.e. plain SGD through the same code path.
- **B (`nested_b`)**: per-level learning rates lr/16, lr/4, lr (input side first). With this split it is the same
  thing as my Flat + LLRD with `slow_side: input` (old label `llrd_fast_deep`), so section 2 is a regression check.
- **D (`nested_d_lam…`), one reading of the plan's "nested objectives"**: every level trains on the data loss;
  levels step every 16 / 4 / 1 minibatches (input side first) on their averaged gradients, all at the base lr; and each
  slower level also gets λ × the gradient of the faster neighbour's squared gradient norm (double backward).
  λ = 0 is variant A (periodic stepping) at the same lr.

Clipping: every arm clips each minibatch's new gradient to norm 1.0; D's accumulated gradients are not clipped again.
Logged per task: optimizer updates per level, how far each level's weights moved, D's penalty-to-data gradient ratio,
minutes, peak GPU memory.

Order: 0 tests → 1 smoke → 2 regression + B lr grid → 3 D lr grid, then λ grid → 4 control (test set, 3 seeds).
Every section is resumable: finished runs are skipped.
"""),
    code(SETUP),
    code(NESTED_TABLE),
    md("## 0. Tests (tiny CPU configs, a few minutes)\nRun after every `git pull`. Stop here if anything fails."),
    code(r'''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=CODE, capture_output=True, text=True,
                   env={**os.environ, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": ""})   # CPU: deterministic
print(r.stdout[-6000:]); print(r.stderr[-3000:])
assert r.returncode == 0, "tests failed: read the output above before running anything else"'''),
    md("""
## 1. Smoke test (2 experiences × 50 classes, 1 epoch, validation split)

Checks that each arm runs on the GPU with Avalanche 0.6.0 (the team's code was written for 0.5.0), and measures cost.
One epoch over 2 experiences sees the same training images as one epoch of the 20-task setup, so a full
10-epoch run trains for about **10 × `train_min`** (evaluation time comes on top).
`penalty_ratio/<level>` = D's penalty gradient size / data gradient size for that level (input for setting the λ grid).
"""),
    code(r'''SMOKE = [sr.arm_cfg(name, stage="smoke_nested", n_experiences=2, epochs=1, lr=0.1, eval_split="val")
         for name in ["flat", "nested_b", "nested_d"]]
out = run_sweep(SMOKE, sr.run_experiment, paths, force=True)   # force: always re-time after code changes
smoke = table(SMOKE)
smoke["train_min"] = [sum(out[run_id(c)]["metrics"]["train_minutes_per_exp"]) if out[run_id(c)] else None
                      for c in SMOKE]
smoke["est_full_run_train_min"] = 10 * smoke["train_min"]
display(smoke.T)'''),
    md("""
## 2. Regression check, then B's learning-rate grid (validation, seed 0)

**Regression check (criterion fixed before running):** B at lr 0.5 must match the old-path LLRD run with the same
settings (`llrd_fast_deep`, lr 0.5, from `stage1_lr_grid`) to within **0.5 points** on average incremental accuracy and
on accuracy on the just-learned task. If the init test passed, the only difference is GPU non-determinism.
If it fails, stop: the nested path is not reproducing the control.

**B's grid:** the old grid for this arm stopped at its best value, 0.5, and the control then used 0.6 without
validating it. Here B gets 0.5 / 0.6 / 0.7 / 0.9. Selection rule as before: highest `avg_incremental_acc`;
if the best is at the top of the grid, extend it. The old-path points below 0.5 (0.03-0.3) are shown for context
(they are only comparable if the regression check passes).
"""),
    code(r'''EPOCHS = 10

def val_cfg(name, lr, **kw):
    return sr.arm_cfg(name, stage="stage1_nested_lr_grid", epochs=EPOCHS, seed=0, lr=lr, eval_split="val", **kw)

OLD_B = {lr: sr.make_cfg(stage="stage1_lr_grid", strategy="naive", epochs=EPOCHS, seed=0, lr=lr, eval_split="val",
                         lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="fast_deep")
         for lr in [0.03, 0.1, 0.2, 0.3, 0.5]}
B_GRID = [0.5, 0.6, 0.7, 0.9]     # 0.5 first: it is the regression run
B_CONFIGS = [val_cfg("nested_b", lr) for lr in B_GRID]
for c in B_CONFIGS: print(run_id(c), sr.arm_label(c), c["lr"])'''),
    code("out = run_sweep(B_CONFIGS, sr.run_experiment, paths)"),
    code(r'''old, new = load_result(OLD_B[0.5], paths), load_result(B_CONFIGS[0], paths)
assert old is not None, "old llrd_fast_deep lr 0.5 val run not found on Drive"
if new is not None:
    reg = table([OLD_B[0.5], B_CONFIGS[0]])
    display(reg[["arm", "lr", "run_id", "avg_incremental_acc", "acc_just_learned", "final_avg_acc", "avg_forgetting", "minutes"]])
    d_inc = abs(reg["avg_incremental_acc"].iloc[0] - reg["avg_incremental_acc"].iloc[1])
    d_jl = abs(reg["acc_just_learned"].iloc[0] - reg["acc_just_learned"].iloc[1])
    A_old, A_new = np.array(old["metrics"]["acc_matrix"]), np.array(new["metrics"]["acc_matrix"])
    print(f"|diff| avg incremental {d_inc:.2f}, just-learned {d_jl:.2f}; acc-matrix max |diff| {np.abs(A_old - A_new).max():.2f}")
    print("REGRESSION CHECK", "PASSED" if max(d_inc, d_jl) < 0.5 else "FAILED: stop and investigate")'''),
    code(r'''bdf = table(list(OLD_B.values()) + B_CONFIGS)
display(bdf[["arm", "lr", "avg_incremental_acc", "acc_just_learned", "final_avg_acc", "minutes"]])
done = bdf[bdf["arm"] == "nested_b"].dropna(subset=["avg_incremental_acc"])
if len(done):
    best = done.loc[done["avg_incremental_acc"].idxmax()]
    print(f"best B lr on its grid: {best['lr']} (avg incremental {best['avg_incremental_acc']:.2f})")
    if best["lr"] == max(B_GRID): print("WARNING: best is at the top of the grid; extend B_GRID before choosing.")'''),
    md("""
## 3. D: learning-rate grid (λ = 0.1), then λ grid at the chosen lr (validation, seed 0)

Same selection rule. The λ grid is searched at D's best lr (one parameter at a time, not a full lr × λ grid,
to fit the budget). λ = 0 equals variant A at that lr: it shows whether the penalty adds anything to periodic stepping.
If the smoke test's `penalty_ratio` is far from 0.01-1 at λ = 0.1, re-centre `LAM_GRID` before running it (and note why).
"""),
    code(r'''D_LR_GRID = [0.03, 0.1, 0.2, 0.3, 0.5]
D_LR_CONFIGS = [val_cfg("nested_d", lr) for lr in D_LR_GRID]   # surprise_weight 0.1 from nested_d.yaml
for c in D_LR_CONFIGS: print(run_id(c), sr.arm_label(c), c["lr"])
out = run_sweep(D_LR_CONFIGS, sr.run_experiment, paths)'''),
    code(r'''ddf = table(D_LR_CONFIGS)
display(ddf.drop(columns=["seed", "run_id"], errors="ignore"))
done = ddf.dropna(subset=["avg_incremental_acc"]) if "avg_incremental_acc" in ddf else ddf.iloc[:0]
if len(done):
    best = done.loc[done["avg_incremental_acc"].idxmax()]
    print(f"best D lr: {best['lr']} (avg incremental {best['avg_incremental_acc']:.2f})")
    if best["lr"] in (min(D_LR_GRID), max(D_LR_GRID)): print("WARNING: best lr is at the edge of the grid; extend it first.")'''),
    code(r'''D_LR = None                          # <- fill in from the table above
# Re-centred after the smoke test (lr 0.1, lambda 0.1): penalty/data gradient ratio was 3.9 (stem+layer1) and
# 1.2 (layer2+layer3), i.e. the penalty dominated. This grid spans ratio ~0.1 to ~4 on the slow level.
# 0.1 at D_LR is already in the lr grid (same config -> reused). 0 = variant A at the same lr.
LAM_GRID = [0.0, 0.003, 0.01, 0.03, 0.1]
if D_LR is None:
    print("set D_LR first")
else:
    D_LAM_CONFIGS = [val_cfg("nested_d", D_LR, rule_kwargs=dict(surprise_weight=lam)) for lam in LAM_GRID]
    for c in D_LAM_CONFIGS: print(run_id(c), sr.arm_label(c), c["lr"])
    out = run_sweep(D_LAM_CONFIGS, sr.run_experiment, paths)
    display(table(D_LAM_CONFIGS).drop(columns=["seed", "run_id"], errors="ignore"))'''),
    md("""
## 4. Control: Flat, B, D on the test set (3 seeds, checkpoints)

Fill `LR` and `D_LAM` from sections 2-3 (Flat uses its tuned lr 0.1 from `stage1_lr_grid`). An arm still at `None` is
skipped. Checkpoints go to `MyDrive/fyp_nested_cl/checkpoints/` for `stage1_analysis`, `stage1_drift_diagnostic` and
`stage1_structural` (all CPU). The old-path control runs (Flat lr 0.1, `llrd_fast_deep` lr 0.6) are shown for comparison.
The summary **asserts** equal optimizer-step counts (minibatch backward passes) and parameter counts across all rows.
"""),
    code(r'''SEEDS = [0, 1, 2]
LR = {"flat": 0.1, "nested_b": None, "nested_d": None}
D_LAM = None
CONTROL = []
for s in SEEDS:                       # seed outer: each seed completes every arm before the next starts
    for name, lr in LR.items():
        if lr is None or (name == "nested_d" and D_LAM is None):
            continue
        kw = dict(rule_kwargs=dict(surprise_weight=D_LAM)) if name == "nested_d" else {}
        CONTROL.append(sr.arm_cfg(name, stage="stage1_nested_control", epochs=EPOCHS, seed=s, lr=lr,
                                  eval_split="test", save_checkpoints=True, **kw))
for name, lr in LR.items():
    if lr is None or (name == "nested_d" and D_LAM is None): print(f"skipping {name}: set LR / D_LAM first")
for c in CONTROL: print(run_id(c), sr.arm_label(c), c["lr"], c["seed"])'''),
    code("out = run_sweep(CONTROL, sr.run_experiment, paths)"),
    code(r'''OLD_CONTROL = [sr.make_cfg(stage="stage1_control", strategy="naive", epochs=EPOCHS, seed=s, lr=lr, eval_split="test",
                           save_checkpoints=True, **arm)
               for s in SEEDS for lr, arm in [(0.1, dict(lr_schedule="flat")),
                                              (0.6, dict(lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="fast_deep"))]]
cdf = table(OLD_CONTROL + CONTROL)
done = cdf[cdf["status"] != "not finished"] if "status" in cdf else cdf
assert done["steps"].nunique() == 1 and done["n_params"].nunique() == 1, "steps or param count differ between arms. The comparison is NOT matched"
cols = ["avg_incremental_acc", "acc_just_learned", "final_avg_acc", "avg_forgetting", "minutes", "peak_gpu_mb"]
display(done.groupby(["arm", "lr"])[cols].agg(["mean", "std", "count"]).round(2))
lvl = [c for c in done.columns if c.startswith(("updates/", "rel_displacement/", "deployed_rel_displacement/", "penalty_ratio/"))]
if lvl: display(done.groupby(["arm", "lr"])[lvl].mean().T.round(4))'''),
    md("""
**Next:** run `stage1_analysis`, `stage1_drift_diagnostic` and `stage1_structural` (CPU runtimes). They pick up both
`stage1_control` and `stage1_nested_control`, so old and new arms land in one table. Gate 1 (proposed) is judged on
per-level probe forgetting at matched just-learned accuracy, against Flat at its tuned lr and LLRD.
"""),
    md("""
## 5. D: learning-rate re-check at the chosen λ (validation, seed 0)

D's lr (0.7) was picked at λ = 0.1, where the penalty was 5-10× the data gradient and the clip shrank the data part of
each step, so the best lr may have moved at λ = 0.01. Same selection rule (highest `avg_incremental_acc`).
The lr 0.7 point already exists from the λ grid (same config, reused). If the best lr here differs clearly
(beyond ~0.5 points of noise), re-run D's control at that lr and report both; otherwise keep 0.7.
"""),
    code(r'''D_LAM_CHOSEN = 0.01
D_LR_RECHECK = [0.2, 0.4, 0.7]
D_RECHECK = [val_cfg("nested_d", lr, rule_kwargs=dict(surprise_weight=D_LAM_CHOSEN)) for lr in D_LR_RECHECK]
for c in D_RECHECK: print(run_id(c), sr.arm_label(c), c["lr"])
out = run_sweep(D_RECHECK, sr.run_experiment, paths)
rdf = table(D_RECHECK)
display(rdf.drop(columns=["seed", "run_id"], errors="ignore"))
done = rdf.dropna(subset=["avg_incremental_acc"]) if "avg_incremental_acc" in rdf else rdf.iloc[:0]
if len(done):
    best = done.loc[done["avg_incremental_acc"].idxmax()]
    print(f"best D lr at lambda {D_LAM_CHOSEN}: {best['lr']} (avg incremental {best['avg_incremental_acc']:.2f})")
    if best["lr"] in (min(D_LR_RECHECK), max(D_LR_RECHECK)): print("WARNING: best lr is at the edge of the re-check grid.")'''),
    md("""
## 6. Steps per task: Flat at 2 and 3 epochs per task (test set, 3 seeds, checkpoints) — plan R5

**Question:** is deep-layer drift set by the *number* of steps per task rather than the step size? So far a level's own
drift was the same (layer3 ≈ 4.7-4.9 points) whatever its lr, schedule or stepping period, which fits the hypothesis
that ~790 steps per task let every level converge regardless. With 2-3 epochs there are ~160-240 steps per task.

**Predictions, written before running:** if the hypothesis holds, own drift (drift diagnostic) and probe forgetting at
layer3/layer4 drop clearly at 2-3 epochs compared with 10. If they stay about the same, the number of steps is not the
limit and the hypothesis is wrong.

**Confound to keep in mind:** fewer epochs also means each task is learned less well (lower `acc_just_learned`), and
probe forgetting is measured from that lower starting point. Always read it next to `acc_just_learned` and
`probe_at_learn`; a drop in forgetting that only mirrors a lower peak is not protection.

lr stays at Flat's tuned 0.1 (tuned at 10 epochs; it is not re-tuned here, since this is a diagnostic, not an arm).
Flat runs through the same path as B and D. Optional: add `"nested_b"` to `EPT_ARMS` (at its lr) to test whether slowing
levels starts to help once there are fewer steps per task — that is the comparison that would matter for nesting.
"""),
    code(r'''EPT_EPOCHS = [2, 3]
EPT_ARMS = {"flat": 0.1}          # optional: "nested_b": 0.6
EPT = [sr.arm_cfg(name, stage="stage1_epochs_per_task", epochs=ep, seed=s, lr=lr,
                  eval_split="test", save_checkpoints=True)
       for s in SEEDS for ep in EPT_EPOCHS for name, lr in EPT_ARMS.items()]
for c in EPT: print(run_id(c), sr.arm_label(c), "epochs", c["epochs"], "seed", c["seed"])
out = run_sweep(EPT, sr.run_experiment, paths)'''),
    code(r'''REF = [c for c in CONTROL if sr.arm_label(c) in {sr.arm_label(sr.arm_cfg(n, epochs=EPOCHS)) for n in EPT_ARMS}]   # same arms at 10 epochs
edf = table(REF + EPT)
edf["epochs"] = [c["epochs"] for c in REF + EPT]
cols = ["avg_incremental_acc", "acc_just_learned", "final_avg_acc", "minutes"]
display(edf.groupby(["arm", "epochs"])[cols].agg(["mean", "std", "count"]).round(2))
lvl = [c for c in edf.columns if c.startswith("rel_displacement/")]
display(edf.groupby(["arm", "epochs"])[lvl].mean().T.round(4))'''),
    md("""
Then run `stage1_analysis` and `stage1_drift_diagnostic` (they include `stage1_epochs_per_task`): compare probe forgetting
and **own** drift per level at 2 / 3 / 10 epochs, next to `probe_at_learn` / `acc_just_learned`.
"""),
])

write("stage1_nested_ac.ipynb", [
    md("""
# Stage 1: nested variants A and C (through my runner)

Same protocol as `stage1_nested_bd` (same benchmark, model, data order, batch size, epochs, clipping, metrics, and the
same three levels `[stem, layer1] | [layer2, layer3] | [layer4, fc]`, `slow_side: input`).

- **A (`nested_a`), periodic stepping**: levels step every 16 / 4 / 1 minibatches (input side first) on the mean of their
  accumulated gradients, all at the base lr. Same stepping as D with λ = 0, without the double backward.
  The wide version (`nested_a_1-16-256`) steps the input-side level only about 3 times per task; it is run on validation
  only, and enters the control only after the supervisor answers the ratio question (R7.4).
- **C (`nested_c_cw…`), EMA coupling**: the live network trains like Flat. After every step a shadow copy updates each
  level as an EMA of the live level, τ = 0.9375 / 0.75 / 0 (input side first, i.e. 1 − 1/k for k = 16, 4, 1). The
  **shadow is the deployed model**: it is what is evaluated, checkpointed and probed. An optional consistency term pulls
  the live logits toward the shadow's. Expect C ≈ Flat with these τ: they average over only ~16 / 4 steps out of ~790
  per task, so one longer setting (τ = 0.999 / 0.99 / 0) is also tried on validation.

Order: 0 tests → 1 smoke → 2 A regression check (vs D with λ = 0) → 3 A lr grids → 4 C lr grid, consistency grid,
long τ → 5 control (test set, 3 seeds). Every section is resumable.
"""),
    code(SETUP),
    code(NESTED_TABLE),
    md("## 0. Tests (tiny CPU configs, a few minutes)\nRun after every `git pull`. Stop here if anything fails."),
    code(r'''subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pytest"], check=True)
r = subprocess.run([sys.executable, "-m", "pytest", "tests", "-q"], cwd=CODE, capture_output=True, text=True,
                   env={**os.environ, "WANDB_MODE": "disabled", "CUDA_VISIBLE_DEVICES": ""})   # CPU: deterministic
print(r.stdout[-6000:]); print(r.stderr[-3000:])
assert r.returncode == 0, "tests failed: read the output above before running anything else"'''),
    md("""
## 1. Smoke test (2 experiences × 50 classes, 1 epoch, validation split)

Checks that A and C run on the GPU and measures cost: a full 10-epoch run trains for about **10 × `train_min`**.
For C, compare `rel_displacement` (live) with `deployed_rel_displacement` (shadow) on the slow levels: with the
teammate's τ they should be almost equal (the shadow follows the live weights closely), with the long τ the shadow
should move less.
"""),
    code(r'''LONG_TAUS = [0.0, 0.99, 0.999]   # fastest first: stem+layer1 averages over ~1000 steps, layer2+layer3 over ~100
SMOKE = [sr.arm_cfg("nested_a", stage="smoke_nested", n_experiences=2, epochs=1, lr=0.1, eval_split="val"),
         sr.arm_cfg("nested_c", stage="smoke_nested", n_experiences=2, epochs=1, lr=0.1, eval_split="val"),
         sr.arm_cfg("nested_c", stage="smoke_nested", n_experiences=2, epochs=1, lr=0.1, eval_split="val",
                    nesting=dict(ema_taus=LONG_TAUS))]
out = run_sweep(SMOKE, sr.run_experiment, paths, force=True)   # force: always re-time after code changes
smoke = table(SMOKE)
smoke["train_min"] = [sum(out[run_id(c)]["metrics"]["train_minutes_per_exp"]) if out[run_id(c)] else None
                      for c in SMOKE]
smoke["est_full_run_train_min"] = 10 * smoke["train_min"]
display(smoke.T)'''),
    md("""
## 2. Regression check: A vs D with λ = 0 (validation, seed 0, lr 0.7)

A and D with λ = 0 apply the same update (D only adds a zero penalty), so they must agree up to run-to-run GPU noise.
The D run already exists from `stage1_nested_bd`'s λ grid (avg incremental 14.18), so this costs one A run.

**Criterion (fixed before running):** average incremental accuracy within **0.5 points**. That metric averages all
210 cells of the accuracy matrix; just-learned accuracy is shown but not used as a criterion, because on a single seed it
moved by ~1 point between two runs of identical updates in B's regression check.
"""),
    code(r'''EPOCHS = 10

def val_cfg(name, lr, **kw):
    return sr.arm_cfg(name, stage="stage1_nested_lr_grid", epochs=EPOCHS, seed=0, lr=lr, eval_split="val", **kw)

D_ZERO = val_cfg("nested_d", 0.7, rule_kwargs=dict(surprise_weight=0.0))   # from stage1_nested_bd, reused
A_REG = val_cfg("nested_a", 0.7)
print(run_id(D_ZERO), sr.arm_label(D_ZERO), "| exists:", load_result(D_ZERO, paths) is not None)
out = run_sweep([A_REG], sr.run_experiment, paths)
reg = table([D_ZERO, A_REG])
display(reg[["arm", "lr", "run_id", "avg_incremental_acc", "acc_just_learned", "final_avg_acc", "minutes"]])
if reg["avg_incremental_acc"].notna().all():
    d = abs(reg["avg_incremental_acc"].iloc[0] - reg["avg_incremental_acc"].iloc[1])
    print(f"|diff| avg incremental {d:.2f}")
    print("REGRESSION CHECK", "PASSED" if d < 0.5 else "FAILED: stop and investigate")'''),
    md("""
## 3. A: learning-rate grids (validation, seed 0)

Same selection rule as every other arm: highest `avg_incremental_acc`; extend a grid whose best value is at its edge.
lr 0.7 is the regression run above (same config, reused). The wide-ratio grid is for information until the supervisor
decides on (1, 16, 256).
"""),
    code(r'''A_GRID = [0.1, 0.3, 0.5, 0.7, 0.9]
A_CONFIGS = [val_cfg("nested_a", lr) for lr in A_GRID]
A_WIDE_CONFIGS = [val_cfg("nested_a", lr, nesting=dict(periods=[1, 16, 256])) for lr in A_GRID]
for c in A_CONFIGS + A_WIDE_CONFIGS: print(run_id(c), sr.arm_label(c), c["lr"])
out = run_sweep(A_CONFIGS + A_WIDE_CONFIGS, sr.run_experiment, paths)'''),
    code(r'''def pick(configs, grid, name):
    t = table(configs)
    display(t.drop(columns=["seed", "run_id"], errors="ignore"))
    done = t.dropna(subset=["avg_incremental_acc"]) if "avg_incremental_acc" in t else t.iloc[:0]
    if len(done):
        best = done.loc[done["avg_incremental_acc"].idxmax()]
        print(f"best {name} lr: {best['lr']} (avg incremental {best['avg_incremental_acc']:.2f})")
        if best["lr"] in (min(grid), max(grid)): print(f"WARNING: best {name} lr is at the edge of its grid; extend it first.")

pick(A_CONFIGS, A_GRID, "A (1,4,16)")
pick(A_WIDE_CONFIGS, A_GRID, "A (1,16,256)")'''),
    md("""
## 4. C: lr grid (teammate's τ, consistency 0.1), then consistency grid, then one long-τ run (validation, seed 0)

C's live network trains like Flat, so its grid sits around Flat's (Flat's best was 0.1). After choosing the lr, the
consistency weight is searched over {0, 0.1, 1} at that lr (0 = pure EMA coupling as the plan describes it).
Then one run with the long τ at the chosen lr and weight. **Rule, fixed before running:** the main arm keeps the
teammate's τ unless the long τ is better by more than 0.5 points of avg incremental accuracy; if it is, take it to the
supervisor before it goes into the control (it is no longer matched to the 1/4/16 periods of A, B and D).
"""),
    code(r'''C_GRID = [0.03, 0.1, 0.2, 0.3]
C_CONFIGS = [val_cfg("nested_c", lr) for lr in C_GRID]
for c in C_CONFIGS: print(run_id(c), sr.arm_label(c), c["lr"])
out = run_sweep(C_CONFIGS, sr.run_experiment, paths)
pick(C_CONFIGS, C_GRID, "C")'''),
    code(r'''C_LR = None                       # <- fill in from the table above
CW_GRID = [0.0, 0.1, 1.0]         # 0.1 at C_LR is already in the lr grid (same config -> reused)
if C_LR is None:
    print("set C_LR first")
else:
    CW_CONFIGS = [val_cfg("nested_c", C_LR, rule_kwargs=dict(consistency_weight=w)) for w in CW_GRID]
    for c in CW_CONFIGS: print(run_id(c), sr.arm_label(c), c["lr"])
    out = run_sweep(CW_CONFIGS, sr.run_experiment, paths)
    display(table(CW_CONFIGS).drop(columns=["seed", "run_id"], errors="ignore"))'''),
    code(r'''C_CW = None                       # <- fill in from the table above (highest avg_incremental_acc)
if C_LR is None or C_CW is None:
    print("set C_LR and C_CW first")
else:
    pair = [val_cfg("nested_c", C_LR, rule_kwargs=dict(consistency_weight=C_CW)),
            val_cfg("nested_c", C_LR, rule_kwargs=dict(consistency_weight=C_CW), nesting=dict(ema_taus=LONG_TAUS))]
    out = run_sweep(pair, sr.run_experiment, paths)
    t = table(pair)
    display(t.drop(columns=["seed", "run_id"], errors="ignore").T)
    if t["avg_incremental_acc"].notna().all():
        gain = t["avg_incremental_acc"].iloc[1] - t["avg_incremental_acc"].iloc[0]
        print(f"long tau minus teammate's tau: {gain:+.2f} points avg incremental")
        print("-> take the long tau to the supervisor" if gain > 0.5 else "-> keep the teammate's tau for the control")'''),
    md("""
## 5. Control: A and C on the test set (3 seeds, checkpoints), next to Flat, B and D

Fill `LR` and `C_CW` from sections 3-4. Same stage as the B/D control (`stage1_nested_control`), so `stage1_analysis`,
`stage1_drift_diagnostic` and `stage1_structural` pick these runs up with no change. For C, checkpoints hold the
**shadow** (deployed) network, so the probes and drift measure what is evaluated. The summary asserts equal step and
parameter counts across all arms.
"""),
    code(r'''SEEDS = [0, 1, 2]
LR = {"nested_a": None, "nested_c": None}
C_CW_CONTROL = None                 # usually = C_CW
CONTROL_AC = []
for s in SEEDS:
    for name, lr in LR.items():
        if lr is None or (name == "nested_c" and C_CW_CONTROL is None):
            continue
        kw = dict(rule_kwargs=dict(consistency_weight=C_CW_CONTROL)) if name == "nested_c" else {}
        CONTROL_AC.append(sr.arm_cfg(name, stage="stage1_nested_control", epochs=EPOCHS, seed=s, lr=lr,
                                     eval_split="test", save_checkpoints=True, **kw))
for name, lr in LR.items():
    if lr is None or (name == "nested_c" and C_CW_CONTROL is None): print(f"skipping {name}: set LR / C_CW_CONTROL first")
for c in CONTROL_AC: print(run_id(c), sr.arm_label(c), c["lr"], c["seed"])'''),
    code("out = run_sweep(CONTROL_AC, sr.run_experiment, paths)"),
    code(r'''# Flat, B and D control runs from stage1_nested_bd (reused, not re-run), for one table
BD = [sr.arm_cfg(name, stage="stage1_nested_control", epochs=EPOCHS, seed=s, lr=lr, eval_split="test",
                 save_checkpoints=True, **kw)
      for s in SEEDS for name, lr, kw in [("flat", 0.1, {}), ("nested_b", 0.6, {}),
                                          ("nested_d", 0.7, dict(rule_kwargs=dict(surprise_weight=0.01)))]]
cdf = table(BD + CONTROL_AC)
done = cdf[cdf["status"] != "not finished"] if "status" in cdf else cdf
assert done["steps"].nunique() == 1 and done["n_params"].nunique() == 1, "steps or param count differ between arms. The comparison is NOT matched"
cols = ["avg_incremental_acc", "acc_just_learned", "final_avg_acc", "avg_forgetting", "minutes", "peak_gpu_mb"]
display(done.groupby(["arm", "lr"])[cols].agg(["mean", "std", "count"]).round(2))
lvl = [c for c in done.columns if c.startswith(("updates/", "rel_displacement/", "deployed_rel_displacement/", "penalty_ratio/"))]
if lvl: display(done.groupby(["arm", "lr"])[lvl].mean().T.round(4))'''),
    md("""
**Next:** re-run `stage1_analysis`, `stage1_drift_diagnostic` and `stage1_structural` (CPU). Finished analyses are
skipped, so only the new A and C runs are processed, and all arms land in one table.
"""),
])

write("stage1_structural.ipynb", [
    md("""
# Stage 1: CKA drift and gradient cosine per level (from checkpoints)

No training; CPU runtime is fine. Replaces the team's in-loop structural plugin with post-hoc measures on the saved
per-task checkpoints, using fixed image sets (definitions in `src/stage1_structural.py`):

- **`cka_to_final`** per level: how similar the final model's features of task *j*'s test images are to the features
  right after learning *j* (1 = unchanged; lower = more drift). The representation-similarity partner of probe forgetting.
- **`cka_consecutive`** per level: similarity between consecutive checkpoints on a fixed 1000-image test subset.
- **`grad_cos`** per parameter group (3 depth groups + head): cosine between the gradients of the previous task and the
  new task at the same checkpoint. Negative = the new task pulls the group against the old one.

Levels are layer1 / layer3 / layer4 (pooled), as in the probe analysis. Cost per training run: not measured yet.
"""),
    code(SETUP.replace("REQUIRE_GPU = True", "REQUIRE_GPU = False   # analysis only: CPU runtime is fine")),
    code(r"""import stage1_analysis as sa
import stage1_structural as st
STAGES = ["stage1_control", "stage1_nested_control", "stage1_epochs_per_task"]   # old control, Flat/B/D via the nested path, steps-per-task test
TRAIN_RUNS = sa.list_runs(paths, stage=STAGES)
for r in TRAIN_RUNS:
    c = r["config"]; print(r["run_id"], sr.arm_label(c), "lr", c["lr"], "seed", c["seed"])
SCFGS = st.structural_configs(TRAIN_RUNS)"""),
    code("out = run_sweep(SCFGS, st.structural_run, paths)"),
    code(r"""import numpy as np, pandas as pd
rows = []
for a in SCFGS:
    r = load_result(a, paths)
    if r is None: continue
    m = r["metrics"]; c = m["train_config"]
    row = {"arm": sr.arm_label(c), "lr": c["lr"], "seed": c["seed"]}
    for l, name in enumerate(m["level_modules"]):
        row[f"cka_to_final/{name}"] = m["cka_to_final_avg"][l]
        row[f"cka_consecutive/{name}"] = m["cka_consecutive_avg"][l]
    # group names differ between the two model classes (conv1+bn1+layer1 vs stem+layer1), so label by position
    n_g = len(m["grad_cos_avg"])
    for g, v in enumerate(m["grad_cos_avg"]):
        row["grad_cos/" + ("head" if g == n_g - 1 else f"group{g}")] = v
    rows.append(row)
sdf = pd.DataFrame(rows)
display(sdf.drop(columns="seed").groupby(["arm", "lr"]).agg(["mean", "std"]).T.round(3))"""),
    md("""
**How to read it.** The thesis predicts that slow levels keep a higher `cka_to_final` than the same levels in Flat.
Differences smaller than the seed std are not meaningful with 3 seeds. `grad_cos` near 0 or negative at a level means
consecutive tasks pull that level in unrelated or opposing directions.
"""),
])
