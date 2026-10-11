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
BRANCH = "main"

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
if os.path.isdir(CODE):
    subprocess.run(["git", "-C", CODE, "pull", "-q"], check=True)
else:
    subprocess.run(["git", "clone", "-q", "-b", BRANCH, url, CODE], check=True)
print(subprocess.check_output(["git", "-C", CODE, "log", "-1", "--oneline"]).decode())

# Dependencies (torch/torchvision come preinstalled with Colab's CUDA build; don't reinstall them).
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", f"{CODE}/requirements.txt"], check=True)

# wandb: key comes from the Colab secret WANDB_API_KEY (key icon in the left sidebar). Never hardcode it.
import wandb
wandb.login(key=userdata.get("WANDB_API_KEY"))

sys.path.insert(0, f"{CODE}/src")
# Python caches imported modules for the whole kernel session, so after a git pull
# a plain `import` would silently keep running the OLD code. Drop the cached copies.
for _m in ["fyp_infra", "stage1_runner", "stage1_analysis"]:
    sys.modules.pop(_m, None)
import torch
from fyp_infra import setup_paths, run_sweep, run_id, load_result
import stage1_runner as sr
paths = setup_paths()   # -> /content/drive/MyDrive/fyp_nested_cl/{data,results,logs}
print(paths)
# Hard stop: a CPU run silently takes ~3x longer and its timings are useless for budgeting.
assert torch.cuda.is_available(), "No GPU! Runtime -> Change runtime type -> T4 GPU, then restart."
print("GPU:", torch.cuda.get_device_name(0))'''

SUMMARY = r'''import numpy as np, pandas as pd
rows = []
for cfg in CONFIGS:
    r = load_result(cfg, paths)
    if r is None:
        rows.append({**{k: cfg[k] for k in SHOW}, "status": "not finished"}); continue
    m = r["metrics"]
    rows.append({**{k: cfg[k] for k in SHOW}, "run_id": r["run_id"],
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

Tunes the base learning rate for each control arm with the **same budget**: lr ∈ {0.01, 0.03, 0.1},
10 epochs, seed 0, evaluated on the held-out validation split (never the test set).

**Selection rule (fixed before looking):** pick the lr with the highest `avg_incremental_acc`.
`final_avg_acc` is useless here (with Naive training every old task ends at 0%, so it is just last-task / 20),
and forgetting can't be the criterion because it rewards learning less.
If the best lr is at the edge of the grid (0.01 or 0.1), extend the grid in that direction before choosing.

For LLRD the tuned value is the **fast-level** lr; slower levels stay at lr/4 and lr/16.
Tuning each arm separately is option (c) from the "matched effective lr" question. If your supervisor
picks option (a) instead, LLRD just reuses Flat's best lr and you can skip the LLRD rows.

The Flat lr=0.03 point is the 10-epoch run from the epoch sweep (same config, so it is reused, not re-run).
"""),
    code(SETUP),
    code(r'''EPOCHS = 10
LRS = [0.01, 0.03, 0.1]
ARMS = {
    "flat":      dict(lr_schedule="flat"),
    "flat_llrd": dict(lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="slow_deep"),
}

def grid_cfg(arm, lr):
    # The flat/0.03 point already exists as the epoch-sweep run; keeping its stage name
    # gives the identical config -> identical run_id -> run_sweep reuses that result.
    stage = "stage1_epoch_sweep" if (arm == "flat" and lr == 0.03) else "stage1_lr_grid"
    return sr.make_cfg(stage=stage, strategy="naive", epochs=EPOCHS, seed=0, lr=lr,
                       eval_split="val", **ARMS[arm])

CONFIGS = [grid_cfg(arm, lr) for arm in ARMS for lr in LRS]
SHOW = ["lr_schedule", "lr", "epochs", "eval_split"]
for c in CONFIGS: print(run_id(c), {k: c[k] for k in SHOW})'''),
    code("out = run_sweep(CONFIGS, sr.run_experiment, paths)"),
    code(SUMMARY),
    code(r'''best = df.loc[df.groupby("lr_schedule")["avg_incremental_acc"].idxmax(), ["lr_schedule", "lr", "avg_incremental_acc"]]
print(best)
for _, r in best.iterrows():
    if r["lr"] in (min(LRS), max(LRS)):
        print(f"WARNING: best lr for {r['lr_schedule']} is at the grid edge ({r['lr']}); extend the grid before choosing.")'''),
])

write("stage1_control.ipynb", [
    md("""
# Stage 1 — Deflationary control: Flat vs Flat + LLRD

The comparison that decides the project (plan §1.2). Both arms use the same network, data order (seed),
batch size and epochs, so parameter count and optimiser-step count are identical. The summary cell **asserts** this.
LLRD: depth levels shallow→deep train at lr/1, lr/4, lr/16 (`slow_deep`). The head is in the fast level.
Before running: settle the head placement and "matched effective lr" with your supervisor,
and fill `LR` from `stage1_lr_grid`. Checkpoints (~90 MB/run) go to `MyDrive/fyp_nested_cl/checkpoints/`.
Nesting arms A–D get added here after you approve the next step.
"""),
    code(SETUP),
    code(r'''EPOCHS = 10            # from the epoch sweep
SEEDS = [0, 1, 2]
# Base (fast-level) lr per arm, from stage1_lr_grid. Fill these in before running.
LR = {"flat": None, "flat_llrd": None}
assert None not in LR.values(), "set LR from stage1_lr_grid first"
ARMS = {
    "flat":      dict(lr_schedule="flat"),
    "flat_llrd": dict(lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="slow_deep"),
}
CONFIGS = [sr.make_cfg(stage="stage1_control", strategy="naive", epochs=EPOCHS, seed=s, lr=LR[name],
                       eval_split="test",          # final comparison -> real test set
                       save_checkpoints=True,      # weights after every task, for CKA / probes later
                       **arm)
           for s in SEEDS for name, arm in ARMS.items()]
SHOW = ["lr_schedule", "lr", "level_ratios", "seed", "epochs"]
for c in CONFIGS: print(run_id(c), {k: c[k] for k in SHOW})'''),
    code("out = run_sweep(CONFIGS, sr.run_experiment, paths)"),
    code(SUMMARY),
    code(r'''done = df[df.get("status") != "not finished"] if "status" in df else df
assert done["steps"].nunique() == 1 and done["n_params"].nunique() == 1, "steps or param count differ between arms. The comparison is NOT matched"
print(done.groupby("lr_schedule")[["final_avg_acc", "avg_forgetting", "avg_incremental_acc"]].agg(["mean", "std", "count"]))
print(done.groupby("lr_schedule")["acc_just_learned"].agg(["mean", "std"]))'''),
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
Expect a few minutes per training run on GPU.
"""),
    code(SETUP),
    code(r"""import stage1_analysis as sa
TRAIN_RUNS = sa.list_runs(paths, stage="stage1_control")
for r in TRAIN_RUNS:
    c = r["config"]; print(r["run_id"], c["lr_schedule"], "lr", c["lr"], "seed", c["seed"])
ACFGS = sa.analysis_configs(TRAIN_RUNS)"""),
    code("out = run_sweep(ACFGS, sa.analyze_run, paths)"),
    code(r"""import numpy as np, pandas as pd
rows = []
for a in ACFGS:
    r = load_result(a, paths)
    if r is None: continue
    m = r["metrics"]; c = m["train_config"]
    row = {"arm": c["lr_schedule"], "lr": c["lr"], "seed": c["seed"],
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
