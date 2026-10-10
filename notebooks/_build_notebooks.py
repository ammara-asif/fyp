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
for _m in ["fyp_infra", "stage1_runner"]:
    sys.modules.pop(_m, None)
import torch
from fyp_infra import setup_paths, run_sweep, run_id, load_result
import stage1_runner as sr
paths = setup_paths()   # -> /content/drive/MyDrive/fyp_nested_cl/{data,results,logs}
print(paths)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - switch runtime to GPU!")'''

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
CONFIGS = [sr.make_cfg(stage="stage0", strategy="ewc", ewc_lambda=1.0, epochs=EPOCHS, seed=s)
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
CONFIGS = [sr.make_cfg(stage="stage1_epoch_sweep", strategy="naive", epochs=e, seed=0) for e in EPOCH_GRID]
SHOW = ["epochs", "seed"]
for c in CONFIGS: print(run_id(c), {k: c[k] for k in SHOW})'''),
    code("out = run_sweep(CONFIGS, sr.run_experiment, paths)"),
    code(SUMMARY),
])

write("stage1_control.ipynb", [
    md("""
# Stage 1 — Deflationary control: Flat vs Flat + LLRD

The comparison that decides the project (plan §1.2). Both arms use the same network, data order (seed),
batch size and epochs, so parameter count and optimiser-step count are identical. The summary cell **asserts** this.
LLRD: depth levels shallow→deep train at lr/1, lr/4, lr/16 (`slow_deep`). The head is in the fast level.
Nesting arms A–D get added here after you approve the next step.
"""),
    code(SETUP),
    code(r'''EPOCHS = None          # <-- set from the epoch sweep before running
SEEDS = [0, 1, 2]
assert EPOCHS is not None, "set EPOCHS from the epoch sweep first"
ARMS = {
    "flat":      dict(lr_schedule="flat"),
    "flat_llrd": dict(lr_schedule="llrd", level_ratios=(1, 4, 16), level_order="slow_deep"),
}
CONFIGS = [sr.make_cfg(stage="stage1_control", strategy="naive", epochs=EPOCHS, seed=s, **arm)
           for s in SEEDS for arm in ARMS.values()]
SHOW = ["lr_schedule", "level_ratios", "seed", "epochs"]
for c in CONFIGS: print(run_id(c), {k: c[k] for k in SHOW})'''),
    code("out = run_sweep(CONFIGS, sr.run_experiment, paths)"),
    code(SUMMARY),
    code(r'''done = df[df.get("status") != "not finished"] if "status" in df else df
assert done["steps"].nunique() == 1 and done["n_params"].nunique() == 1, "steps or param count differ between arms. The comparison is NOT matched"
print(done.groupby("lr_schedule")[["final_avg_acc", "avg_forgetting", "avg_incremental_acc"]].agg(["mean", "std", "count"]))'''),
])
