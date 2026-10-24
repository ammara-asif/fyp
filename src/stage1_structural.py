"""Post-hoc structural metrics from saved checkpoints: CKA drift and gradient cosine per level.

Replaces the team's in-loop StructuralProbePlugin (src/metrics/structural.py) for my runs (only
its CKA formula is reused):
that plugin needs no checkpoints but draws random subsets each time, probes the logits as
the last level, and cannot be re-run. Here everything comes from the per-task checkpoints,
with fixed image sets, so it is reproducible and runs on CPU for any finished arm.

Definitions (keep stable; bump STRUCT_VERSION if any changes):
- Level-l features: as in stage1_analysis, the global-average-pooled output of the last
  module of depth group l (layer1, layer3, layer4).
- cka_to_final[l][j]   = linear CKA between level-l features of checkpoint j and of the final
                         checkpoint, on task j's TEST images, j < T-1. 1 = task j's images are
                         represented the same way (up to rotation and scale); lower = more drift.
- cka_consecutive[l][i] = linear CKA(checkpoint i-1, checkpoint i) on a fixed stratified test
                         subset (CKA_PER_CLASS images per class, all 100 classes), i = 1..T-1.
- grad_cos[g][t]       = cosine between the gradients of parameter group g of the mean
                         cross-entropy loss (full 100-way head) on N_GRAD fixed TRAIN images of
                         task t-1 and of task t, both at checkpoint t-1, model in eval mode
                         (BN running stats, no weight change), t = 1..T-1. Groups g: the three
                         depth groups, then the head. Images use the eval transform.
  Positive = learning task t pushes the group the same way task t-1 did; negative = interference.
"""
import os

import numpy as np
import torch
import torch.nn.functional as F

from stage1_analysis import _cifar_tensors, _forward, load_result_by_id

STRUCT_VERSION = "stage1_structural_v1"
STRUCT_SEED = 2024   # fixed: same images for every run and arm
CKA_PER_CLASS = 10
N_GRAD = 512


def linear_cka(a, b):
    """Linear CKA of two feature matrices [n, d] (numpy). Same formula as the team's
    src/metrics/structural.py, copied rather than imported because that module also imports
    an Avalanche 0.5 plugin path; computed in float64 for stability."""
    X = torch.from_numpy(a).double()
    Y = torch.from_numpy(b).double()
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.T @ Y).norm() ** 2
    return float(hsic / ((X.T @ X).norm() * (Y.T @ Y).norm() + 1e-12))


def _group_grads(model, groups, x, y, device, bs=256):
    """Flattened gradient of the mean CE loss over (x, y), one vector per parameter group."""
    model.zero_grad(set_to_none=True)
    n = len(x)
    for s in range(0, n, bs):
        logits = model(x[s:s + bs].to(device))
        loss = F.cross_entropy(logits, torch.as_tensor(y[s:s + bs]).to(device), reduction="sum") / n
        loss.backward()
    out = [torch.cat([p.grad.flatten() for p in ps if p.grad is not None]).cpu() for ps in groups]
    model.zero_grad(set_to_none=True)
    return out


def structural_run(acfg, paths):
    """run_fn for run_sweep. acfg = {"analysis": STRUCT_VERSION, "train_run_id": ...}."""
    import stage1_runner as sr

    res = load_result_by_id(acfg["train_run_id"], paths)
    cfg, m = res["config"], res["metrics"]
    classes, ckdir = m["classes_per_exp"], m["checkpoint_dir"]
    T = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = sr.MODEL_BUILDERS[cfg["model"]](cfg, 100).to(device).eval()
    dgroups, head = model.depth_groups()
    hook_mods = [g[-1] for g in dgroups]
    names = {id(mod): n for n, mod in model.named_modules()}
    level_names = [names[id(g[-1])] for g in dgroups]
    param_groups = [[p for mod in g for p in mod.parameters()] for g in dgroups] + [list(head.parameters())]
    group_names = ["+".join(names[id(mod)] for mod in g) for g in dgroups] + [names[id(head)]]
    L = len(hook_mods)

    xtr, ytr = _cifar_tensors(paths, train=True)
    xte, yte = _cifar_tensors(paths, train=False)
    te_idx = [np.flatnonzero(np.isin(yte, c)) for c in classes]
    rng = np.random.RandomState(STRUCT_SEED)
    cka_idx = np.concatenate([rng.choice(np.flatnonzero(yte == c), CKA_PER_CLASS, replace=False)
                              for c in range(100)])
    grad_idx = [rng.choice(np.flatnonzero(np.isin(ytr, c)), N_GRAD, replace=False) for c in classes]

    def load(i):
        model.load_state_dict(torch.load(os.path.join(ckdir, f"after_exp{i:02d}.pt"),
                                         map_location=device))
        model.eval()

    cka_cons = np.full((L, T), np.nan)
    grad_cos = np.full((len(param_groups), T), np.nan)
    own_task_feats = {}   # j -> level features of checkpoint j on task j's test images
    prev_fixed = None
    for i in range(T):
        load(i)
        _, fixed = _forward(model, hook_mods, xte[cka_idx], device)
        if prev_fixed is not None:
            for l in range(L):
                cka_cons[l, i] = linear_cka(prev_fixed[l], fixed[l])
        prev_fixed = fixed
        if i < T - 1:
            _, own_task_feats[i] = _forward(model, hook_mods, xte[te_idx[i]], device)
            g_old = _group_grads(model, param_groups, xtr[grad_idx[i]], ytr[grad_idx[i]], device)
            g_new = _group_grads(model, param_groups, xtr[grad_idx[i + 1]], ytr[grad_idx[i + 1]], device)
            for g, (a, b) in enumerate(zip(g_old, g_new)):
                grad_cos[g, i + 1] = float(F.cosine_similarity(a, b, dim=0))
        print(f"  checkpoint {i + 1}/{T}")

    cka_final = np.full((L, T - 1), np.nan)   # final checkpoint is still loaded
    for j in range(T - 1):
        _, f_final = _forward(model, hook_mods, xte[te_idx[j]], device)
        for l in range(L):
            cka_final[l, j] = linear_cka(own_task_feats[j][l], f_final[l])

    def lst(a):
        return [[None if np.isnan(v) else float(v) for v in row] for row in a]

    return {
        "train_run_id": acfg["train_run_id"],
        "train_config": cfg,
        "level_modules": level_names,
        "grad_groups": group_names,
        "cka_to_final": lst(cka_final),
        "cka_to_final_avg": [float(np.mean(r)) for r in cka_final],
        "cka_consecutive": lst(cka_cons),
        "cka_consecutive_avg": [float(np.nanmean(r)) for r in cka_cons],
        "grad_cos": lst(grad_cos),
        "grad_cos_avg": [float(np.nanmean(r)) for r in grad_cos],
    }


def structural_configs(train_runs):
    return [{"analysis": STRUCT_VERSION, "train_run_id": r["run_id"]} for r in train_runs]
