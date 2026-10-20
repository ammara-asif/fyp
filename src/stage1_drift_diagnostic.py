"""Post-hoc drift diagnostic on saved Stage 1 checkpoints (no retraining).

Question: under Flat + LLRD, layer4 trains ~10x slower than under Flat, yet its
linear-probe forgetting is the same. Where does a slow level's drift come from?
  (own)       its own weights changed,
  (inherited) its INPUTS changed, because the faster levels upstream changed,
  (BN)        its BatchNorm running statistics moved to the newest task's data
              (they update at a fixed rate, whatever the learning rate is).

Method, for every old task j < T-1, with OLD = checkpoint after task j and
NEW = final checkpoint. Features of level l are computed by chaining the depth
groups (same groups as LLRD/nesting) and global-average-pooling the output.
Each "module" carries its own weights AND its own BN statistics.

Conditions (probe accuracy, %, same probe protocol as stage1_analysis):
- OO     upstream OLD, level l OLD   (= probe right after learning j)
- ON     upstream OLD, level l NEW   (only level l's own change applied)
- NO     upstream NEW, level l OLD   (only the upstream change applied)
- NN     upstream NEW, level l NEW   (= probe on the final model)
- BNj    NEW model with ALL BN running stats re-estimated on task j's train images
- BNlast NEW model with BN stats re-estimated on the LAST task's train images.
         Control for the re-estimation procedure itself (eval-transform images,
         cumulative average) without moving the statistics to old-task data.

Derived (points, averaged over j < T-1; definitions kept stable):
- total_drift[l]       = OO - NN        (equals stage1_analysis probe_forgetting)
- own_drift[l]         = OO - ON
- inherited_drift[l]   = OO - NO
- interaction[l]       = total - own - inherited   (the two changes are not additive)
- bn_recovery[l]       = BNj - BNlast   (how much putting old-task BN stats back helps)
- bn_procedure_effect  = BNlast - NN    (should be ~0; if not, read bn_recovery with care)
Level 0 has no upstream: own = total, inherited = interaction = 0 by definition.

BNj uses old-task data, so it is a diagnostic only, never a method.
"""
import copy
import os

import numpy as np
import torch
import torch.nn.functional as F

from stage1_analysis import _cifar_tensors, _probe_acc, load_result_by_id

DIAGNOSTIC_VERSION = "stage1_drift_v1"  # bump if any definition above changes
CONDITIONS = ["OO", "ON", "NO", "NN", "BNj", "BNlast"]


def resnet_stages(m):
    """The 3 depth groups of ReducedResNet18 as functions, matching depth_groups():
    [stem + layer1] -> [layer2 + layer3] -> [layer4]. Outputs are unpooled feature maps."""
    if not hasattr(m, "layer4"):
        raise NotImplementedError("drift diagnostic is written for ReducedResNet18")
    return [lambda x: m.layer1(F.relu(m.bn1(m.conv1(x)))),
            lambda h: m.layer3(m.layer2(h)),
            lambda h: m.layer4(h)]


def _pool(t):
    return F.adaptive_avg_pool2d(t, 1).flatten(1).cpu()


@torch.no_grad()
def reestimate_bn(model, x, device, bs=500):
    """Copy of `model` whose BN running mean/var are recomputed on images `x`
    (cumulative average over batches). Weights are untouched."""
    m = copy.deepcopy(model)
    bns = [mod for mod in m.modules() if isinstance(mod, torch.nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats()
        b.momentum = None  # None = exact cumulative average instead of an EMA
    m.train()
    for s in range(0, len(x), bs):
        m(x[s:s + bs].to(device))
    return m.eval()


@torch.no_grad()
def _condition_features(old, new, bnj, bnlast, x, device, bs=500):
    """Pooled level features for every condition, computed batch by batch so the
    large unpooled maps are never stored. Returns {cond: [np.array per level or None]}."""
    so, sn, sj, sl = (resnet_stages(m) for m in (old, new, bnj, bnlast))
    L = len(so)
    out = {c: [[] for _ in range(L)] for c in CONDITIONS}

    def chain(stages, xb):
        hs, h = [], xb
        for st in stages:
            h = st(h)
            hs.append(h)
        return hs

    for s in range(0, len(x), bs):
        xb = x[s:s + bs].to(device)
        ho, hn = chain(so, xb), chain(sn, xb)
        hj, hl = chain(sj, xb), chain(sl, xb)
        for l in range(L):
            out["OO"][l].append(_pool(ho[l]))
            out["NN"][l].append(_pool(hn[l]))
            out["BNj"][l].append(_pool(hj[l]))
            out["BNlast"][l].append(_pool(hl[l]))
            if l > 0:
                out["ON"][l].append(_pool(sn[l](ho[l - 1])))  # old inputs -> new level
                out["NO"][l].append(_pool(so[l](hn[l - 1])))  # new inputs -> old level
    return {c: [torch.cat(v).numpy() if v else None for v in per_level]
            for c, per_level in out.items()}


def diagnose_run(acfg, paths):
    """run_fn for run_sweep. acfg = {"analysis": DIAGNOSTIC_VERSION, "train_run_id": ...}."""
    import stage1_runner as sr

    res = load_result_by_id(acfg["train_run_id"], paths)
    cfg, m = res["config"], res["metrics"]
    classes, ckdir = m["classes_per_exp"], m["checkpoint_dir"]
    T = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load(i):
        mdl = sr.MODEL_BUILDERS[cfg["model"]](cfg, 100).to(device)
        mdl.load_state_dict(torch.load(os.path.join(ckdir, f"after_exp{i:02d}.pt"),
                                       map_location=device))
        return mdl.eval()

    new = load(T - 1)
    groups, _ = new.depth_groups()
    names = {id(mod): n for n, mod in new.named_modules()}
    level_names = [names[id(g[-1])] for g in groups]
    L = len(level_names)

    xtr, ytr = _cifar_tensors(paths, train=True)
    xte, yte = _cifar_tensors(paths, train=False)
    tr_idx = [np.flatnonzero(np.isin(ytr, c)) for c in classes]
    te_idx = [np.flatnonzero(np.isin(yte, c)) for c in classes]

    bnlast = reestimate_bn(new, xtr[tr_idx[T - 1]], device)
    acc = {c: np.full((L, T - 1), np.nan) for c in CONDITIONS}
    for j in range(T - 1):
        old = load(j)
        bnj = reestimate_bn(new, xtr[tr_idx[j]], device)
        ftr = _condition_features(old, new, bnj, bnlast, xtr[tr_idx[j]], device)
        fte = _condition_features(old, new, bnj, bnlast, xte[te_idx[j]], device)
        for c in CONDITIONS:
            for l in range(L):
                if ftr[c][l] is not None:
                    acc[c][l, j] = _probe_acc(ftr[c][l], ytr[tr_idx[j]], fte[c][l], yte[te_idx[j]])
        print(f"  task {j + 1}/{T - 1}: deepest-level drift {acc['OO'][-1, j] - acc['NN'][-1, j]:.1f} pts")

    mean = {c: np.nanmean(acc[c], axis=1) for c in CONDITIONS}
    total = mean["OO"] - mean["NN"]
    own, inherited = mean["OO"] - mean["ON"], mean["OO"] - mean["NO"]
    own[0], inherited[0] = total[0], 0.0  # level 0 has no upstream
    interaction = total - own - inherited

    def lst(a):
        return [float(v) for v in a]

    return {
        "train_run_id": acfg["train_run_id"],
        "train_config": cfg,
        "level_modules": level_names,
        "probe_mean": {c: [None if np.isnan(v) else float(v) for v in mean[c]] for c in CONDITIONS},
        "probe_per_task": {c: [[None if np.isnan(v) else float(v) for v in row] for row in acc[c]]
                           for c in CONDITIONS},
        "total_drift": lst(total),
        "own_drift": lst(own),
        "inherited_drift": lst(inherited),
        "interaction": lst(interaction),
        "bn_recovery": lst(mean["BNj"] - mean["BNlast"]),
        "bn_procedure_effect": lst(mean["BNlast"] - mean["NN"]),
    }


def diagnostic_configs(train_runs):
    return [{"analysis": DIAGNOSTIC_VERSION, "train_run_id": r["run_id"]} for r in train_runs]
