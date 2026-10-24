"""Post-hoc analysis of saved Stage 1 checkpoints (no retraining).

Why: with Naive training in class-IL, every old task ends at 0% accuracy for
every arm, so class-IL forgetting only measures how well a task was learned.
These measures ask whether old-task knowledge survives *inside* the network
even though the shared head has overwritten it.

Metric definitions (keep stable; changing them invalidates earlier analyses):
- til_matrix[i][j]  = task-IL accuracy (%) of checkpoint i on task j's TEST images,
                      predicting only among task j's 5 classes (j <= i; NaN above).
- til_final_avg_acc = mean_j til_matrix[T-1][j]
- til_avg_forgetting = mean_{j<T-1} ( max_{j<=i<T-1} til[i][j] - til[T-1][j] )
- probe_at_learn[l][j] = accuracy (%) of a linear probe on level-l features of
                      checkpoint j (right after learning task j), trained on task j's
                      TRAIN images and tested on its TEST images (5-way).
- probe_final[l][j] = same probe protocol, but on features of the final checkpoint.
- probe_forgetting[l] = mean_{j<T-1} ( probe_at_learn[l][j] - probe_final[l][j] )
  Level-l features = global-average-pooled output of the last module in depth
  group l (same groups as LLRD/nesting: for ReducedResNet18 -> layer1, layer3, layer4).
  Probe = StandardScaler + LogisticRegression(C=1.0, max_iter=2000), fixed for all runs.
"""
import json
import os

import numpy as np
import torch

ANALYSIS_VERSION = "stage1_til_probe_v1"  # bump if any definition above changes


def load_result_by_id(rid, paths):
    with open(os.path.join(paths["results"], f"{rid}.json")) as f:
        return json.load(f)


def list_runs(paths, stage):
    """Finished training runs of `stage` (a name or a list of names) that saved checkpoints."""
    stages = {stage} if isinstance(stage, str) else set(stage)
    out = []
    for fn in sorted(os.listdir(paths["results"])):
        if not fn.endswith(".json"):
            continue
        with open(os.path.join(paths["results"], fn)) as f:
            r = json.load(f)
        cfg = r.get("config", {})
        if (r.get("status") == "finished" and cfg.get("stage") in stages
                and r.get("metrics", {}).get("checkpoint_dir")):
            out.append(r)
    return out


_DATA_CACHE = {}


def _cifar_tensors(paths, train):
    """All CIFAR-100 images of one split, with the SAME eval transform used in training
    evaluation (no augmentation), as one CPU tensor. Cached per session."""
    if train in _DATA_CACHE:
        return _DATA_CACHE[train]
    from avalanche.benchmarks.classic.ccifar100 import _default_cifar100_eval_transform
    from torchvision.datasets import CIFAR100

    ds = CIFAR100(paths["data"], train=train, download=True,
                  transform=_default_cifar100_eval_transform)
    loader = torch.utils.data.DataLoader(ds, batch_size=1000, num_workers=2)
    xs = torch.cat([x for x, _ in loader])
    ys = np.asarray(ds.targets)
    _DATA_CACHE[train] = (xs, ys)
    return xs, ys


@torch.no_grad()
def _forward(model, hook_mods, x, device, bs=500):
    """Returns logits [N, C] and a list of pooled features [N, d_l] per level."""
    feats = [[] for _ in hook_mods]
    cur = {}
    hs = [m.register_forward_hook(lambda _m, _i, o, k=k: cur.__setitem__(k, o))
          for k, m in enumerate(hook_mods)]
    logits = []
    try:
        for s in range(0, len(x), bs):
            logits.append(model(x[s:s + bs].to(device)).float().cpu())
            for k in range(len(hook_mods)):
                feats[k].append(torch.nn.functional.adaptive_avg_pool2d(cur[k], 1).flatten(1).cpu())
    finally:
        for h in hs:
            h.remove()
    return torch.cat(logits), [torch.cat(f).numpy() for f in feats]


def _probe_acc(ftr, ytr, fte, yte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
    clf.fit(ftr, ytr)
    return 100.0 * float((clf.predict(fte) == yte).mean())


def analyze_run(acfg, paths):
    """run_fn for run_sweep. acfg = {"analysis": ANALYSIS_VERSION, "train_run_id": ...}."""
    import stage1_runner as sr

    res = load_result_by_id(acfg["train_run_id"], paths)
    cfg, m = res["config"], res["metrics"]
    classes = m["classes_per_exp"]
    T = len(classes)
    ckdir = m["checkpoint_dir"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = sr.MODEL_BUILDERS[cfg["model"]](cfg, 100).to(device).eval()
    groups, _head = model.depth_groups()
    hook_mods = [g[-1] for g in groups]  # last module of each depth group
    names = {id(mod): n for n, mod in model.named_modules()}
    L = len(hook_mods)

    xtr, ytr = _cifar_tensors(paths, train=True)
    xte, yte = _cifar_tensors(paths, train=False)
    tr_idx = [np.flatnonzero(np.isin(ytr, c)) for c in classes]
    te_idx = [np.flatnonzero(np.isin(yte, c)) for c in classes]

    def load(i):
        sd = torch.load(os.path.join(ckdir, f"after_exp{i:02d}.pt"), map_location=device)
        model.load_state_dict(sd)
        model.eval()

    til = np.full((T, T), np.nan)
    probe_at_learn = np.zeros((L, T))
    probe_final = np.zeros((L, T))
    test_feats_final, train_feats_final = {}, {}

    for i in range(T):
        load(i)
        for j in range(i + 1):
            logits, fte = _forward(model, hook_mods, xte[te_idx[j]], device)
            cls = torch.tensor(classes[j])
            pred = cls[logits[:, cls].argmax(1)].numpy()  # task-IL: only task j's classes
            til[i, j] = 100.0 * float((pred == yte[te_idx[j]]).mean())
            if j == i:  # probe right after learning task j
                _, ftr = _forward(model, hook_mods, xtr[tr_idx[j]], device)
                for l in range(L):
                    probe_at_learn[l, j] = _probe_acc(ftr[l], ytr[tr_idx[j]], fte[l], yte[te_idx[j]])
            if i == T - 1:
                test_feats_final[j] = fte
        print(f"  checkpoint {i + 1}/{T}: task-IL on seen tasks {np.nanmean(til[i, :i + 1]):.1f}%")

    # probes on the final checkpoint (model is still loaded at i = T-1)
    for j in range(T):
        _, ftr = _forward(model, hook_mods, xtr[tr_idx[j]], device)
        for l in range(L):
            probe_final[l, j] = _probe_acc(ftr[l], ytr[tr_idx[j]],
                                           test_feats_final[j][l], yte[te_idx[j]])

    til_forg = [float(np.nanmax(til[j:T - 1, j]) - til[T - 1, j]) for j in range(T - 1)]
    return {
        "train_run_id": acfg["train_run_id"],
        "train_config": cfg,
        "til_matrix": [[None if np.isnan(v) else float(v) for v in row] for row in til],
        "til_final_avg_acc": float(np.mean(til[T - 1])),
        "til_avg_forgetting": float(np.mean(til_forg)),
        "probe_at_learn": probe_at_learn.tolist(),
        "probe_final": probe_final.tolist(),
        "probe_final_avg": probe_final.mean(1).tolist(),
        "probe_forgetting": [float(np.mean(probe_at_learn[l, :T - 1] - probe_final[l, :T - 1]))
                             for l in range(L)],
        "level_modules": [names[id(mm)] for mm in hook_mods],  # e.g. layer1, layer3, layer4
    }


def analysis_configs(train_runs):
    return [{"analysis": ANALYSIS_VERSION, "train_run_id": r["run_id"]} for r in train_runs]
