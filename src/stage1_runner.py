"""Stage 0/1 experiment runner (Avalanche, class-incremental Split-CIFAR-100).

To add a model or a CL method, register it in MODEL_BUILDERS / STRATEGY_BUILDERS.
The training loop, wandb logging, accuracy matrix and forgetting calculation
should not need to change.

Metric definitions (keep stable; changing them invalidates earlier runs):
- acc_matrix[i][j] = top-1 accuracy (%) on experience j's evaluation set (held-out
  validation slice if eval_split="val", real test set if "test") after
  training on experience i. Single 100-way head, no task labels (class-IL).
- final_avg_acc     = mean_j acc_matrix[T-1][j]
- avg_incremental_acc = mean_i ( mean_{j<=i} acc_matrix[i][j] )
- forgetting_j      = max_{i<T-1} acc_matrix[i][j] - acc_matrix[T-1][j], for j < T-1
- avg_forgetting    = mean_j forgetting_j
"""
import os
import random
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# The team's code (src/nested, src/models, src/strategies) imports itself as `src.…`, so the repo
# root must be importable too, not only src/ (which is how this file and fyp_infra are imported).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from src.models.resnet import ReducedResNet18 as _BlockResNet18Base  # noqa: E402

DEFAULTS = dict(
    stage="stage1",            # wandb group
    benchmark="split_cifar100",  # or "synthetic" for CPU smoke tests
    n_experiences=20,          # 100 must be divisible by this
    model="reduced_resnet18",
    nf=20,                     # ResNet base width: 20 = "reduced" (GEM/Mammoth), 64 = full
    strategy="naive",          # "naive" (= Flat) or "ewc"
    ewc_lambda=1.0,                # stability needs lr*lambda*Fisher < 2; 100 diverged on Colab
    epochs=1,                  # per experience; set by the epoch sweep
    batch_size=32,
    eval_batch_size=256,
    lr=0.03,                   # SGD lr of the fastest level (= the whole net for Flat)
    momentum=0.0,
    weight_decay=0.0,
    grad_clip_norm=1.0,        # max global grad norm; None disables. Added after EWC diverged to NaN.
    # Learning-rate schedule across depth. "flat" = one lr everywhere.
    # "llrd" = flat net, but each depth level trains at lr / level_ratios[level].
    lr_schedule="flat",
    level_ratios=(1, 4, 16),
    level_order="slow_deep",   # slow_deep: deepest level slowest; fast_deep: reversed
    head_level="fast",         # which level the classifier head belongs to: "fast" or "slow"
    # "val": train on (1 - val_fraction) of each class's training images and evaluate on the
    #        held-out rest. Use this for ALL tuning (epochs, lr, lambda, ...).
    # "test": train on the full training set, evaluate on the real test set. Final runs only.
    eval_split="val",
    val_fraction=0.1,
    seed=0,                    # controls weight init AND class order
    num_workers=2,
    synthetic_per_class=20,    # only used by benchmark="synthetic"
    # Save model weights after every experience to checkpoints/{run_id}/ (for CKA drift,
    # linear probes, gradient cosine later). ~4.5 MB x 20 per run for the reduced ResNet-18.
    save_checkpoints=False,
    # Nested arms only (strategy="nested"; see nested_builder.py). Load them with arm_cfg().
    # rule: "per_level_lr" (B; Flat = B with lr_ratios all 1) or "nested_objectives" (D).
    # nesting: {depth, slow_side: "input"|"output", lr_ratios (B) | periods (D)}, rates fastest first.
    # rule_kwargs: D only, {surprise_weight, accumulate}.
    rule=None,
    nesting=None,
    rule_kwargs=None,
)

NESTED_KEYS = ("rule", "nesting", "rule_kwargs")
NESTED_RULES = ("per_level_lr", "nested_objectives")
ARM_DIR = os.path.join(REPO_ROOT, "configs", "arm")


def _normalise_nested(cfg):
    """Canonical form of the nested fields, so equal behaviour always gives an equal run_id:
    numbers cast (JSON hashes 1 and 1.0 differently), defaults written out, and a field the
    rule would ignore is an error rather than silently hashed (a config can't claim a setting
    that does nothing)."""
    rule = cfg["rule"]
    if rule not in NESTED_RULES:
        raise ValueError(f"rule must be one of {NESTED_RULES} on this branch, got {rule!r}")
    n = dict(cfg["nesting"] or {})
    if "ordering" in n:
        raise KeyError("nesting.ordering is ambiguous; use slow_side: input|output (plan R1)")
    depth = int(n.pop("depth"))
    slow_side = n.pop("slow_side")
    if slow_side not in ("input", "output"):
        raise ValueError(f"slow_side must be 'input' or 'output', got {slow_side!r}")
    own = "lr_ratios" if rule == "per_level_lr" else "periods"
    if set(n) != {own}:
        raise KeyError(f"rule {rule} takes nesting keys depth, slow_side, {own}; got extra/missing {sorted(set(n) ^ {own})}")
    vals = [float(v) for v in n[own]] if own == "lr_ratios" else [int(v) for v in n[own]]
    if len(vals) != depth:
        raise ValueError(f"{own} needs {depth} values, got {vals}")
    if len(set(vals)) == 1:
        slow_side = "input"  # all levels equal (e.g. Flat): orientation has no effect
    cfg["nesting"] = {"depth": depth, "slow_side": slow_side, own: vals}

    kw = dict(cfg["rule_kwargs"] or {})
    if rule == "per_level_lr":
        if kw:
            raise KeyError(f"per_level_lr takes no rule_kwargs, got {kw}")
        cfg["rule_kwargs"] = {}
    else:
        out = {"surprise_weight": float(kw.pop("surprise_weight", 0.1)),
               "accumulate": kw.pop("accumulate", "mean")}
        if kw:
            raise KeyError(f"unknown rule_kwargs for {rule}: {sorted(kw)}")
        cfg["rule_kwargs"] = out
    if cfg["lr_schedule"] != "flat":
        raise ValueError("nested arms set per-level lrs through the rule; leave lr_schedule='flat'")
    if cfg["model"] != "reduced_resnet18_blocks":
        raise ValueError("nested arms need model='reduced_resnet18_blocks' (it has blocks())")


def make_cfg(**overrides):
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    cfg = dict(DEFAULTS, **overrides)
    if cfg["strategy"] == "nested":
        _normalise_nested(cfg)
    else:
        # Deleted, not set to None: a key that is present changes the hash, so keeping it
        # would change the run_id of every run made before these fields existed.
        for k in NESTED_KEYS:
            if cfg[k] is not None:
                raise ValueError(f"{k} is only used with strategy='nested'")
            del cfg[k]
    if isinstance(cfg["level_ratios"], list):
        cfg["level_ratios"] = tuple(cfg["level_ratios"])
    if cfg["lr_schedule"] == "flat":
        # These fields have no effect on a flat run; normalise them so two
        # identical flat runs never get different run_ids.
        cfg["level_ratios"], cfg["level_order"], cfg["head_level"] = None, None, None
    if cfg["strategy"] != "ewc":
        cfg["ewc_lambda"] = None
    if cfg["benchmark"] != "synthetic":
        cfg["synthetic_per_class"] = None
    if cfg["eval_split"] not in ("val", "test"):
        raise ValueError(cfg["eval_split"])
    if cfg["eval_split"] == "test" or cfg["benchmark"] == "synthetic":
        cfg["val_fraction"] = None
    if not cfg["save_checkpoints"]:
        # Saving weights doesn't change training, so leave the key out when off.
        # This keeps run_ids of runs made before this field existed unchanged
        # (they are reused, e.g. the epoch-sweep run inside the lr grid).
        del cfg["save_checkpoints"]
    return cfg


def arm_cfg(name, **overrides):
    """Config for a nested arm from configs/arm/{name}.yaml, plus overrides.

    `nesting` and `rule_kwargs` overrides are merged into the YAML's values (e.g.
    rule_kwargs=dict(surprise_weight=0.0)); everything else replaces. The YAML's `name` is for
    reading only and is not part of the config, so it never changes a run_id.
    """
    import yaml

    with open(os.path.join(ARM_DIR, f"{name}.yaml")) as f:
        arm = yaml.safe_load(f)
    cfg = dict(strategy="nested", model="reduced_resnet18_blocks", rule=arm["rule"],
               nesting={**(arm.get("nesting") or {}), **overrides.pop("nesting", {})},
               rule_kwargs={**(arm.get("rule_kwargs") or {}), **overrides.pop("rule_kwargs", {})})
    cfg.update(overrides)
    return make_cfg(**cfg)


def arm_label(cfg):
    """Short arm name for tables: 'flat', 'llrd_slow_deep', 'llrd_fast_deep' (+ ratios if not 1-4-16);
    nested path: 'flat_nl', 'nested_b', 'nested_d_lam<λ>' (+ '_slow_output', + rates if not 1-4-16).
    Display only; it does not change any config or run_id.
    NB the llrd_* labels use the OLD orientation names: llrd_fast_deep = slow_side input (plan R1)."""
    if cfg["strategy"] == "nested":
        n = cfg["nesting"]
        if cfg["rule"] == "per_level_lr":
            rates = [round(1 / r, 6) for r in n["lr_ratios"]]
            if len(set(rates)) == 1:
                return "flat_nl"
            lab = "nested_b"
        else:
            rates = n["periods"]
            lab = f"nested_d_lam{cfg['rule_kwargs']['surprise_weight']:g}"
        if n["slow_side"] != "input":
            lab += "_slow_output"
        if list(rates) != [1, 4, 16]:
            lab += "_" + "-".join(f"{r:g}" for r in rates)
        return lab
    if cfg["lr_schedule"] == "flat":
        return "flat"
    lab = f"llrd_{cfg['level_order']}"
    if list(cfg["level_ratios"]) != [1, 4, 16]:
        lab += "_" + "-".join(str(r) for r in cfg["level_ratios"])
    return lab


# ----------------------------------------------------------------------------- models

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False), nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ReducedResNet18(nn.Module):
    """ResNet-18 for 32x32 inputs: 3x3 stem, no max-pool, width nf (GEM/Mammoth variant)."""

    def __init__(self, num_classes=100, nf=20):
        super().__init__()
        self.conv1 = nn.Conv2d(3, nf, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(nf)
        self.layer1 = self._make(nf, nf, 1)
        self.layer2 = self._make(nf, nf * 2, 2)
        self.layer3 = self._make(nf * 2, nf * 4, 2)
        self.layer4 = self._make(nf * 4, nf * 8, 2)
        self.linear = nn.Linear(nf * 8, num_classes)

    @staticmethod
    def _make(in_planes, planes, stride):
        return nn.Sequential(BasicBlock(in_planes, planes, stride), BasicBlock(planes, planes, 1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.linear(out)

    def depth_groups(self):
        """Body split into 3 depth groups, shallow -> deep. Used by LLRD now and
        by the nesting levels later, so both arms cut the network the same way."""
        return [
            [self.conv1, self.bn1, self.layer1],
            [self.layer2, self.layer3],
            [self.layer4],
        ], self.linear


class SimpleCNN(nn.Module):
    """Tiny CNN, for CPU smoke tests only."""

    def __init__(self, num_classes=100, nf=8):
        super().__init__()
        self.c1 = nn.Conv2d(3, nf, 3, padding=1)
        self.c2 = nn.Conv2d(nf, nf * 2, 3, padding=1)
        self.c3 = nn.Conv2d(nf * 2, nf * 4, 3, padding=1)
        self.linear = nn.Linear(nf * 4, num_classes)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.c1(x)), 2)
        x = F.max_pool2d(F.relu(self.c2(x)), 2)
        x = F.relu(self.c3(x))
        return self.linear(F.adaptive_avg_pool2d(x, 1).flatten(1))

    def depth_groups(self):
        return [[self.c1], [self.c2], [self.c3]], self.linear


class BlockResNet18(_BlockResNet18Base):
    """The team's ReducedResNet18 (src/models/resnet.py; has blocks()/forward_blocks(), which
    NestedLearner needs). Same architecture as ReducedResNet18 above; only module names differ
    (stem = conv1+bn1+relu, fc = linear). It creates its layers in the same order, so with the
    same seed the initial weights should be identical (tests/test_nested.py checks this).

    depth_groups() is added so it is cut exactly like ReducedResNet18 by LLRD and the analysis
    code: [stem, layer1] | [layer2, layer3] | [layer4], head separate. The team's file is unchanged."""

    def depth_groups(self):
        return [
            [self.stem, self.layer1],
            [self.layer2, self.layer3],
            [self.layer4],
        ], self.fc


MODEL_BUILDERS = {
    "reduced_resnet18": lambda cfg, n_cls: ReducedResNet18(n_cls, cfg["nf"]),
    "reduced_resnet18_blocks": lambda cfg, n_cls: BlockResNet18(n_cls, cfg["nf"]),
    "simple_cnn": lambda cfg, n_cls: SimpleCNN(n_cls, cfg["nf"]),
}


# ----------------------------------------------------------------------------- optimiser

def build_param_groups(model, cfg):
    """Flat: one group. LLRD: one group per depth level with lr / ratio.

    level_ratios (1,4,16) with slow_deep means shallow body at lr, middle at
    lr/4, deep at lr/16. fast_deep reverses which end is slow. The head sits
    in the fastest or slowest level according to head_level.
    """
    if cfg["lr_schedule"] == "flat":
        return [{"params": list(model.parameters()), "lr": cfg["lr"], "level": 0}]
    if cfg["lr_schedule"] != "llrd":
        raise ValueError(cfg["lr_schedule"])
    groups, head = model.depth_groups()
    ratios = list(cfg["level_ratios"])
    if len(ratios) != len(groups):
        raise ValueError(f"need {len(groups)} level_ratios, got {ratios}")
    # ratio_for_group[g] = divisor for depth group g (g=0 shallowest)
    if cfg["level_order"] == "slow_deep":
        ratio_for_group = ratios
    elif cfg["level_order"] == "fast_deep":
        ratio_for_group = ratios[::-1]
    else:
        raise ValueError(cfg["level_order"])
    pgs = []
    for g, mods in enumerate(groups):
        params = [p for m in mods for p in m.parameters()]
        pgs.append({"params": params, "lr": cfg["lr"] / ratio_for_group[g], "level": g})
    head_ratio = min(ratios) if cfg["head_level"] == "fast" else max(ratios)
    pgs.append({"params": list(head.parameters()), "lr": cfg["lr"] / head_ratio, "level": "head"})
    n_grouped = sum(p.numel() for pg in pgs for p in pg["params"])
    assert n_grouped == sum(p.numel() for p in model.parameters()), "param groups miss parameters"
    return pgs


# ----------------------------------------------------------------------------- benchmarks

class _TensorClsDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x, self.targets = x, y.tolist()

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, i):
        return self.x[i], self.targets[i]


def _nc_benchmark():
    try:
        from avalanche.benchmarks import nc_benchmark
    except ImportError:
        from avalanche.benchmarks.scenarios.deprecated.generators import nc_benchmark
    return nc_benchmark


VAL_SPLIT_SEED = 12345  # fixed: the held-out images are identical for every run and seed


def _cifar100_val_benchmark(cfg, paths):
    """Split-CIFAR-100 whose evaluation stream is a held-out slice of the TRAINING set.

    Stratified: val_fraction of every class (50 images/class at 0.1). The split
    depends only on VAL_SPLIT_SEED, never on the run seed, so all arms and seeds
    are tuned against the same held-out images. Held-out images use the eval
    transform (no augmentation), exactly like the test set.
    """
    import copy

    from avalanche.benchmarks.classic.ccifar100 import (
        _default_cifar100_eval_transform, _default_cifar100_train_transform)
    from torchvision.datasets import CIFAR100

    full = CIFAR100(paths["data"], train=True, download=True)
    targets = np.asarray(full.targets)
    rng = np.random.RandomState(VAL_SPLIT_SEED)
    val_idx = []
    for c in range(100):
        idx = np.flatnonzero(targets == c)
        rng.shuffle(idx)
        val_idx.extend(idx[:int(round(cfg["val_fraction"] * len(idx)))])
    val_mask = np.zeros(len(targets), dtype=bool)
    val_mask[val_idx] = True

    def subset(mask):
        ds = copy.copy(full)  # shallow copy of the torchvision dataset, then slice its arrays
        ds.data = full.data[mask]
        ds.targets = targets[mask].tolist()
        return ds

    train_ds, val_ds = subset(~val_mask), subset(val_mask)
    bench = _nc_benchmark()(train_ds, val_ds, cfg["n_experiences"], task_labels=False,
                            seed=cfg["seed"],
                            train_transform=_default_cifar100_train_transform,
                            eval_transform=_default_cifar100_eval_transform)
    return bench


def build_benchmark(cfg, paths):
    if cfg["benchmark"] == "split_cifar100":
        assert 100 % cfg["n_experiences"] == 0, "100 must be divisible by n_experiences"
        if cfg["eval_split"] == "val":
            return _cifar100_val_benchmark(cfg, paths), 100
        from avalanche.benchmarks.classic import SplitCIFAR100
        return SplitCIFAR100(n_experiences=cfg["n_experiences"], seed=cfg["seed"],
                             return_task_id=False, dataset_root=paths["data"]), 100
    if cfg["benchmark"] == "synthetic":
        # Random 3x32x32 images, 2 classes per experience. Plumbing test only.
        nc_benchmark = _nc_benchmark()
        n_cls = 2 * cfg["n_experiences"]
        g = torch.Generator().manual_seed(1234)

        def make(n):
            y = torch.arange(n_cls).repeat_interleave(n)
            x = torch.randn(len(y), 3, 32, 32, generator=g) + y.view(-1, 1, 1, 1) * 0.1
            return _TensorClsDataset(x, y)

        k = cfg["synthetic_per_class"]
        return nc_benchmark(make(k), make(max(k // 2, 2)), cfg["n_experiences"],
                            task_labels=False, seed=cfg["seed"]), n_cls
    raise ValueError(cfg["benchmark"])


# ----------------------------------------------------------------------------- strategies

def _common(model, optimizer, cfg, device, evaluator, plugins):
    return dict(model=model, optimizer=optimizer, criterion=nn.CrossEntropyLoss(),
                train_mb_size=cfg["batch_size"], train_epochs=cfg["epochs"],
                eval_mb_size=cfg["eval_batch_size"], device=device,
                evaluator=evaluator, plugins=plugins)


def _naive(model, optimizer, cfg, device, evaluator, plugins):
    from avalanche.training.supervised import Naive
    return Naive(**_common(model, optimizer, cfg, device, evaluator, plugins))


def _ewc(model, optimizer, cfg, device, evaluator, plugins):
    from avalanche.training.supervised import EWC
    return EWC(ewc_lambda=cfg["ewc_lambda"], mode="separate",
               **_common(model, optimizer, cfg, device, evaluator, plugins))


STRATEGY_BUILDERS = {"naive": _naive, "ewc": _ewc}


# ----------------------------------------------------------------------------- metrics

def summarize(acc):
    """acc: T x T list of lists (percent). Returns the summary metrics above."""
    A = np.asarray(acc, dtype=float)
    T = A.shape[0]
    forg = [float(A[:T - 1, j].max() - A[T - 1, j]) for j in range(T - 1)]
    return {
        "final_avg_acc": float(A[T - 1].mean()),
        "avg_incremental_acc": float(np.mean([A[i, :i + 1].mean() for i in range(T)])),
        "avg_forgetting": float(np.mean(forg)) if forg else 0.0,
        "forgetting_per_exp": forg,
    }


def _git_commit():
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        return subprocess.check_output(["git", "-C", here, "rev-parse", "--short", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------------- main entry

def run_experiment(cfg, paths):
    """Train sequentially on every experience; return metrics dict (JSON-serialisable)."""
    import wandb
    from avalanche.core import SupervisedPlugin
    from avalanche.evaluation.metrics import accuracy_metrics
    from avalanche.training.plugins import EvaluationPlugin

    from fyp_infra import run_id

    rid = run_id(cfg)
    nested = cfg["strategy"] == "nested"
    set_seed(cfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    bench, n_cls = build_benchmark(cfg, paths)
    # Built at the same point in the RNG stream on both paths, so the same seed gives the same init.
    model = MODEL_BUILDERS[cfg["model"]](cfg, n_cls)

    class StepCounter(SupervisedPlugin):
        # Counts optimiser steps, so "matched total gradient steps" is checkable.
        def __init__(self):
            super().__init__()
            self.steps = 0

        def after_update(self, strategy, **kw):
            self.steps += 1

    class TrainGuard(SupervisedPlugin):
        # 1) Stops the run the moment the loss is NaN/inf, so a diverged run
        #    crashes (no result file, retried later) instead of saving garbage.
        # 2) Clips the global gradient norm (cfg grad_clip_norm). Nested arms clip inside the
        #    update rule instead (nested_builder.install_fresh_grad_clip), because some rules
        #    hold accumulated gradients in .grad that must not be clipped again.
        # 3) Accumulates the training loss so it can be logged per experience.
        def __init__(self):
            super().__init__()
            self.loss_sum, self.n = 0.0, 0

        def after_backward(self, strategy, **kw):
            loss = strategy.loss.detach()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite training loss ({loss.item()}) "
                                         f"at experience {strategy.experience.current_experience}")
            if cfg["grad_clip_norm"] is not None and not nested:
                torch.nn.utils.clip_grad_norm_(strategy.model.parameters(), cfg["grad_clip_norm"])
            self.loss_sum += loss.item()
            self.n += 1

    counter, guard = StepCounter(), TrainGuard()
    evaluator = EvaluationPlugin(accuracy_metrics(experience=True, stream=True), loggers=[])
    monitor = None
    if nested:
        import nested_builder as nb
        common = _common(model, None, cfg, device, evaluator, [counter, guard])
        model, rule, strategy, monitor, level_info = nb.build(cfg, model, device, common,
                                                              [counter, guard])
        ckpt_model = nb.checkpoint_module(rule)
        lr_per_group = {li["blocks"]: li["lr"] for li in level_info}
    else:
        pgs = build_param_groups(model, cfg)
        optimizer = torch.optim.SGD(pgs, lr=cfg["lr"], momentum=cfg["momentum"],
                                    weight_decay=cfg["weight_decay"])
        strategy = STRATEGY_BUILDERS[cfg["strategy"]](model, optimizer, cfg, device, evaluator,
                                                      [counter, guard])
        ckpt_model, level_info = model, None
        lr_per_group = {str(pg["level"]): pg["lr"] for pg in pgs}

    n_params = sum(p.numel() for p in model.parameters())
    run = wandb.init(project="fyp-nested-cl", group=cfg["stage"], name=rid, config=cfg,
                     reinit="finish_previous")
    run.summary["n_params"] = n_params
    T = len(bench.train_stream)
    ckpt_dir = None
    if cfg.get("save_checkpoints"):
        ckpt_dir = os.path.join(paths["checkpoints"], rid)
        os.makedirs(ckpt_dir, exist_ok=True)
    classes_per_exp = [sorted(int(c) for c in e.classes_in_this_experience)
                       for e in bench.train_stream]
    acc, train_losses, train_minutes = [], [], []
    try:
        for i, exp in enumerate(bench.train_stream):
            guard.loss_sum, guard.n = 0.0, 0
            t0 = time.time()
            strategy.train(exp, num_workers=cfg["num_workers"])
            train_minutes.append((time.time() - t0) / 60)
            train_losses.append(guard.loss_sum / max(guard.n, 1))
            if ckpt_dir is not None:
                # Weights after training experience i (incl. BatchNorm stats), written
                # atomically so a disconnect can't leave a truncated file.
                p = os.path.join(ckpt_dir, f"after_exp{i:02d}.pt")
                torch.save(ckpt_model.state_dict(), p + ".tmp")
                os.replace(p + ".tmp", p)
            res = strategy.eval(bench.test_stream, num_workers=cfg["num_workers"])
            row = [100.0 * res[f"Top1_Acc_Exp/eval_phase/test_stream/Task000/Exp{j:03d}"]
                   for j in range(T)]
            acc.append(row)
            log = {"exp": i, "acc_seen_avg": float(np.mean(row[:i + 1])),
                   "acc_current_exp": row[i], "optimizer_steps": counter.steps,
                   "train_loss": train_losses[-1], "train_minutes": train_minutes[-1]}
            log.update({f"acc_exp/{j:02d}": row[j] for j in range(i + 1)})
            if monitor is not None:
                log.update({f"level/{k}": v for k, v in monitor.rows[-1].items()})
            wandb.log(log, step=i)
            print(f"  exp {i + 1}/{T}: seen-avg {log['acc_seen_avg']:.2f}%  current {row[i]:.2f}%"
                  f"  train-loss {train_losses[-1]:.3f}")
        summary = summarize(acc)
        for k, v in summary.items():
            if not isinstance(v, list):
                run.summary[k] = v
        run.finish()
    except BaseException:
        run.finish(exit_code=1)
        raise
    peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20 if device.type == "cuda" else None
    return dict(summary, eval_split=cfg["eval_split"], acc_matrix=acc,
                train_loss_per_exp=train_losses, classes_per_exp=classes_per_exp,
                checkpoint_dir=ckpt_dir, n_params=n_params, optimizer_steps=counter.steps,
                lr_per_group=lr_per_group, level_info=level_info,
                level_log_per_exp=monitor.rows if monitor is not None else None,
                train_minutes_per_exp=train_minutes, peak_gpu_mem_mb=peak_mb,
                device=str(device), git_commit=_git_commit(), wandb_url=run.url)
