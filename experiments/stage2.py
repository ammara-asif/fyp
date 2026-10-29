#!/usr/bin/env python3
"""
Stage 2 — the full system, minimal version.

    nested main learner (novelty handler)
    + autonomous spawning on competence plateau
    + warm-start cloning into offspring
    + plastic (or frozen) offspring
    + non-parametric control plane (class centroids, radii, counts — no gradients)
    + prototype routing with an abstain path back to the main learner
    + per-module anchor sets

Deliberately NOT included yet: Dream Phase, merging, ancestry tracking.

Modes
    plastic   offspring keep training on data routed to them        (the proposal)
    frozen    offspring are read-only after spawning                 (isolation baseline)
    single    one learner, no spawning, replay from a buffer holding
              the same number of raw samples per class as the anchor
              sets (memory-matched experience-replay baseline)

Streams
    disjoint   20 tasks x 5 classes, each class seen once            (standard Split-CIFAR-100)
    recurring  10 class groups seen twice; the second visit uses the
               other half of each class's training images, so nothing
               is repeated exactly. Tests whether recurring domains are
               absorbed by existing modules instead of spawning new ones.

Usage
    python experiments/stage2.py --mode plastic --stream disjoint --arm nested_a
    python experiments/stage2.py --fake-data --epochs 1 --n-tasks 6     # 1-minute smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models.resnet import ReducedResNet18          # noqa: E402
from src.nested.learner import NestedLearner           # noqa: E402
from src.nested.update_rules import build_rule         # noqa: E402
from src.utils import seed_everything, count_params    # noqa: E402

NUM_CLASSES = 100
SUPPORTED_RULES = {"periodic", "per_level_lr", "grad_accum_flat", "nested_objectives"}


# =============================================================================
# Data: whole dataset on the GPU as uint8, augmentation done on-device
# =============================================================================

class Data:
    MEAN = (0.5071, 0.4865, 0.4409)
    STD = (0.2673, 0.2564, 0.2762)

    def __init__(self, device, root: str, fake: bool, seed: int):
        self.device = device
        if fake:
            xtr, ytr, xte, yte = self._fake(seed)
        else:
            from torchvision.datasets import CIFAR100
            tr = CIFAR100(root, train=True, download=True)
            te = CIFAR100(root, train=False, download=True)
            xtr = torch.from_numpy(tr.data).permute(0, 3, 1, 2).contiguous()
            xte = torch.from_numpy(te.data).permute(0, 3, 1, 2).contiguous()
            ytr, yte = torch.tensor(tr.targets), torch.tensor(te.targets)
        self.xtr, self.ytr = xtr.to(device), ytr.to(device)
        self.xte, self.yte = xte.to(device), yte.to(device)
        self.mean = torch.tensor(self.MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(self.STD, device=device).view(1, 3, 1, 1)

    @staticmethod
    def _fake(seed):
        """Learnable random data: a fixed template per class plus noise."""
        g = torch.Generator().manual_seed(seed)
        tmpl = torch.randint(0, 256, (NUM_CLASSES, 3, 32, 32), generator=g).float()

        def make(n_per):
            y = torch.arange(NUM_CLASSES).repeat_interleave(n_per)
            noise = torch.randn(len(y), 3, 32, 32, generator=g) * 60
            x = (tmpl[y] + noise).clamp(0, 255).to(torch.uint8)
            return x, y

        xtr, ytr = make(60)
        xte, yte = make(20)
        return xtr, ytr, xte, yte

    def prep(self, x_u8: torch.Tensor, train: bool) -> torch.Tensor:
        x = x_u8.float().div_(255.0)
        if train:
            x = self._augment(x)
        return (x - self.mean) / self.std

    @staticmethod
    def _augment(x):
        """RandomCrop(32, padding=4) + RandomHorizontalFlip, per sample, on-device."""
        B, dev = x.size(0), x.device
        flip = torch.rand(B, device=dev) < 0.5
        x = torch.where(flip[:, None, None, None], x.flip(3), x)
        p = F.pad(x, (4, 4, 4, 4))
        i = torch.randint(0, 9, (B,), device=dev)
        j = torch.randint(0, 9, (B,), device=dev)
        ar = torch.arange(32, device=dev)
        rows = (i[:, None] + ar)[:, :, None]
        cols = (j[:, None] + ar)[:, None, :]
        b = torch.arange(B, device=dev)[:, None, None]
        out = p.permute(0, 2, 3, 1)[b, rows, cols]
        return out.permute(0, 3, 1, 2).contiguous()


def build_stream(data: Data, kind: str, n_tasks: int, cpt: int, seed: int):
    """Fixed class order 0..99, identical to Stage 1."""
    rng = np.random.RandomState(seed)
    ytr = data.ytr.cpu().numpy()
    yte = data.yte.cpu().numpy()
    by_class = {c: np.where(ytr == c)[0] for c in range(NUM_CLASSES)}
    test_by_class = {c: np.where(yte == c)[0] for c in range(NUM_CLASSES)}

    def idx_of(classes, part=None):
        out = []
        for c in classes:
            ix = by_class[c]
            if part is not None:
                perm = np.random.RandomState(seed * 1000 + c).permutation(ix)
                half = len(perm) // 2
                ix = perm[:half] if part == 0 else perm[half:]
            out.append(ix)
        return np.concatenate(out)

    tasks, groups = [], {}
    if kind == "disjoint":
        assert n_tasks * cpt <= NUM_CLASSES
        for t in range(n_tasks):
            cls = list(range(t * cpt, (t + 1) * cpt))
            tasks.append(dict(classes=cls, train_idx=idx_of(cls), group=t))
            groups[t] = cls
    elif kind == "recurring":
        half = n_tasks // 2
        assert half * cpt <= NUM_CLASSES
        for visit in (0, 1):
            for g in range(half):
                cls = list(range(g * cpt, (g + 1) * cpt))
                tasks.append(dict(classes=cls, train_idx=idx_of(cls, part=visit), group=g))
                groups[g] = cls
    else:
        raise ValueError(kind)

    test_idx = {g: torch.tensor(np.concatenate([test_by_class[c] for c in cls]),
                                device=data.device)
                for g, cls in groups.items()}
    for t in tasks:
        t["train_idx"] = torch.tensor(t["train_idx"], device=data.device)
    return tasks, test_idx


# =============================================================================
# Control plane pieces
# =============================================================================

class ClassBuffer:
    """Class-balanced reservoir of raw uint8 images. Used for anchor sets,
    the main learner's frontier buffer, and the replay baseline's memory."""

    def __init__(self, per_class: int):
        self.k = per_class
        self.x: dict[int, torch.Tensor] = {}
        self.seen: dict[int, int] = {}

    def add(self, x: torch.Tensor, y: torch.Tensor):
        for c in y.unique().tolist():
            xc = x[y == c]
            buf, seen = self.x.get(c), self.seen.get(c, 0)
            for k in range(xc.size(0)):
                seen += 1
                if buf is None:
                    buf = xc[k:k + 1].clone()
                elif buf.size(0) < self.k:
                    buf = torch.cat([buf, xc[k:k + 1]])
                else:
                    r = random.randrange(seen)
                    if r < self.k:
                        buf[r] = xc[k]
            self.x[c], self.seen[c] = buf, seen

    def classes(self, min_seen: int = 1):
        return [c for c, s in self.seen.items() if s >= min_seen and c in self.x]

    def restrict(self, classes) -> "ClassBuffer":
        nb = ClassBuffer(self.k)
        for c in classes:
            if c in self.x:
                nb.x[c] = self.x[c].clone()
                nb.seen[c] = self.seen[c]
        return nb

    def all(self):
        if not self.x:
            return None, None
        xs, ys = [], []
        for c, xc in self.x.items():
            xs.append(xc)
            ys.append(torch.full((xc.size(0),), c, dtype=torch.long, device=xc.device))
        return torch.cat(xs), torch.cat(ys)

    def sample(self, n: int):
        x, y = self.all()
        if x is None or n <= 0:
            return None, None
        idx = torch.randint(0, x.size(0), (min(n, x.size(0)),), device=x.device)
        return x[idx], y[idx]

    def __len__(self):
        return sum(v.size(0) for v in self.x.values())

    def nbytes(self):
        return sum(v.numel() * v.element_size() for v in self.x.values())


class Unit:
    """One learner plus its row in the control plane. Every field below the
    learner is plain statistics, updated by accumulation, never by gradients."""

    def __init__(self, uid, learner, rule, frozen, born_task, generation):
        self.uid, self.learner, self.rule, self.frozen = uid, learner, rule, frozen
        self.born_task, self.generation = born_task, generation
        self.classes: set[int] = set()
        self.buffer: ClassBuffer | None = None
        self.centroids = None        # (n_cls, d) unit vectors
        self.radius = 1.0            # 90th pct of own-anchor distance to nearest centroid
        self.fired = 0               # samples routed here
        self.trained_steps = 0
        self.dirty = False           # weights changed since last prototype refresh

    def class_mask(self, device):
        m = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=device)
        if self.classes:
            m[list(self.classes)] = True
        return m


def embed(learner, x):
    """(L2-normalised pooled layer4 features, logits) in one forward."""
    outs = learner.backbone.forward_blocks(x)
    f = outs[-2]
    if f.dim() == 4:
        f = f.mean(dim=(2, 3))
    return F.normalize(f, dim=1), outs[-1]


@torch.no_grad()
def refresh_prototypes(unit: Unit, data: Data, chunk=512):
    """Per-class centroids in the unit's OWN feature space, plus a radius that
    makes scores comparable across units with different feature geometry."""
    x, y = unit.buffer.all()
    if x is None:
        unit.centroids = None
        return
    unit.learner.eval()
    feats = torch.cat([embed(unit.learner, data.prep(x[i:i + chunk], False))[0]
                       for i in range(0, x.size(0), chunk)])
    cents = []
    for c in sorted(set(y.tolist())):
        cents.append(F.normalize(feats[y == c].mean(0), dim=0))
    unit.centroids = torch.stack(cents)
    dmin = (1.0 - feats @ unit.centroids.T).min(1).values
    unit.radius = float(max(torch.quantile(dmin, 0.9).item(), 1e-3))
    unit.dirty = False


@torch.no_grad()
def score(unit: Unit, x_norm):
    """Normalised distance to the unit's nearest class centroid (lower = more
    familiar). <= tau means 'this unit recognises the input'."""
    unit.learner.eval()
    f, logits = embed(unit.learner, x_norm)
    d = (1.0 - f @ unit.centroids.T).min(1).values
    return d / unit.radius, logits


# =============================================================================
# The system
# =============================================================================

class System:
    def __init__(self, args, arm, data: Data, device):
        self.a, self.arm, self.data, self.dev = args, arm, data, device
        self.lr = float(args.lr if args.lr else arm.get("lr", 0.1))
        self.main_learner, self.main_rule = self._new_learner()
        self.mods: list[Unit] = []
        self.owner: dict[int, int] = {}          # class -> uid that owns it
        self.frontier = ClassBuffer(args.anchors_per_class)
        self.replay = ClassBuffer(args.anchors_per_class)   # single mode only
        self.pend_x, self.pend_y = [], []
        self.win_loss = deque(maxlen=args.window)
        self.win_acc = deque(maxlen=args.window)
        self.it_since_spawn = 0
        self.global_it = 0
        self.spawn_log = []
        self.route_stats = dict(routed=0, abstained=0, correct=0, total=0)
        self.uid_counter = 0

    # ---------------------------------------------------------------- learners
    def _new_learner(self):
        backbone = ReducedResNet18(num_classes=NUM_CLASSES, nf=self.a.nf)
        learner = NestedLearner(backbone, **self.arm["nesting"]).to(self.dev)
        return learner, self._rule_for(learner)

    def _rule_for(self, learner):
        return build_rule(self.arm["rule"], learner, lr=self.lr, momentum=0.0,
                          weight_decay=0.0, grad_clip=self.a.grad_clip,
                          **(self.arm.get("rule_kwargs") or {}))

    def _step(self, learner, rule, x_norm, y):
        learner.train()
        rule.zero_grad()
        logits = learner(x_norm)
        loss = F.cross_entropy(logits, y)
        rule.backward(loss, None)
        rule.step()
        return float(loss.item()), float((logits.argmax(1) == y).float().mean().item())

    # ---------------------------------------------------------------- training
    def train_batch(self, xr, yr, task_id):
        if self.a.mode == "single":
            return self._train_single(xr, yr)
        if not self.mods:
            self.route_stats["abstained"] += xr.size(0)
            self.route_stats["total"] += xr.size(0)
            self.route_stats["correct"] += xr.size(0)
            return self._feed_main(xr, yr, task_id)

        xn = self.data.prep(xr, False)
        S = torch.stack([score(u, xn)[0] for u in self.mods], 1)
        best, arg = S.min(1)
        accept = best <= self.a.tau

        # Routing correctness during training (diagnostic only, never used by the system):
        # accepted -> chosen unit already knows this class; abstained -> no unit owns it.
        ys, acc_l, arg_l = yr.tolist(), accept.tolist(), arg.tolist()
        for y, a_, k_ in zip(ys, acc_l, arg_l):
            ok = (y in self.mods[k_].classes) if a_ else (y not in self.owner)
            self.route_stats["correct"] += int(ok)
        self.route_stats["total"] += len(ys)
        self.route_stats["routed"] += int(accept.sum())
        self.route_stats["abstained"] += int((~accept).sum())

        for k, u in enumerate(self.mods):
            m = accept & (arg == k)
            if m.any():
                u.fired += int(m.sum())
                self._train_module(u, xr[m], yr[m])
        if (~accept).any():
            self._feed_main(xr[~accept], yr[~accept], task_id)

    def _train_module(self, u: Unit, xr, yr):
        if u.frozen:
            return                           # read-only: routed data is not learned
        ax, ay = u.buffer.sample(self.a.replay)
        x = xr if ax is None else torch.cat([xr, ax])
        y = yr if ay is None else torch.cat([yr, ay])
        self._step(u.learner, u.rule, self.data.prep(x, True), y)
        u.buffer.add(xr, yr)
        for c in yr.unique().tolist():
            if c not in u.classes:
                u.classes.add(c)             # plastic modules absorb what they are taught
                self.owner.setdefault(c, u.uid)
        u.trained_steps += 1
        u.dirty = True

    def _feed_main(self, xr, yr, task_id):
        """The main learner trains only on full batches of abstained data."""
        self.pend_x.append(xr)
        self.pend_y.append(yr)
        n = sum(t.size(0) for t in self.pend_x)
        bs = self.a.batch_size
        if n < bs:
            return
        X, Y = torch.cat(self.pend_x), torch.cat(self.pend_y)
        while X.size(0) >= bs:
            xb, yb, X, Y = X[:bs], Y[:bs], X[bs:], Y[bs:]
            loss, acc = self._step(self.main_learner, self.main_rule, self.data.prep(xb, True), yb)
            self.frontier.add(xb, yb)
            self.win_loss.append(loss)
            self.win_acc.append(acc)
            self.it_since_spawn += 1
            self.global_it += 1
            self._maybe_spawn(task_id)
        self.pend_x, self.pend_y = ([X], [Y]) if X.size(0) else ([], [])

    def _train_single(self, xr, yr):
        rx, ry = self.replay.sample(self.a.batch_size)
        x = xr if rx is None else torch.cat([xr, rx])
        y = yr if ry is None else torch.cat([yr, ry])
        self._step(self.main_learner, self.main_rule, self.data.prep(x, True), y)
        self.replay.add(xr, yr)
        self.global_it += 1

    # ---------------------------------------------------------------- spawning
    def _maybe_spawn(self, task_id):
        a = self.a
        if self.it_since_spawn < a.min_iters:
            return
        full = len(self.win_acc) == self.win_acc.maxlen
        competent = full and float(np.mean(self.win_acc)) >= a.acc_target
        plateau = False
        if full:
            w = list(self.win_loss)
            h = len(w) // 2
            first, second = float(np.mean(w[:h])), float(np.mean(w[h:]))
            plateau = (first - second) / max(first, 1e-8) < a.plateau_tol
        capped = self.it_since_spawn >= a.cap_iters
        if not (competent or plateau or capped):
            return

        new_cls = [c for c in self.frontier.classes(min_seen=a.min_class_samples)
                   if c not in self.owner]
        if not new_cls:
            if capped:                       # only stray old-class trickle: discard it
                self.frontier = ClassBuffer(a.anchors_per_class)
                self.it_since_spawn = 0
                self.win_loss.clear()
                self.win_acc.clear()
            return
        reason = "competence" if competent else "plateau" if plateau else "cap"
        self._spawn(new_cls, task_id, reason)

    def _spawn(self, classes, task_id, reason):
        child = copy.deepcopy(self.main_learner)        # warm start: main keeps its weights
        for p in child.parameters():
            p.grad = None                               # don't inherit half-accumulated grads
        frozen = self.a.mode == "frozen"
        if frozen:
            for p in child.parameters():
                p.requires_grad_(False)
            rule = None
        else:
            rule = self._rule_for(child)
        self.uid_counter += 1
        u = Unit(self.uid_counter, child, rule, frozen, task_id, generation=len(self.mods))
        u.classes = set(classes)
        u.buffer = self.frontier.restrict(classes)
        for c in classes:
            self.owner[c] = u.uid
        refresh_prototypes(u, self.data)
        self.mods.append(u)
        self.spawn_log.append(dict(uid=u.uid, task=task_id, classes=sorted(classes),
                                   reason=reason, main_iters=self.it_since_spawn,
                                   global_it=self.global_it))
        print(f"    spawn #{u.uid} at task {task_id}: classes {sorted(classes)} "
              f"({reason}, after {self.it_since_spawn} main iters)", flush=True)
        self.frontier = ClassBuffer(self.a.anchors_per_class)
        self.it_since_spawn = 0
        self.win_loss.clear()
        self.win_acc.clear()

    def end_of_task(self):
        for u in self.mods:
            if u.dirty:
                refresh_prototypes(u, self.data)       # re-anchoring

    # ---------------------------------------------------------------- evaluation
    def _main_as_candidate(self):
        cls = [c for c in self.frontier.classes(1) if c not in self.owner]
        if not cls:
            return None
        u = Unit(0, self.main_learner, None, True, -1, -1)
        u.classes = set(cls)
        u.buffer = self.frontier.restrict(cls)
        refresh_prototypes(u, self.data)
        return u if u.centroids is not None else None

    @torch.no_grad()
    def evaluate(self, test_idx: dict, groups, chunk=500):
        accs, r_ok, r_tot = {}, 0, 0
        if self.a.mode == "single":
            seen = sorted(set(self.replay.classes(1)))
            mask = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=self.dev)
            mask[seen] = True
            self.main_learner.eval()
            for g in groups:
                idx = test_idx[g]
                correct = 0
                for i in range(0, len(idx), chunk):
                    x = self.data.prep(self.data.xte[idx[i:i + chunk]], False)
                    y = self.data.yte[idx[i:i + chunk]]
                    lg = self.main_learner(x).masked_fill(~mask, float("-inf"))
                    correct += int((lg.argmax(1) == y).sum())
                accs[g] = 100.0 * correct / len(idx)
            return accs, float("nan")

        cands = list(self.mods)
        mc = self._main_as_candidate()
        if mc is not None:
            cands.append(mc)
        masks = [u.class_mask(self.dev) for u in cands]
        for g in groups:
            idx = test_idx[g]
            correct = 0
            for i in range(0, len(idx), chunk):
                x = self.data.prep(self.data.xte[idx[i:i + chunk]], False)
                y = self.data.yte[idx[i:i + chunk]]
                S, preds = [], []
                for u, m in zip(cands, masks):
                    s, lg = score(u, x)
                    S.append(s)
                    preds.append(lg.masked_fill(~m, float("-inf")).argmax(1))
                S, preds = torch.stack(S, 1), torch.stack(preds, 1)
                arg = S.argmin(1)
                pred = preds.gather(1, arg[:, None]).squeeze(1)
                correct += int((pred == y).sum())
                chosen_masks = torch.stack(masks)[arg]                 # (B, 100)
                r_ok += int(chosen_masks.gather(1, y[:, None]).sum())
                r_tot += y.size(0)
            accs[g] = 100.0 * correct / len(idx)
        return accs, 100.0 * r_ok / max(r_tot, 1)

    # ---------------------------------------------------------------- accounting
    def footprint(self):
        params = count_params(self.main_learner) + sum(count_params(u.learner) for u in self.mods)
        anchor = sum(u.buffer.nbytes() for u in self.mods) + self.frontier.nbytes() + self.replay.nbytes()
        return params, anchor


# =============================================================================

def load_arm(name):
    p = ROOT / "configs" / "arm" / f"{name}.yaml"
    arm = yaml.safe_load(p.read_text())
    if arm["rule"] not in SUPPORTED_RULES:
        raise SystemExit(f"arm '{name}' uses rule '{arm['rule']}', not supported in Stage 2 "
                         f"(supported: {sorted(SUPPORTED_RULES)})")
    return arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["plastic", "frozen", "single"], default="plastic")
    ap.add_argument("--stream", choices=["disjoint", "recurring"], default="disjoint")
    ap.add_argument("--arm", default="nested_a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--classes-per-task", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None, help="default: the arm's tuned lr")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--nf", type=int, default=20)
    # control plane
    ap.add_argument("--tau", type=float, default=1.0,
                    help="route if normalised distance <= tau, else abstain to main")
    ap.add_argument("--anchors-per-class", type=int, default=20)
    ap.add_argument("--replay", type=int, default=16, help="anchors replayed per routed update")
    ap.add_argument("--acc-target", type=float, default=0.75)
    ap.add_argument("--plateau-tol", type=float, default=0.02)
    ap.add_argument("--min-class-samples", type=int, default=50)
    ap.add_argument("--window", type=int, default=0, help="0 = one epoch of one task")
    ap.add_argument("--min-iters", type=int, default=0, help="0 = two epochs of one task")
    ap.add_argument("--cap-iters", type=int, default=0, help="0 = three tasks' worth")
    # io
    ap.add_argument("--data-root", default=str(Path.home() / ".avalanche" / "data" / "cifar100"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fake-data", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = False      # speed; seeds still fix data order
    torch.backends.cudnn.benchmark = False   # True exhausted host RAM with 7 concurrent jobs
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    arm = load_arm(args.arm)

    name = f"{args.mode}_{args.arm}_{args.stream}"
    out = Path(args.out) if args.out else ROOT / "results" / "stage2" / name / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = Data(device, args.data_root, args.fake_data, args.seed)
    tasks, test_idx = build_stream(data, args.stream, args.n_tasks, args.classes_per_task, args.seed)

    iters_per_task = math.ceil(len(tasks[0]["train_idx"]) / args.batch_size)
    args.window = args.window or iters_per_task
    args.min_iters = args.min_iters or 2 * iters_per_task
    args.cap_iters = args.cap_iters or 3 * args.epochs * iters_per_task

    system = System(args, arm, data, device)
    print(f"{name} seed={args.seed} device={device} lr={system.lr} "
          f"window={args.window} min_iters={args.min_iters} cap={args.cap_iters}", flush=True)

    rows, history = [], {}
    seen_groups: list[int] = []
    for t, task in enumerate(tasks):
        tt = time.time()
        r0 = dict(system.route_stats)
        idx = task["train_idx"]
        for _ in range(args.epochs):
            perm = idx[torch.randperm(len(idx), device=device)]
            for b in range(0, len(perm), args.batch_size):
                bi = perm[b:b + args.batch_size]
                system.train_batch(data.xtr[bi], data.ytr[bi], t)
        system.end_of_task()

        if task["group"] not in seen_groups:
            seen_groups.append(task["group"])
        accs, r_test = system.evaluate(test_idx, seen_groups)
        for g, v in accs.items():
            history.setdefault(g, []).append(v)
        avg = float(np.mean(list(accs.values())))
        dr = {k: system.route_stats[k] - r0[k] for k in r0}
        r_train = 100.0 * dr["correct"] / max(dr["total"], 1)
        abst = 100.0 * dr["abstained"] / max(dr["total"], 1)
        params, anchor_bytes = system.footprint()
        row = dict(task=t, group=task["group"], n_modules=len(system.mods),
                   avg_acc=avg, routing_acc_test=r_test, routing_acc_train=r_train,
                   abstain_pct=abst, params=params, anchor_kb=anchor_bytes / 1024,
                   secs=time.time() - tt)
        rows.append(row)
        print(f"  task {t:>2}  modules {len(system.mods):>2}  acc {avg:6.2f}  "
              f"route(test) {r_test:6.2f}  route(train) {r_train:6.2f}  "
              f"abstain {abst:5.1f}%  {row['secs']:.0f}s", flush=True)

    final = rows[-1]
    forgetting = [max(h) - h[-1] for g, h in history.items() if len(h) > 1]
    summary = dict(
        run=name, mode=args.mode, stream=args.stream, arm=args.arm, seed=args.seed,
        epochs=args.epochs, lr=system.lr, tau=args.tau,
        final_avg_acc=final["avg_acc"],
        avg_incremental_acc=float(np.mean([r["avg_acc"] for r in rows])),
        forgetting=float(np.mean(forgetting)) if forgetting else 0.0,
        n_modules=final["n_modules"], n_tasks=len(tasks),
        modules_per_task=final["n_modules"] / len(tasks),
        routing_acc_test=final["routing_acc_test"],
        params=final["params"], anchor_kb=final["anchor_kb"],
        spawn_reasons={r: sum(s["reason"] == r for s in system.spawn_log)
                       for r in ("competence", "plateau", "cap")},
        spawn_log=system.spawn_log, per_task=rows,
        per_group_history=history, minutes=(time.time() - t0) / 60,
        fake_data=args.fake_data,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfinal acc {summary['final_avg_acc']:.2f}  avg-inc {summary['avg_incremental_acc']:.2f}  "
          f"forgetting {summary['forgetting']:.2f}  modules {summary['n_modules']}  "
          f"({summary['minutes']:.1f} min)  -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
