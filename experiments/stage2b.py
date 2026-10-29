#!/usr/bin/env python3
"""
Stage 2b — interference-triggered splitting. Any learner can split.

What changed from Stage 2, and why
----------------------------------
Stage 2 gave spawning to one designated main learner and relied on the router
to send it novel data. The router could not detect novelty, so after the first
spawn every new class was absorbed by offspring #1 and the system collapsed to
a single network.

Here there is no main learner. Every learner is equal and the first is just the
root. Spawning no longer depends on novelty detection:

  1. Training data is routed by label. A class someone owns goes to its owner.
     A class no one owns goes to the learner whose prototypes it is closest to.
     (Labels are available during training; the test-time router never sees them.)

  2. Every `check_every` steps, a learner that has taken on new classes asks:
     "would the update I am about to apply damage the classes I already own?"

       predictive  (ours)  compare the PENDING update at each level with the
                           gradient of the loss on held-out anchors of the
                           learner's established classes, BEFORE committing it.
                           Under periodic stepping the slow level's pending
                           update is the sum of 16 steps, a much steadier
                           estimate than a single step. Levels are weighted by
                           their period, so damage to slow levels counts more.
       reactive   (DEN-style) measure anchor accuracy AFTER committing, and split
                           once it has dropped by `delta`.
       never               no splitting; one plastic learner absorbs everything.

  3. On a split the child takes the new classes AND the pending update. The
     parent discards the pending update, so its slow levels are never
     overwritten, and keeps only its established classes.

  4. If no conflict appears for `establish_checks` checks, the new classes are
     absorbed and become established.

Test time: two routers are reported, both label-free.
    proto  nearest class centroid in each learner's own feature space
    conf   learner with the highest softmax confidence over its own classes

Usage
    python experiments/stage2b.py --trigger predictive --arm nested_a --stream disjoint
    python experiments/stage2b.py --fake-data --epochs 2 --n-tasks 6     # smoke test
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import SGD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from stage2 import Data, build_stream, embed, load_arm, NUM_CLASSES   # noqa: E402
from src.models.resnet import ReducedResNet18                         # noqa: E402
from src.nested.learner import NestedLearner                          # noqa: E402
from src.utils import seed_everything, count_params                   # noqa: E402


# =============================================================================
# Anchor buffer with a held-out check set
# =============================================================================

class AnchorBuffer:
    """Class-balanced reservoir of raw images.

    The first `n_check` slots of each class are a CHECK set: used to measure
    damage, never replayed. Replayed anchors get memorised, which would hide
    forgetting from the trigger.
    """

    def __init__(self, per_class: int, n_check: int):
        self.k, self.n_check = per_class, n_check
        self.x: dict[int, torch.Tensor] = {}
        self.seen: dict[int, int] = {}

    def add(self, x, y):
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

    def _gather(self, classes, part):
        xs, ys = [], []
        for c in classes:
            if c not in self.x:
                continue
            xc = self.x[c]
            xc = xc[: self.n_check] if part == "check" else xc[self.n_check:]
            if xc.size(0) == 0:
                continue
            xs.append(xc)
            ys.append(torch.full((xc.size(0),), c, dtype=torch.long, device=xc.device))
        if not xs:
            return None, None
        return torch.cat(xs), torch.cat(ys)

    def check_set(self, classes):
        return self._gather(classes, "check")

    def sample_replay(self, n):
        x, y = self._gather(list(self.x), "replay")
        if x is None or n <= 0:
            return None, None
        idx = torch.randint(0, x.size(0), (min(n, x.size(0)),), device=x.device)
        return x[idx], y[idx]

    def all(self):
        xs, ys = [], []
        for c, xc in self.x.items():
            xs.append(xc)
            ys.append(torch.full((xc.size(0),), c, dtype=torch.long, device=xc.device))
        if not xs:
            return None, None
        return torch.cat(xs), torch.cat(ys)

    def restrict(self, classes):
        nb = AnchorBuffer(self.k, self.n_check)
        for c in classes:
            if c in self.x:
                nb.x[c] = self.x[c].clone()
                nb.seen[c] = self.seen[c]
        return nb

    def nbytes(self):
        return sum(v.numel() * v.element_size() for v in self.x.values())


# =============================================================================
# A learner: network, per-level optimisers, and its control-plane row
# =============================================================================

class Learner:
    def __init__(self, uid, net: NestedLearner, lr, parent=None, born_task=0):
        self.uid, self.net, self.lr = uid, net, lr
        self.parent, self.born_task = parent, born_task
        self.opts = [SGD(list(l.parameters()), lr=lr) for l in net.levels]
        self.t = 0                          # training steps taken
        self.classes: set[int] = set()      # everything it predicts at test time
        self.established: set[int] = set()  # what it "already owns"
        self.buffer: AnchorBuffer | None = None
        self.hist = deque(maxlen=64)        # trigger values since new classes arrived
        self.base_acc = None                # reactive trigger baseline
        self.centroids = None
        self.radius = 1.0
        self.dirty = False

    @property
    def new_classes(self):
        return self.classes - self.established

    def mask(self, device):
        m = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=device)
        if self.classes:
            m[sorted(self.classes)] = True
        return m


@torch.no_grad()
def refresh(L: Learner, data: Data, chunk=512):
    x, y = L.buffer.all()
    if x is None:
        L.centroids = None
        return
    L.net.eval()
    f = torch.cat([embed(L.net, data.prep(x[i:i + chunk], False))[0]
                   for i in range(0, x.size(0), chunk)])
    cents = [F.normalize(f[y == c].mean(0), dim=0) for c in sorted(set(y.tolist()))]
    L.centroids = torch.stack(cents)
    d = (1.0 - f @ L.centroids.T).min(1).values
    L.radius = float(max(torch.quantile(d, 0.9).item(), 1e-3))
    L.dirty = False


@torch.no_grad()
def proto_score(L: Learner, xn):
    L.net.eval()
    f, logits = embed(L.net, xn)
    return (1.0 - f @ L.centroids.T).min(1).values / L.radius, logits


# =============================================================================
# The system
# =============================================================================

class System:
    def __init__(self, args, arm, data: Data, device):
        self.a, self.arm, self.data, self.dev = args, arm, data, device
        self.lr = float(args.lr if args.lr else arm.get("lr", 0.1))
        self.learners: list[Learner] = []
        self.owner: dict[int, int] = {}        # class -> learner uid
        self.uid = 0
        self.split_log, self.event_log = [], []
        self.conflicts: list[float] = []
        self.task = 0
        self._add(self._new_net(), parent=None)

    # ------------------------------------------------------------------ build
    def _new_net(self):
        bb = ReducedResNet18(num_classes=NUM_CLASSES, nf=self.a.nf)
        return NestedLearner(bb, **self.arm["nesting"]).to(self.dev)

    def _add(self, net, parent, src: Learner | None = None):
        self.uid += 1
        L = Learner(self.uid, net, self.lr, parent=parent, born_task=self.task)
        L.buffer = AnchorBuffer(self.a.anchors_per_class, self.a.n_check)
        if src is not None:
            L.t = src.t
        self.learners.append(L)
        return L

    def by_uid(self, uid):
        for L in self.learners:
            if L.uid == uid:
                return L
        raise KeyError(uid)

    # ------------------------------------------------------------------ routing (training)
    def route_train(self, xr, yr):
        """Owned classes go to their owner. An unowned class is assigned, as a
        whole, to the learner it looks most familiar to."""
        ys = yr.tolist()
        unowned = sorted({c for c in ys if c not in self.owner})
        if unowned:
            cands = [L for L in self.learners if L.centroids is not None]
            for c in unowned:
                if not cands:
                    L = self.learners[0]
                else:
                    xc = self.data.prep(xr[yr == c], False)
                    s = torch.stack([proto_score(L_, xc)[0].mean() for L_ in cands])
                    L = cands[int(s.argmin())]
                if not L.new_classes and self.a.trigger == "reactive":
                    L.base_acc = self.anchor_acc(L)
                L.classes.add(c)
                self.owner[c] = L.uid
                self.event_log.append(dict(task=self.task, event="assign", cls=c, uid=L.uid))
        groups = {}
        for i, c in enumerate(ys):
            groups.setdefault(self.owner[c], []).append(i)
        return groups

    # ------------------------------------------------------------------ training
    def train_batch(self, xr, yr):
        for uid, idx in self.route_train(xr, yr).items():
            ix = torch.tensor(idx, device=self.dev)
            self.train_on(self.by_uid(uid), xr[ix], yr[ix])

    def train_on(self, L: Learner, xr, yr):
        rx, ry = L.buffer.sample_replay(self.a.replay)
        x = xr if rx is None else torch.cat([xr, rx])
        y = yr if ry is None else torch.cat([yr, ry])
        L.net.train()
        loss = F.cross_entropy(L.net(self.data.prep(x, True)), y)
        loss.backward()                      # accumulates into pending .grad
        L.buffer.add(xr, yr)
        L.t += 1
        L.dirty = True

        child = None
        if L.t % self.a.check_every == 0:
            child = self.check(L)
        self.commit(L)
        if child is not None:
            self.commit(child)

    def commit(self, L: Learner):
        """Apply each level's pending update when its period comes round."""
        for lvl, opt in zip(L.net.levels, L.opts):
            if L.t % lvl.period:
                continue
            ps = [p for p in lvl.parameters() if p.grad is not None]
            if not ps:
                continue
            if lvl.period > 1:
                for p in ps:
                    p.grad.div_(lvl.period)
            torch.nn.utils.clip_grad_norm_(ps, self.a.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------ trigger
    def conflict(self, L: Learner):
        """Timescale-weighted conflict between each level's PENDING update and
        the gradient of the loss on held-out anchors of established classes.

        Update = -lr * pending. It increases anchor loss when
        <grad_anchor, pending> < 0, so conflict per level = max(0, -cos).
        Levels are weighted by period: slow-level damage is costly to repair.
        """
        ax, ay = L.buffer.check_set(sorted(L.established))
        if ax is None:
            return None, []
        L.net.eval()
        levels = L.net.levels
        per_level = [list(l.parameters()) for l in levels]
        flat = [p for ps in per_level for p in ps]
        loss = F.cross_entropy(L.net(self.data.prep(ax, False)), ay)
        g = torch.autograd.grad(loss, flat, allow_unused=True)
        L.net.train()
        num = den = 0.0
        per, i = [], 0
        for lvl, ps in zip(levels, per_level):
            ga, i = g[i:i + len(ps)], i + len(ps)
            pairs = [(a.flatten(), p.grad.flatten())
                     for a, p in zip(ga, ps) if a is not None and p.grad is not None]
            if not pairs:
                per.append(None)
                continue
            A = torch.cat([a for a, _ in pairs])
            B = torch.cat([b for _, b in pairs])
            c = max(0.0, -float(F.cosine_similarity(A, B, dim=0)))
            per.append(c)
            num += lvl.period * c
            den += lvl.period
        return (num / den if den else None), per

    @torch.no_grad()
    def anchor_acc(self, L: Learner):
        ax, ay = L.buffer.check_set(sorted(L.established))
        if ax is None:
            return None
        L.net.eval()
        lg = L.net(self.data.prep(ax, False)).masked_fill(~L.mask(self.dev), float("-inf"))
        L.net.train()
        return float((lg.argmax(1) == ay).float().mean())

    def check(self, L: Learner):
        new = L.new_classes
        if not new:
            L.hist.clear()
            return None

        signal = None
        if L.established and self.a.trigger == "predictive":
            signal, _ = self.conflict(L)
            if signal is not None:
                self.conflicts.append(signal)
        elif L.established and self.a.trigger == "reactive":
            acc = self.anchor_acc(L)
            if acc is not None and L.base_acc is not None:
                signal = L.base_acc - acc
        L.hist.append(signal if signal is not None else 0.0)

        thr = self.a.theta if self.a.trigger == "predictive" else self.a.delta
        if (self.a.trigger != "never" and L.established and len(L.hist) >= self.a.smooth
                and float(np.mean(list(L.hist)[-self.a.smooth:])) > thr):
            if len(self.learners) < self.a.max_modules:
                return self.split(L, new, signal)
            reason = "budget"
        else:
            reason = "no-conflict"

        if len(L.hist) >= self.a.establish_checks:
            L.established |= new
            L.hist.clear()
            L.base_acc = None
            self.event_log.append(dict(task=self.task, event="absorb", uid=L.uid,
                                       classes=sorted(new), reason=reason))
        return None

    def split(self, L: Learner, new: set, signal):
        """Child takes the new classes and the pending update; parent discards
        the update, so its slow levels are untouched, and keeps what it owned."""
        child = self._add(copy.deepcopy(L.net), parent=L.uid, src=L)
        # deepcopy does not copy Parameter.grad, so hand the pending update over explicitly
        for ps, pc in zip(L.net.parameters(), child.net.parameters()):
            pc.grad = None if ps.grad is None else ps.grad.detach().clone()
        child.classes = set(new)
        child.established = set(new)
        child.buffer = L.buffer.restrict(new)
        for c in new:
            self.owner[c] = child.uid

        L.classes = set(L.established)
        L.buffer = L.buffer.restrict(L.established)
        for p in L.net.parameters():
            p.grad = None
        L.hist.clear()
        L.base_acc = None

        refresh(L, self.data)
        refresh(child, self.data)
        self.split_log.append(dict(task=self.task, parent=L.uid, child=child.uid,
                                   classes=sorted(new), signal=signal, step=L.t))
        print(f"    split: learner {L.uid} -> child {child.uid} takes {sorted(new)} "
              f"(signal {signal:.3f})" if signal is not None else
              f"    split: learner {L.uid} -> child {child.uid} takes {sorted(new)}", flush=True)
        return child

    def end_of_task(self):
        for L in self.learners:
            if L.dirty:
                refresh(L, self.data)

    # ------------------------------------------------------------------ evaluation
    @torch.no_grad()
    def evaluate(self, test_idx, groups, chunk=500):
        cands = [L for L in self.learners if L.centroids is not None and L.classes]
        masks = torch.stack([L.mask(self.dev) for L in cands])
        res = {"proto": {}, "conf": {}}
        r_ok = {"proto": 0, "conf": 0}
        r_tot = 0
        for g in groups:
            idx = test_idx[g]
            correct = {"proto": 0, "conf": 0}
            for i in range(0, len(idx), chunk):
                x = self.data.prep(self.data.xte[idx[i:i + chunk]], False)
                y = self.data.yte[idx[i:i + chunk]]
                S, C, P = [], [], []
                for L, m in zip(cands, masks):
                    s, lg = proto_score(L, x)
                    lg = lg.masked_fill(~m, float("-inf"))
                    pr = lg.softmax(1)
                    S.append(s)
                    C.append(pr.max(1).values)
                    P.append(pr.argmax(1))
                S, C, P = torch.stack(S, 1), torch.stack(C, 1), torch.stack(P, 1)
                for name, arg in (("proto", S.argmin(1)), ("conf", C.argmax(1))):
                    pred = P.gather(1, arg[:, None]).squeeze(1)
                    correct[name] += int((pred == y).sum())
                    r_ok[name] += int(masks[arg].gather(1, y[:, None]).sum())
                r_tot += y.size(0)
            for name in res:
                res[name][g] = 100.0 * correct[name] / len(idx)
        routing = {k: 100.0 * v / max(r_tot, 1) for k, v in r_ok.items()}
        return res, routing

    def footprint(self):
        params = sum(count_params(L.net) for L in self.learners)
        mem = sum(L.buffer.nbytes() for L in self.learners)
        return params, mem


# =============================================================================

def carve_validation(data: Data, tasks, frac: float, seed: int):
    """Hold out `frac` of each task's training images, per class, as validation.

    Training uses only the remaining images. Evaluation is redirected to the
    held-out images, so thresholds can be selected without touching the real
    test set. Returns group -> indices into the new evaluation tensors.
    """
    held_by_group = {}
    for t, task in enumerate(tasks):
        idx = task["train_idx"]
        keep, hold = [], []
        for c in task["classes"]:
            ic = idx[data.ytr[idx] == c]
            g = torch.Generator().manual_seed(seed * 7919 + c * 31 + t)
            perm = ic[torch.randperm(len(ic), generator=g).to(ic.device)]
            n = max(1, int(round(frac * len(perm))))
            hold.append(perm[:n])
            keep.append(perm[n:])
        task["train_idx"] = torch.cat(keep)
        held_by_group.setdefault(task["group"], []).append(torch.cat(hold))

    parts, new_idx, off = [], {}, 0
    for gid, chunks in held_by_group.items():
        v = torch.cat(chunks)
        parts.append(v)
        new_idx[gid] = torch.arange(off, off + len(v), device=data.device)
        off += len(v)
    V = torch.cat(parts)
    data.xte, data.yte = data.xtr[V], data.ytr[V]     # real test set is no longer used
    return new_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trigger", choices=["predictive", "reactive", "never"], default="predictive")
    ap.add_argument("--stream", choices=["disjoint", "recurring"], default="disjoint")
    ap.add_argument("--arm", default="nested_a")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-tasks", type=int, default=20)
    ap.add_argument("--classes-per-task", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--nf", type=int, default=20)
    # trigger — fixed numbers, nothing derived from task size
    ap.add_argument("--check-every", type=int, default=16)
    ap.add_argument("--theta", type=float, default=0.1, help="predictive: conflict threshold")
    ap.add_argument("--delta", type=float, default=0.10, help="reactive: anchor-accuracy drop")
    ap.add_argument("--smooth", type=int, default=2, help="checks averaged before splitting")
    ap.add_argument("--establish-checks", type=int, default=6)
    ap.add_argument("--max-modules", type=int, default=20)
    ap.add_argument("--anchors-per-class", type=int, default=25)
    ap.add_argument("--n-check", type=int, default=5)
    ap.add_argument("--replay", type=int, default=16)
    ap.add_argument("--val", action="store_true",
                    help="evaluate on a validation slice carved from TRAIN; the test set is untouched")
    ap.add_argument("--val-frac", type=float, default=0.1)
    # io
    ap.add_argument("--data-root", default=str(Path.home() / ".avalanche" / "data" / "cifar100"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--fake-data", action="store_true")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seed_everything(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    arm = load_arm(args.arm)
    if arm["rule"] != "periodic":
        raise SystemExit(f"stage2b supports periodic-stepping arms only; '{args.arm}' is {arm['rule']}")

    tag = f"_th{args.theta}" if args.trigger == "predictive" else ""
    name = f"{args.trigger}_{args.arm}_{args.stream}{tag}"
    out = Path(args.out) if args.out else ROOT / "results" / "stage2b" / name / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = Data(device, args.data_root, args.fake_data, args.seed)
    tasks, test_idx = build_stream(data, args.stream, args.n_tasks, args.classes_per_task, args.seed)
    if args.val:
        test_idx = carve_validation(data, tasks, args.val_frac, args.seed)
        print(f"VALIDATION MODE: evaluating on {args.val_frac:.0%} of train held out; "
              f"test set untouched", flush=True)
    system = System(args, arm, data, device)
    print(f"{name} seed={args.seed} device={device} lr={system.lr} "
          f"levels={[l.period for l in system.learners[0].net.levels]}", flush=True)

    rows, hist = [], {"proto": {}, "conf": {}}
    seen = []
    for t, task in enumerate(tasks):
        tt = time.time()
        system.task = t
        n_conf0 = len(system.conflicts)
        idx = task["train_idx"]
        for _ in range(args.epochs):
            perm = idx[torch.randperm(len(idx), device=device)]
            for b in range(0, len(perm), args.batch_size):
                bi = perm[b:b + args.batch_size]
                system.train_batch(data.xtr[bi], data.ytr[bi])
        system.end_of_task()

        if task["group"] not in seen:
            seen.append(task["group"])
        res, routing = system.evaluate(test_idx, seen)
        for k in res:
            for g, v in res[k].items():
                hist[k].setdefault(g, []).append(v)
        params, mem = system.footprint()
        tc = system.conflicts[n_conf0:]
        row = dict(task=t, group=task["group"], n_modules=len(system.learners),
                   avg_acc=float(np.mean(list(res["proto"].values()))),
                   avg_acc_conf=float(np.mean(list(res["conf"].values()))),
                   routing_acc_test=routing["proto"], routing_acc_conf=routing["conf"],
                   splits=len(system.split_log),
                   mean_conflict=float(np.mean(tc)) if tc else None,
                   params=params, anchor_kb=mem / 1024, secs=time.time() - tt)
        rows.append(row)
        mc = f"{row['mean_conflict']:.3f}" if row["mean_conflict"] is not None else "  -  "
        print(f"  task {t:>2}  learners {row['n_modules']:>2}  "
              f"acc proto {row['avg_acc']:6.2f} conf {row['avg_acc_conf']:6.2f}  "
              f"route proto {routing['proto']:6.2f} conf {routing['conf']:6.2f}  "
              f"conflict {mc}  {row['secs']:.0f}s", flush=True)

    def forgetting(h):
        f = [max(v) - v[-1] for v in h.values() if len(v) > 1]
        return float(np.mean(f)) if f else 0.0

    cv = np.array(system.conflicts) if system.conflicts else np.array([np.nan])
    final = rows[-1]
    summary = dict(
        run=name, stage="2b", trigger=args.trigger, arm=args.arm, stream=args.stream,
        seed=args.seed, theta=args.theta, delta=args.delta, epochs=args.epochs, lr=system.lr,
        final_avg_acc=final["avg_acc"], final_avg_acc_conf=final["avg_acc_conf"],
        avg_incremental_acc=float(np.mean([r["avg_acc"] for r in rows])),
        avg_incremental_acc_conf=float(np.mean([r["avg_acc_conf"] for r in rows])),
        forgetting=forgetting(hist["proto"]), forgetting_conf=forgetting(hist["conf"]),
        n_modules=final["n_modules"], n_tasks=len(tasks),
        n_splits=len(system.split_log),
        n_absorbs=sum(e["event"] == "absorb" for e in system.event_log),
        routing_acc_test=final["routing_acc_test"], routing_acc_conf=final["routing_acc_conf"],
        params=final["params"], anchor_kb=final["anchor_kb"],
        conflict_pct={str(p): float(np.nanpercentile(cv, p)) for p in (10, 25, 50, 75, 90)},
        split_log=system.split_log, event_log=system.event_log,
        per_task=rows, per_group_history=hist,
        minutes=(time.time() - t0) / 60, fake_data=args.fake_data, val=args.val,
    )
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfinal acc proto {summary['final_avg_acc']:.2f} conf {summary['final_avg_acc_conf']:.2f}  "
          f"learners {summary['n_modules']}  splits {summary['n_splits']}  "
          f"absorbs {summary['n_absorbs']}  conflict p50 {summary['conflict_pct']['50']:.3f}  "
          f"({summary['minutes']:.1f} min) -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
