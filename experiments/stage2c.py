#!/usr/bin/env python3
"""
Stage 2c — ownership-aware learners, referral routing, per-class placement and splitting.

Builds on Stage 2b (interference-triggered splitting, any learner can split).
Two changes, aimed at the routing collapse seen in 2b:

1. Every learner learns what is NOT its own
   - an extra "mine" output, trained with three kinds of examples:
       own images                          -> mine
       other learners' stored images       -> not mine   (foreign negatives)
       its own images with patches shuffled -> not mine  (self-damaged negatives:
         same colours and textures, structure destroyed, so "mine" has to mean
         the object's structure; this is what lets rejection extend to classes
         the learner has never seen)
   - its 100-way head also trains lightly on other learners' stored images, so a
     non-owner can say WHOSE an image is (a referral)
   - every `refresh_every` steps, and after every split, each learner's two HEADS
     (100-way and 'mine') are refreshed on all stored images with the features
     frozen (--refresh-scope head). No learner's representation is ever changed
     by another learner's classes.

   Test-time routing (label-free):
       score(learner k) = P_k(mine) + mean over other learners of the
                          probability mass they refer to k's classes
       P(learner | x)   = softmax(score / T)
       P(class | x)     = sum_k P(k | x) * P(class | x, learner k)      ("soft")
   The final router multiplies four signals (see evaluate()): own, referral
   consensus, a shared-space model in the root's slow levels (read-only), and a
   likelihood ratio in each learner's own space. Each is also reported alone,
   plus ORACLE (each image sent to its true owner: the ceiling).

2. Classes, not tasks, are the unit
   - a new class goes to the learner it CONFLICTS LEAST with (gradient of the new
     class vs the gradient on that learner's held-out anchors, weighted by level
     period). If it conflicts with every learner beyond theta, it starts a new
     learner; new classes arriving together that agree with each other share one.
   - the predictive check is PER CLASS: absorb the dog, split off the truck.
     Classes split off together are grouped by whether their gradients agree.

Fully task-free: nothing here uses task boundaries (no end-of-task hooks).

    python experiments/stage2c.py --trigger predictive --arm nested_a --stream disjoint
    python experiments/stage2c.py --fake-data --epochs 2 --n-tasks 6
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
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from stage2 import Data, build_stream, load_arm, NUM_CLASSES   # noqa: E402
from stage2b import AnchorBuffer, carve_validation              # noqa: E402
from src.models.resnet import ReducedResNet18                  # noqa: E402
from src.nested.learner import NestedLearner                   # noqa: E402
from src.utils import seed_everything, count_params            # noqa: E402


# =============================================================================
# Network with a "mine" output on the fast level
# =============================================================================

class OwnedResNet(ReducedResNet18):
    def __init__(self, num_classes=100, nf=20):
        super().__init__(num_classes=num_classes, nf=nf)
        self.own = nn.Linear(nf * 8, 1)


def build_net(arm, nf, device):
    bb = OwnedResNet(num_classes=NUM_CLASSES, nf=nf)
    net = NestedLearner(bb, **arm["nesting"])
    fast = min(net.levels, key=lambda l: l.period)
    fast.modules.append(bb.own)            # the "mine" head learns at the fast timescale
    return net.to(device)


def forward_all(net, x):
    """(pooled features, 100-way logits, 'mine' logit) in one pass."""
    outs = net.backbone.forward_blocks(x)
    f = outs[-2]
    if f.dim() == 4:
        f = f.mean(dim=(2, 3))
    return f, outs[-1], net.backbone.own(f).squeeze(1)


def patch_shuffle(x, g):
    """Cut each image into g x g patches and shuffle them: textures survive, structure doesn't."""
    B, C, H, W = x.shape
    ph, pw = H // g, W // g
    p = x.reshape(B, C, g, ph, g, pw).permute(0, 2, 4, 1, 3, 5).reshape(B, g * g, C, ph, pw)
    perm = torch.stack([torch.randperm(g * g, device=x.device) for _ in range(B)])
    p = p[torch.arange(B, device=x.device)[:, None], perm]
    return p.reshape(B, g, g, C, ph, pw).permute(0, 3, 1, 4, 2, 5).reshape(B, C, H, W)


def self_damage(x):
    if x.size(0) == 0:
        return x
    return patch_shuffle(x, random.choice([2, 4]))


ROUTERS = ["soft", "soft_oc", "own", "consensus", "shared", "lr", "oracle", "taskil"]
# oracle : each image sent to its true owner, which answers among ITS classes.
#          The ceiling routing is chasing. Not comparable across systems with
#          different numbers of learners (a 5-way quiz vs a 100-way one).
# taskil : the task's 5 classes are given; each class is scored by its owner.
#          The same 5-way quiz for every system, including the single learner,
#          so this is the one to compare RETENTION with (forgetting_taskil) and
#          how well new tasks are learned (new_task_acc_taskil).


# =============================================================================
# Learner
# =============================================================================

class Learner:
    def __init__(self, uid, net, lr, parent=None, born_task=0):
        self.uid, self.net, self.lr = uid, net, lr
        self.parent, self.born_task = parent, born_task
        self.opts = [SGD(list(l.parameters()), lr=lr) for l in net.levels]
        self.t = 0
        self.classes: set[int] = set()
        self.established: set[int] = set()
        self.buffer: AnchorBuffer | None = None
        self.class_hist: dict[int, deque] = {}
        self.hist = deque(maxlen=64)       # reactive trigger only
        self.base_acc = None
        self.born_t = 0

    @property
    def new_classes(self):
        return self.classes - self.established


# =============================================================================
# System
# =============================================================================

class System:
    def __init__(self, args, arm, data: Data, device):
        self.a, self.arm, self.data, self.dev = args, arm, data, device
        self.lr = float(args.lr if args.lr else arm.get("lr", 0.1))
        self.learners: list[Learner] = []
        self.owner: dict[int, int] = {}
        self.uid = 0
        self.gstep = 0
        self.task = 0
        self.split_log, self.event_log, self.conflicts = [], [], []
        self._add(build_net(arm, args.nf, device), parent=None)

    # ------------------------------------------------------------------ helpers
    def _add(self, net, parent, src: Learner | None = None):
        self.uid += 1
        L = Learner(self.uid, net, self.lr, parent=parent, born_task=self.task)
        L.buffer = AnchorBuffer(self.a.anchors_per_class, self.a.n_check)
        if src is not None:
            L.t = src.t
        L.born_t = L.t
        self.learners.append(L)
        return L

    def slots(self):
        """Learners that may still be created; --max-modules 0 means no cap."""
        if self.a.max_modules <= 0:
            return 10 ** 6
        return self.a.max_modules - len(self.learners)

    def by_uid(self, uid):
        for L in self.learners:
            if L.uid == uid:
                return L
        raise KeyError(uid)

    def prep(self, x, train):
        return self.data.prep(x, train)

    def class_images(self, L, c, n):
        x, y = L.buffer._gather([c], "replay")
        if x is None:
            x, y = L.buffer._gather([c], "check")
        if x is None:
            return None, None
        if x.size(0) > n:
            idx = torch.randperm(x.size(0), device=x.device)[:n]
            x, y = x[idx], y[idx]
        return x, y

    def foreign(self, L, n):
        xs, ys = [], []
        for M in self.learners:
            if M is L or not M.buffer.x:
                continue
            x, y = M.buffer._gather(sorted(M.buffer.x), "replay")
            if x is not None:
                xs.append(x)
                ys.append(y)
        if not xs or n <= 0:
            return None, None
        X, Y = torch.cat(xs), torch.cat(ys)
        idx = torch.randint(0, X.size(0), (min(n, X.size(0)),), device=X.device)
        return X[idx], Y[idx]

    # ------------------------------------------------------------------ gradients
    def grads(self, L, xn, y):
        """Per level, per parameter, gradient of classification loss on (xn, y)."""
        L.net.eval()
        levels = L.net.levels
        per_level = [list(l.parameters()) for l in levels]
        flat = [p for ps in per_level for p in ps]
        loss = F.cross_entropy(L.net(xn), y)
        g = torch.autograd.grad(loss, flat, allow_unused=True)
        L.net.train()
        out, i = [], 0
        for ps in per_level:
            out.append(list(g[i:i + len(ps)]))
            i += len(ps)
        return out

    def pending(self, L):
        return [[p.grad for p in l.parameters()] for l in L.net.levels]

    def anchor_grads(self, L, classes):
        ax, ay = L.buffer.check_set(sorted(classes))
        if ax is None:
            return None
        return self.grads(L, self.prep(ax, False), ay)

    @staticmethod
    def weighted_conflict(L, A, B):
        """Period-weighted mean over levels of max(0, -cos(A, B)).
        A update along -B raises the loss behind A when cos(A, B) < 0."""
        num = den = 0.0
        for lvl, a_l, b_l in zip(L.net.levels, A, B):
            pairs = [(a.flatten(), b.flatten()) for a, b in zip(a_l, b_l)
                     if a is not None and b is not None]
            if not pairs:
                continue
            va = torch.cat([p for p, _ in pairs])
            vb = torch.cat([q for _, q in pairs])
            c = max(0.0, -float(F.cosine_similarity(va, vb, dim=0)))
            num += lvl.period * c
            den += lvl.period
        return num / den if den else 0.0

    @staticmethod
    def flat_vec(L, G):
        parts = []
        for lvl, g_l in zip(L.net.levels, G):
            v = [g.flatten() for g in g_l if g is not None]
            if v:
                v = torch.cat(v)
                parts.append(F.normalize(v, dim=0) * math.sqrt(lvl.period))
        return torch.cat(parts)

    def group_by_agreement(self, vecs: dict, order):
        """Greedy: a class joins the group whose mean gradient it agrees with (cos >= 0)."""
        groups = []
        for c in order:
            best, best_cos = None, 0.0
            for g in groups:
                m = torch.stack([vecs[x] for x in g]).mean(0)
                cs = float(F.cosine_similarity(vecs[c], m, dim=0))
                if cs >= best_cos:
                    best, best_cos = g, cs
            if best is None:
                groups.append([c])
            else:
                best.append(c)
        return groups

    @staticmethod
    def cap_groups(groups, slots):
        groups = sorted(groups, key=len, reverse=True)
        while len(groups) > max(slots, 1):
            last = groups.pop()
            groups[-1].extend(last)
        return groups

    # ------------------------------------------------------------------ placement
    def route_train(self, xr, yr):
        ys = yr.tolist()
        unowned = sorted({c for c in ys if c not in self.owner})
        if unowned:
            self.place(unowned, xr, yr)
        groups = {}
        for i, c in enumerate(ys):
            groups.setdefault(self.owner[c], []).append(i)
        return groups

    def place(self, unowned, xr, yr):
        """Each new class goes where it conflicts least; classes that conflict with
        every learner start a new one, grouped by mutual agreement."""
        a = self.a
        cands = [L for L in self.learners if L.established]
        if not cands:                                   # root still forming
            for c in unowned:
                self._assign(self.learners[0], c)
            return

        anchor = {L.uid: self.anchor_grads(L, L.established) for L in cands}
        scores_of, worst = {}, []                      # class -> conflict with each candidate
        for c in unowned:
            xc = self.prep(xr[yr == c], False)
            yc = torch.full((xc.size(0),), c, dtype=torch.long, device=self.dev)
            s = []
            for L in cands:
                A = anchor[L.uid]
                s.append(0.0 if A is None else self.weighted_conflict(L, A, self.grads(L, xc, yc)))
            scores_of[c] = s
            if a.trigger == "predictive" and min(s) > a.theta:
                worst.append(c)

        slots = self.slots()
        for c in unowned:
            if c not in worst or slots <= 0:
                self._assign(cands[int(np.argmin(scores_of[c]))], c)

        if worst and slots > 0:
            # group the classes no learner can absorb, measured in one common network:
            # the candidate they conflict with least on average
            k_ref = int(np.argmin(np.mean([scores_of[c] for c in worst], axis=0)))
            ref = cands[k_ref]
            vecs = {}
            for c in worst:
                xc = self.prep(xr[yr == c], False)
                yc = torch.full((xc.size(0),), c, dtype=torch.long, device=self.dev)
                vecs[c] = self.flat_vec(ref, self.grads(ref, xc, yc))
            groups = self.cap_groups(self.group_by_agreement(vecs, worst), slots)
            for grp in groups:
                # warm-start the new learner from the most compatible existing one
                k = int(np.argmin(np.mean([scores_of[c] for c in grp], axis=0)))
                parent = cands[k]
                child = self._add(copy.deepcopy(parent.net), parent=parent.uid, src=parent)
                for p in child.net.parameters():
                    p.grad = None
                for c in grp:
                    self._assign(child, c)
                self.split_log.append(dict(task=self.task, kind="spawn-on-arrival",
                                           parent=parent.uid, child=child.uid,
                                           classes=sorted(grp), step=self.gstep))
                print(f"    new learner {child.uid} (from {parent.uid}) for classes "
                      f"{sorted(grp)}: conflicts with every learner", flush=True)

    def _assign(self, L, c):
        if not L.new_classes and self.a.trigger == "reactive":
            L.base_acc = self.anchor_acc(L)
        L.classes.add(c)
        self.owner[c] = L.uid
        self.event_log.append(dict(task=self.task, event="assign", cls=c, uid=L.uid))

    # ------------------------------------------------------------------ training
    def train_batch(self, xr, yr):
        for uid, idx in self.route_train(xr, yr).items():
            ix = torch.tensor(idx, device=self.dev)
            self.train_on(self.by_uid(uid), xr[ix], yr[ix])
        self.gstep += 1
        if self.a.refresh_every and self.gstep % self.a.refresh_every == 0:
            for L in self.learners:
                self.refresh_fast(L)

    def ownership_loss(self, L, xo_n, yo, fx, fy, heads_only=False):
        """Classification on own data + referral on foreign data + 'mine' on all three.
        heads_only: features are computed without gradient, so only the 100-way head
        and the 'mine' head can change (the representation is untouched)."""
        a = self.a
        parts = [xo_n]
        nf = 0
        if fx is not None:
            parts.append(self.prep(fx, True))
            nf = fx.size(0)
        xd = self_damage(xo_n[: min(a.dmg, xo_n.size(0))])
        parts.append(xd)
        X = torch.cat(parts)
        if heads_only:
            with torch.no_grad():
                f, _, _ = forward_all(L.net, X)
            logits = L.net.backbone.fc(f)
            own = L.net.backbone.own(f).squeeze(1)
        else:
            _, logits, own = forward_all(L.net, X)
        no = xo_n.size(0)
        loss = F.cross_entropy(logits[:no], yo)
        if nf:
            loss = loss + a.lam_ref * F.cross_entropy(logits[no:no + nf], fy)
        pos = own[:no]
        neg = own[no:]
        loss = loss + a.lam_own * 0.5 * (
            F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
            + F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg)))
        return loss

    def train_on(self, L, xr, yr):
        a = self.a
        rx, ry = L.buffer.sample_replay(a.replay)
        xo = xr if rx is None else torch.cat([xr, rx])
        yo = yr if ry is None else torch.cat([yr, ry])
        fx, fy = self.foreign(L, a.neg)
        L.net.train()
        loss = self.ownership_loss(L, self.prep(xo, True), yo, fx, fy)
        # Stage 1 protocol: clip each minibatch's FRESH gradient before it joins a
        # level's accumulated (pending) gradient. Clipping the accumulated buffer at
        # step time would scale a level's update by how long it has accumulated.
        params = list(L.net.parameters())
        pending = [p.grad for p in params]
        for p in params:
            p.grad = None
        loss.backward()
        fresh = [p for p in params if p.grad is not None]
        if fresh:
            torch.nn.utils.clip_grad_norm_(fresh, a.grad_clip)
        for p, g in zip(params, pending):
            if g is not None:
                p.grad = g if p.grad is None else p.grad.add_(g)
        L.buffer.add(xr, yr)
        L.t += 1

        children = []
        if L.t % a.check_every == 0:
            children = self.check(L)
        self.commit(L)
        for ch in children:
            self.commit(ch)

    def commit(self, L):
        # --child-burst N: for its first N batches a new child steps EVERY level
        # every batch, so inherited slow features can adapt to its classes before
        # the slow period takes over. 0 = off (default).
        burst = (self.a.child_burst > 0 and L.parent is not None
                 and (L.t - L.born_t) <= self.a.child_burst)
        for lvl, opt in zip(L.net.levels, L.opts):
            period = 1 if burst else lvl.period
            if L.t % period:
                continue
            ps = [p for p in lvl.parameters() if p.grad is not None]
            if not ps:
                continue
            if period > 1:
                for p in ps:
                    p.grad.div_(period)          # mean of the clipped fresh gradients
            opt.step()
            opt.zero_grad(set_to_none=True)

    def refresh_fast(self, L):
        """Refresh ownership and referrals on ALL stored images.

        --refresh-scope head (default): only the 100-way head and the 'mine' head
            move; features are computed without gradient. An old learner's
            representation is never touched by other learners' classes, so this
            cannot cause forgetting inside the learner.
        --refresh-scope fast: the whole fast level moves (the original 2c
            behaviour; it trained 75% of each learner's weights on 20 replay
            images per class plus foreign images, and was a forgetting leak).
        Pending slow-level updates are stashed and restored untouched."""
        a = self.a
        if a.refresh_steps <= 0 or not L.classes or a.refresh_scope == "none":
            return
        heads_only = a.refresh_scope == "head"
        fast = min(lvl.period for lvl in L.net.levels)
        fast_ids = [i for i, lvl in enumerate(L.net.levels) if lvl.period == fast]
        params = list(L.net.parameters())
        stash = [None if p.grad is None else p.grad.detach().clone() for p in params]
        for p in params:
            p.grad = None
        if heads_only:
            L.net.eval()          # frozen features: no BN statistics drift either
        else:
            L.net.train()
        for _ in range(a.refresh_steps):
            ox, oy = L.buffer._gather(sorted(L.classes), "replay")
            if ox is None:
                break
            idx = torch.randint(0, ox.size(0), (min(32, ox.size(0)),), device=ox.device)
            fx, fy = self.foreign(L, 32)
            loss = self.ownership_loss(L, self.prep(ox[idx], True), oy[idx], fx, fy,
                                       heads_only=heads_only)
            loss.backward()
            for i in fast_ids:
                ps = [p for p in L.net.levels[i].parameters() if p.grad is not None]
                if ps:
                    torch.nn.utils.clip_grad_norm_(ps, a.grad_clip)
                    L.opts[i].step()
            for o in L.opts:
                o.zero_grad(set_to_none=True)
        for p, g in zip(params, stash):
            p.grad = g
        L.net.train()

    # ------------------------------------------------------------------ trigger
    @torch.no_grad()
    def anchor_acc(self, L):
        ax, ay = L.buffer.check_set(sorted(L.established))
        if ax is None:
            return None
        L.net.eval()
        m = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=self.dev)
        m[sorted(L.classes)] = True
        lg = L.net(self.prep(ax, False)).masked_fill(~m, float("-inf"))
        L.net.train()
        return float((lg.argmax(1) == ay).float().mean())

    def check(self, L):
        a = self.a
        new = L.new_classes
        if not new:
            L.class_hist.clear()
            L.hist.clear()
            return []

        to_split, vecs = [], {}
        if L.established and a.trigger == "predictive":
            A = self.anchor_grads(L, L.established)
            if A is not None:
                self.conflicts.append(self.weighted_conflict(L, A, self.pending(L)))
                for c in sorted(new):
                    xc, yc = self.class_images(L, c, 16)
                    if xc is None:
                        L.class_hist.setdefault(c, deque(maxlen=64)).append(0.0)
                        continue
                    G = self.grads(L, self.prep(xc, False), yc)
                    v = self.weighted_conflict(L, A, G)
                    h = L.class_hist.setdefault(c, deque(maxlen=64))
                    h.append(v)
                    vecs[c] = self.flat_vec(L, G)
                    if len(h) >= a.smooth and float(np.mean(list(h)[-a.smooth:])) > a.theta:
                        to_split.append(c)
            else:
                for c in new:
                    L.class_hist.setdefault(c, deque(maxlen=64)).append(0.0)

        elif L.established and a.trigger == "reactive":
            acc = self.anchor_acc(L)
            drop = (L.base_acc - acc) if (acc is not None and L.base_acc is not None) else 0.0
            L.hist.append(drop)
            for c in new:
                L.class_hist.setdefault(c, deque(maxlen=64)).append(0.0)
            if len(L.hist) >= a.smooth and float(np.mean(list(L.hist)[-a.smooth:])) > a.delta:
                to_split = sorted(new)
        else:
            for c in new:
                L.class_hist.setdefault(c, deque(maxlen=64)).append(0.0)

        slots = self.slots()
        if to_split and slots > 0:
            if a.trigger == "predictive" and all(c in vecs for c in to_split):
                order = sorted(to_split, key=lambda c: -np.mean(list(L.class_hist[c])[-a.smooth:]))
                groups = self.cap_groups(self.group_by_agreement(vecs, order), slots)
            else:
                groups = [to_split]
            return self.split(L, groups)

        for c in sorted(new):
            h = L.class_hist.get(c)
            if h is not None and len(h) >= a.establish_checks:
                L.established.add(c)
                L.class_hist.pop(c, None)
                self.event_log.append(dict(task=self.task, event="absorb", uid=L.uid, cls=c))
        if not L.new_classes:
            L.hist.clear()
            L.base_acc = None
        return []

    def split(self, L, groups):
        """Each group becomes a child that takes the pending update. The parent
        discards it, so its slow levels are untouched, and keeps everything else."""
        pend = [p.grad for p in L.net.parameters()]
        children = []
        for grp in groups:
            child = self._add(copy.deepcopy(L.net), parent=L.uid, src=L)
            for pc, g in zip(child.net.parameters(), pend):
                pc.grad = None if g is None else g.detach().clone()
            child.classes = set(grp)
            child.established = set(grp)
            child.buffer = L.buffer.restrict(grp)
            for c in grp:
                self.owner[c] = child.uid
            children.append(child)
        moved = set().union(*[set(g) for g in groups])
        L.classes -= moved
        L.buffer = L.buffer.restrict(L.classes)
        for c in moved:
            L.class_hist.pop(c, None)
        for p in L.net.parameters():
            p.grad = None
        L.hist.clear()
        L.base_acc = None
        for ch in children:
            self.split_log.append(dict(task=self.task, kind="split", parent=L.uid,
                                       child=ch.uid, classes=sorted(ch.classes), step=self.gstep))
            print(f"    split: learner {L.uid} -> child {ch.uid} takes {sorted(ch.classes)}",
                  flush=True)
        self.refresh_fast(L)
        for ch in children:
            self.refresh_fast(ch)
        return children

    # ------------------------------------------------------------------ evaluation
    #
    # Four label-free routing signals, each a distribution over learners:
    #   own        each learner's "mine" probability
    #   consensus  referrals: probability mass the OTHER learners give to its classes
    #   shared     one 100-class model in the ROOT's slow-level space (read-only), fitted
    #              fresh at every evaluation from all stored images, so it is exact
    #   lr         likelihood ratio in each learner's own space: distance to its own class
    #              means vs distance to the nearest stored images of OTHER learners,
    #              standardised on the learner's own held-out check images
    # 'soft' multiplies the enabled signals (product of experts) and then
    # P(class) = sum_k P(k | x) * P(class | x, k).  None of this touches training.

    def _embed(self, net, x, block, chunk=512):
        net.eval()
        out = []
        for i in range(0, x.size(0), chunk):
            h = net.backbone.forward_blocks(self.prep(x[i:i + chunk], False))[block]
            out.append(h.mean(dim=(2, 3)) if h.dim() == 4 else h)
        return torch.cat(out)

    def _memory(self, Ls):
        """All stored images, their labels, and which are held-out check images."""
        xs, ys, chk = [], [], []
        for L in Ls:
            for part in ("check", "replay"):
                x, y = L.buffer._gather(sorted(L.buffer.x), part)
                if x is not None:
                    xs.append(x)
                    ys.append(y)
                    chk.append(torch.full((y.size(0),), part == "check",
                                          dtype=torch.bool, device=self.dev))
        return torch.cat(xs), torch.cat(ys), torch.cat(chk)

    def _fit_shared(self, X, Y):
        ref = self.learners[0].net                                   # root, read-only
        H = self._embed(ref, X, self.a.shared_block)
        classes = sorted(set(Y.tolist()))
        idx_of = torch.full((NUM_CLASSES,), -1, dtype=torch.long, device=self.dev)
        idx_of[classes] = torch.arange(len(classes), device=self.dev)
        Yi = idx_of[Y]
        mu = torch.stack([H[Yi == i].mean(0) for i in range(len(classes))])
        Z = H - mu[Yi]
        d = H.size(1)
        cov = Z.T @ Z / max(H.size(0) - len(classes), 1)
        a = self.a.shrink
        cov = (1 - a) * cov + a * (torch.trace(cov) / d) * torch.eye(d, device=self.dev)
        return dict(ref=ref, mu=mu, prec=torch.linalg.inv(cov),
                    classes=torch.tensor(classes, device=self.dev))

    def _shared_posterior(self, S, x, owner_mat):
        h = self._embed(S["ref"], x, self.a.shared_block)
        diff = h[:, None, :] - S["mu"][None]
        maha = torch.einsum("bcd,de,bce->bc", diff, S["prec"], diff)
        p_cls = (-0.5 * maha).softmax(1)                              # B x C
        return p_cls @ owner_mat[S["classes"]]                        # B x K

    @staticmethod
    def _lr_score(f, mus, others, knn):
        d_in = (1.0 - f @ mus.T).min(1).values
        if others is None or others.size(0) == 0:
            return -d_in                     # 'own' evidence: closeness to my own classes only
        D = 1.0 - f @ others.T
        d_out = D.topk(min(knn, D.size(1)), dim=1, largest=False).values.mean(1)
        return d_out - d_in

    def _fit_lr(self, Ls, X, Y, CHK):
        stats = []
        for L in Ls:
            Fk = F.normalize(self._embed(L.net, X, 4), dim=1)          # pooled layer4
            own = torch.zeros_like(Y, dtype=torch.bool)
            for c in L.classes:
                own |= (Y == c)
            cls = [c for c in sorted(L.classes) if bool((Y == c).any())]
            if cls:
                mus = F.normalize(torch.stack([Fk[Y == c].mean(0) for c in cls]), dim=1)
            else:
                mus = torch.zeros(1, Fk.size(1), device=self.dev)
            others = Fk[~own] if (bool((~own).any()) and self.a.evidence == "ratio") else None
            ref = Fk[own & CHK]
            if ref.size(0) >= 2:
                s = self._lr_score(ref, mus, others, self.a.knn)
                m, sd = float(s.mean()), float(s.std().clamp_min(1e-3))
            else:
                m, sd = 0.0, 1.0
            stats.append(dict(mus=mus, others=others, m=m, sd=sd))
        return stats

    @torch.no_grad()
    def evaluate(self, test_idx, groups, chunk=250):
        a = self.a
        Ls = [L for L in self.learners if L.classes]
        K = len(Ls)
        owner_mat = torch.zeros(NUM_CLASSES, K, device=self.dev)
        for k, L in enumerate(Ls):
            owner_mat[sorted(L.classes), k] = 1.0
        masks = owner_mat.T.bool()
        cls_owner = torch.full((NUM_CLASSES,), -1, dtype=torch.long, device=self.dev)
        for k, L in enumerate(Ls):
            cls_owner[sorted(L.classes)] = k

        sig = set(a.signals.split(","))
        X, Y, CHK = self._memory(Ls)
        S_sh = self._fit_shared(X, Y) if ("shared" in sig and K > 1) else None
        LR = self._fit_lr(Ls, X, Y, CHK) if ("lr" in sig and K > 1) else None

        routers = ROUTERS
        acc = {r: {} for r in routers}
        r_ok = {r: 0 for r in routers}
        r_tot = 0
        eps = 1e-6
        for L in Ls:
            L.net.eval()
        for g in groups:
            idx = test_idx[g]
            correct = {r: 0 for r in routers}
            gmask = torch.zeros(NUM_CLASSES, dtype=torch.bool, device=self.dev)
            gmask[torch.unique(self.data.yte[idx])] = True      # the task's own classes
            for i in range(0, len(idx), chunk):
                x_raw = self.data.xte[idx[i:i + chunk]]
                x = self.prep(x_raw, False)
                y = self.data.yte[idx[i:i + chunk]]
                B = y.size(0)
                own, LG, FE = [], [], []
                for L in Ls:
                    f, lg, o = forward_all(L.net, x)
                    own.append(torch.sigmoid(o))
                    LG.append(lg)
                    FE.append(F.normalize(f, dim=1))
                own = torch.stack(own, 1)
                LG = torch.stack(LG, 0)
                within = torch.nan_to_num(
                    LG.masked_fill(~masks[:, None, :], float("-inf")).softmax(-1), nan=0.0)
                uniform = torch.full((B, K), 1.0 / K, device=self.dev)

                P = {"own": own / own.sum(1, keepdim=True).clamp_min(eps)}
                if K > 1:
                    ref = torch.einsum("kbc,cj->bkj", LG.softmax(-1), owner_mat)
                    cons = (ref.sum(1) - torch.diagonal(ref, dim1=1, dim2=2)) / (K - 1)
                    P["consensus"] = cons / cons.sum(1, keepdim=True).clamp_min(eps)
                    P["soft_oc"] = ((own + cons) / a.route_temp).softmax(1)
                else:
                    P["consensus"] = uniform
                    P["soft_oc"] = uniform
                P["shared"] = (self._shared_posterior(S_sh, x_raw, owner_mat)
                               if S_sh is not None else uniform)
                if LR is not None:
                    z = torch.stack([(self._lr_score(FE[k], LR[k]["mus"], LR[k]["others"], a.knn)
                                      - LR[k]["m"]) / LR[k]["sd"] for k in range(K)], 1)
                    P["lr"] = (z / a.lr_temp).softmax(1)
                else:
                    P["lr"] = uniform

                logp = sum(torch.log(P[s] + eps) for s in ("own", "consensus", "shared", "lr")
                           if s in sig)
                P["soft"] = logp.softmax(1)

                ar = torch.arange(B, device=self.dev)
                true_k = cls_owner[y]
                # S[b, c] = P(class c | x, owner of c): every class scored by its owner
                own_idx = cls_owner.clamp_min(0)[None, None, :].expand(1, B, NUM_CLASSES)
                S = torch.gather(within, 0, own_idx).squeeze(0)
                S = S.masked_fill((cls_owner < 0)[None, :], float("-inf"))
                for r in routers:
                    if r == "oracle":
                        k_sel = true_k
                        pred = within[k_sel, ar].argmax(-1)
                    elif r == "taskil":
                        k_sel = true_k
                        pred = S.masked_fill(~gmask[None, :], float("-inf")).argmax(1)
                    elif r in ("soft", "soft_oc"):
                        k_sel = P[r].argmax(1)
                        pred = torch.einsum("bk,kbc->bc", P[r], within).argmax(1)
                    else:
                        k_sel = P[r].argmax(1)
                        pred = within[k_sel, ar].argmax(-1)
                    correct[r] += int((pred == y).sum())
                    r_ok[r] += int((k_sel == true_k).sum())
                r_tot += B
            for r in routers:
                acc[r][g] = 100.0 * correct[r] / len(idx)
        for L in self.learners:
            L.net.train()
        routing = {r: 100.0 * v / max(r_tot, 1) for r, v in r_ok.items()}
        return acc, routing

    def footprint(self):
        return (sum(count_params(L.net) for L in self.learners),
                sum(L.buffer.nbytes() for L in self.learners))


# =============================================================================

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
    # splitting
    ap.add_argument("--check-every", type=int, default=16)
    ap.add_argument("--theta", type=float, default=0.1)
    ap.add_argument("--delta", type=float, default=0.10)
    ap.add_argument("--smooth", type=int, default=2)
    ap.add_argument("--establish-checks", type=int, default=6)
    ap.add_argument("--max-modules", type=int, default=20, help="0 = no cap")
    ap.add_argument("--child-burst", type=int, default=0,
                    help="batches after birth during which a child steps every level every batch")
    # memory
    ap.add_argument("--anchors-per-class", type=int, default=25)
    ap.add_argument("--n-check", type=int, default=5)
    ap.add_argument("--replay", type=int, default=16)
    # ownership / routing
    ap.add_argument("--neg", type=int, default=16, help="foreign negatives per step")
    ap.add_argument("--dmg", type=int, default=8, help="self-damaged negatives per step")
    ap.add_argument("--lam-own", type=float, default=1.0)
    ap.add_argument("--lam-ref", type=float, default=0.5)
    ap.add_argument("--refresh-every", type=int, default=200, help="global steps")
    ap.add_argument("--refresh-steps", type=int, default=5)
    ap.add_argument("--refresh-scope", choices=["head", "fast", "none"], default="head",
                    help="head: only the two heads move during a refresh (features frozen); "
                         "fast: whole fast level (old behaviour, a forgetting leak)")
    ap.add_argument("--route-temp", type=float, default=0.1)
    ap.add_argument("--signals", default="own,consensus,shared,lr",
                    help="routing signals multiplied in the 'soft' router")
    ap.add_argument("--shared-block", type=int, default=3,
                    help="root block used as the shared space (3 = layer3, end of the medium level)")
    ap.add_argument("--shrink", type=float, default=0.2, help="covariance shrinkage, shared space")
    ap.add_argument("--knn", type=int, default=5)
    ap.add_argument("--lr-temp", type=float, default=1.0)
    ap.add_argument("--evidence", choices=["ratio", "own"], default="ratio",
                    help="lr signal: ratio = own-class distance vs k-NN to others' anchors; "
                         "own = own-class distance only (guilty until proven innocent: a learner "
                         "needs nothing from other learners, at training or at test)")
    # validation / io
    ap.add_argument("--val", action="store_true")
    ap.add_argument("--val-frac", type=float, default=0.1)
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
        raise SystemExit(f"stage2c supports periodic-stepping arms only; '{args.arm}' is {arm['rule']}")

    tag = f"_th{args.theta}" if args.trigger == "predictive" else (
        f"_d{args.delta}" if args.trigger == "reactive" else "")
    name = f"{args.trigger}_{args.arm}_{args.stream}{tag}"
    out = Path(args.out) if args.out else ROOT / "results" / "stage2c" / name / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    data = Data(device, args.data_root, args.fake_data, args.seed)
    tasks, test_idx = build_stream(data, args.stream, args.n_tasks, args.classes_per_task, args.seed)
    if args.val:
        test_idx = carve_validation(data, tasks, args.val_frac, args.seed)
        print(f"VALIDATION MODE: evaluating on {args.val_frac:.0%} of train; test set untouched",
              flush=True)
    system = System(args, arm, data, device)
    print(f"{name} seed={args.seed} device={device} lr={system.lr} "
          f"levels={[l.period for l in system.learners[0].net.levels]}", flush=True)

    routers = ROUTERS
    rows, hist, seen = [], {r: {} for r in routers}, []
    for t, task in enumerate(tasks):
        tt = time.time()
        system.task = t
        n0 = len(system.conflicts)
        idx = task["train_idx"]
        for _ in range(args.epochs):
            perm = idx[torch.randperm(len(idx), device=device)]
            for b in range(0, len(perm), args.batch_size):
                bi = perm[b:b + args.batch_size]
                system.train_batch(data.xtr[bi], data.ytr[bi])

        if task["group"] not in seen:
            seen.append(task["group"])
        acc, routing = system.evaluate(test_idx, seen)
        for r in routers:
            for g, v in acc[r].items():
                hist[r].setdefault(g, []).append(v)
        params, mem = system.footprint()
        tc = system.conflicts[n0:]
        row = dict(task=t, group=task["group"], n_modules=len(system.learners),
                   splits=len(system.split_log),
                   mean_conflict=float(np.mean(tc)) if tc else None,
                   params=params, anchor_kb=mem / 1024, secs=time.time() - tt)
        for r in routers:
            row[f"acc_{r}"] = float(np.mean(list(acc[r].values())))
            row[f"route_{r}"] = routing[r]
        row["avg_acc"] = row["acc_soft"]
        rows.append(row)
        print(f"  task {t:>2}  learners {row['n_modules']:>2}  acc soft {row['acc_soft']:6.2f} "
              f"oracle {row['acc_oracle']:6.2f}  task-IL {row['acc_taskil']:6.2f}  "
              f"route soft {row['route_soft']:6.2f} "
              f"own {row['route_own']:5.1f} cons {row['route_consensus']:5.1f} "
              f"shared {row['route_shared']:5.1f} lr {row['route_lr']:5.1f}  "
              f"{row['secs']:.0f}s", flush=True)

    def forgetting(h):
        f = [max(v) - v[-1] for v in h.values() if len(v) > 1]
        return float(np.mean(f)) if f else 0.0

    def new_task_acc(h):
        """Accuracy on each task right after it was learned (plasticity)."""
        return float(np.mean([v[0] for v in h.values()]))

    final = rows[-1]
    cap_task = (next((r["task"] for r in rows if r["n_modules"] >= args.max_modules), None)
                if args.max_modules > 0 else None)
    cv = np.array(system.conflicts) if system.conflicts else np.array([np.nan])
    summary = dict(
        run=name, stage="2c", trigger=args.trigger, arm=args.arm, stream=args.stream,
        seed=args.seed, theta=args.theta, delta=args.delta, epochs=args.epochs, lr=system.lr,
        final_avg_acc=final["acc_soft"],
        avg_incremental_acc=float(np.mean([r["acc_soft"] for r in rows])),
        forgetting=forgetting(hist["soft"]),
        new_task_acc=new_task_acc(hist["soft"]),
        cap_hit_at_task=cap_task,          # None if the learner cap was never reached
        n_modules=final["n_modules"], n_tasks=len(tasks),
        n_splits=sum(s["kind"] == "split" for s in system.split_log),
        n_spawn_on_arrival=sum(s["kind"] == "spawn-on-arrival" for s in system.split_log),
        n_absorbs=sum(e["event"] == "absorb" for e in system.event_log),
        params=final["params"], anchor_kb=final["anchor_kb"],
        conflict_pct={str(p): float(np.nanpercentile(cv, p)) for p in (10, 25, 50, 75, 90)},
        groups=[sorted(L.classes) for L in system.learners],
        split_log=system.split_log, per_task=rows,
        per_group_history=hist,            # accuracy of every task after every task, per router
        refresh_scope=args.refresh_scope, nf=args.nf,
        minutes=(time.time() - t0) / 60, fake_data=args.fake_data, val=args.val,
    )
    for r in routers:
        summary[f"final_acc_{r}"] = final[f"acc_{r}"]
        summary[f"routing_{r}"] = final[f"route_{r}"]
        summary[f"forgetting_{r}"] = forgetting(hist[r])
        summary[f"new_task_acc_{r}"] = new_task_acc(hist[r])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nfinal acc soft {summary['final_avg_acc']:.2f}  oracle {summary['final_acc_oracle']:.2f}  "
          f"task-IL {summary['final_acc_taskil']:.2f} (forgetting {summary['forgetting_taskil']:.2f}, "
          f"new-task {summary['new_task_acc_taskil']:.2f})  "
          f"routing soft {summary['routing_soft']:.2f}  learners {summary['n_modules']}  "
          f"splits {summary['n_splits']} + {summary['n_spawn_on_arrival']} on arrival  "
          f"({summary['minutes']:.1f} min) -> {out / 'summary.json'}")


if __name__ == "__main__":
    main()
