import copy
from typing import List

import torch
import torch.nn.functional as F
from torch.optim import SGD

from src.nested.learner import NestedLearner


class UpdateRule:
    """Base: one SGD over everything. Subclasses change only the update mechanics."""

    def __init__(self, learner: NestedLearner, lr, momentum=0.0, weight_decay=0.0, **_):
        self.learner = learner
        self.lr, self.momentum, self.wd = lr, momentum, weight_decay
        self.optimizers: List[torch.optim.Optimizer] = [self._sgd(learner.parameters())]

    def _sgd(self, params, lr=None):
        return SGD(params, lr=lr if lr is not None else self.lr, momentum=self.momentum, weight_decay=self.wd)

    @property
    def primary_optimizer(self):
        return self.optimizers[0]

    def reset(self):
        """New experience: clear optimizer state (momentum) and pending grads."""
        for o in self.optimizers:
            o.state.clear()
            o.zero_grad(set_to_none=True)

    def zero_grad(self):
        for o in self.optimizers:
            o.zero_grad(set_to_none=True)

    def compute_loss(self, strategy):
        return strategy.criterion()

    def backward(self, loss, strategy):
        loss.backward()

    def step(self):
        for o in self.optimizers:
            o.step()

    def eval_model(self):
        return self.learner


# -------------------------------------------------------------------------- A
class PeriodicStepping(UpdateRule):
    """Level with period k accumulates gradients over k steps and applies one step.
    accumulate='mean' -> larger batch at same LR; 'sum' -> same batch at k x LR."""

    def __init__(self, learner, lr, accumulate="mean", **kw):
        super().__init__(learner, lr, **kw)
        self.accumulate = accumulate
        self.optimizers = [self._sgd(l.parameters()) for l in learner.levels]
        self.t = 0

    def reset(self):
        super().reset()
        self.t = 0

    def zero_grad(self):
        pass  # grads persist across steps by design; zeroed only when a level steps

    def step(self):
        self.t += 1
        for lvl, opt in zip(self.learner.levels, self.optimizers):
            if self.t % lvl.period != 0:
                continue
            if self.accumulate == "mean" and lvl.period > 1:
                for p in lvl.parameters():
                    if p.grad is not None:
                        p.grad.div_(lvl.period)
            opt.step()
            opt.zero_grad(set_to_none=True)


# -------------------------------------------------------------------------- B
class PerLevelLR(UpdateRule):
    """Level i trains at lr * lr_ratio_i. One SGD with param groups.
    With depth=6 (one block per level) and geometric ratios this *is* LLRD, i.e. the control."""

    def __init__(self, learner, lr, **kw):
        super().__init__(learner, lr, **kw)
        groups = [{"params": list(l.parameters()), "lr": lr * l.lr_ratio} for l in learner.levels]
        self.optimizers = [SGD(groups, lr=lr, momentum=self.momentum, weight_decay=self.wd)]


# -------------------------------------------------------------------------- C
class EMACoupling(UpdateRule):
    """Live weights train normally. Each level has a shadow that is an EMA of the live
    level at rate tau_i (CLS-ER generalised to per-level rates). The shadow composite is
    the deployed model; an optional consistency term pulls live logits toward shadow logits."""

    def __init__(self, learner, lr, consistency_weight=0.0, **kw):
        super().__init__(learner, lr, **kw)
        self.w = consistency_weight
        self.shadow = copy.deepcopy(learner)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    def compute_loss(self, strategy):
        loss = strategy.criterion()
        if self.w > 0:
            self.shadow.eval()
            with torch.no_grad():
                s_logits = self.shadow(strategy.mb_x)
            loss = loss + self.w * F.mse_loss(strategy.mb_output, s_logits)
        return loss

    def step(self):
        super().step()
        self._ema_update()

    @torch.no_grad()
    def _ema_update(self):
        for lvl, s_lvl in zip(self.learner.levels, self.shadow.levels):
            tau = lvl.ema_tau
            for p, sp in zip(lvl.parameters(), s_lvl.parameters()):
                sp.mul_(tau).add_(p, alpha=1.0 - tau)
        for b, sb in zip(self.learner.buffers(), self.shadow.buffers()):
            sb.copy_(b)  # BN running stats are not averaged

    def eval_model(self):
        return self.shadow


# -------------------------------------------------------------------------- D
class NestedObjectives(PeriodicStepping):
    """Level i+1 (slower) optimises over the gradient stream of level i (faster):
        g_{i+1} = d/dθ_{i+1} [ L + λ ||∇_{θ_i} L||^1^ ]
    i.e. the slow level also minimises the *surprise* of the level below it (second-order,
    via double backward). The fastest level sees only ∇L. Stepping is periodic as in A.
    This is one concrete reading of 'nested objectives'; MoNet's DMSM is the other."""

    def __init__(self, learner, lr, surprise_weight=0.1, accumulate="mean", **kw):
        super().__init__(learner, lr, accumulate=accumulate, **kw)
        self.lam = surprise_weight

    def backward(self, loss, strategy):
        levels = self.learner.levels_fast_to_slow()
        per_level = [list(l.parameters()) for l in levels]
        flat = [p for ps in per_level for p in ps]
        g_all = torch.autograd.grad(loss, flat, create_graph=True, allow_unused=True)

        grads, i = [], 0
        for ps in per_level:
            grads.append(list(g_all[i : i + len(ps)]))
            i += len(ps)

        for li, (lvl, ps) in enumerate(zip(levels, per_level)):
            extra = [None] * len(ps)
            if li > 0:
                faster = [g for g in grads[li - 1] if g is not None]
                surprise = sum(g.pow(2).sum() for g in faster)
                extra = torch.autograd.grad(surprise, ps, retain_graph=True, allow_unused=True)
            for p, g_own, g_ex in zip(ps, grads[li], extra):
                tot = None if g_own is None else g_own.detach()
                if g_ex is not None:
                    tot = self.lam * g_ex.detach() if tot is None else tot + self.lam * g_ex.detach()
                if tot is None:
                    continue
                if p.grad is None:
                    p.grad = tot.clone()
                else:
                    p.grad.add_(tot)
        del g_all, grads


RULES = {
    "periodic": PeriodicStepping,
    "per_level_lr": PerLevelLR,
    "ema": EMACoupling,
    "nested_objectives": NestedObjectives,
}


def build_rule(name, learner, lr, momentum, weight_decay, **kwargs):
    return RULES[name](learner, lr=lr, momentum=momentum, weight_decay=weight_decay, **kwargs)