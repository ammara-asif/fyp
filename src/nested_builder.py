"""Builds the nested arms for stage1_runner: model + update rule + strategy.

Glue between my runner and the team's code in src/nested/ and src/strategies/:
  periodic           variant A: level i accumulates gradients and steps every k_i minibatches
  per_level_lr       variant B (and Flat = B with lr_ratios all 1)
  ema                variant C: live weights train like Flat; a shadow copy keeps, per level, an
                     EMA of the live weights at rate tau_i, and the shadow is the deployed model
  nested_objectives  variant D, "one reading": periodic stepping (as A) + a penalty on the
                     faster level's squared gradient norm, via double backward

Config (see make_cfg in stage1_runner):
  rule        one of the names above
  nesting     {depth, slow_side, periods (A, D) | lr_ratios (B) | ema_taus (C)}; rate vectors
              listed FASTEST FIRST
  rule_kwargs A {accumulate}, C {consistency_weight}, D {surprise_weight, accumulate}
slow_side "input" = levels nearest the input are slowest (the main arm, plan R1). The team's
NestedLearner calls this ordering="slow_deep"; the mapping below is the only place it appears.
"""
import torch
from avalanche.core import SupervisedPlugin

from src.nested.learner import NestedLearner
from src.nested.update_rules import build_rule
from src.strategies.nested_strategy import NestedStrategy

SLOW_SIDE_TO_ORDERING = {"input": "slow_deep", "output": "fast_deep"}


def install_fresh_grad_clip(rule, max_norm):
    """Clip each minibatch's NEW gradient to global norm `max_norm`, before it is added to
    gradients the rule is still accumulating.

    Why: D (like A) keeps summing gradients into .grad for 4/16 steps before a slow level
    steps. Clipping the whole .grad after each backward (what TrainGuard does for Flat/LLRD)
    would then also count the slow levels' growing buffers and shrink the fast level's step.
    Clipping only the fresh gradient is the same rule Flat and B already follow (for them
    nothing is held over, so this is identical to the TrainGuard clip).

    Also raises on a non-finite gradient norm: D's penalty could blow up while the data loss
    stays finite, which the loss check in TrainGuard would miss. max_norm=None: no clip, check only.
    """
    inner = rule.backward
    params = list(rule.learner.parameters())

    def backward(loss, strategy):
        held = [p.grad for p in params]
        for p in params:
            p.grad = None
        inner(loss, strategy)  # writes only this minibatch's gradient
        fresh = [p for p in params if p.grad is not None]
        norm = torch.nn.utils.clip_grad_norm_(
            fresh, max_norm if max_norm is not None else float("inf"))
        if not torch.isfinite(norm):
            raise FloatingPointError(f"non-finite gradient norm ({norm.item()})")
        for p, h in zip(params, held):
            if h is None:
                continue
            if p.grad is None:
                p.grad = h
            else:
                p.grad = h.add_(p.grad)  # keep the accumulator tensor, add the clipped fresh grad

    rule.backward = backward  # instance attribute: wraps this rule only, no change to the class


def keep_shadow_in_eval_mode(rule):
    """C only: the shadow (deployed) network is never trained by backprop, so keep it in eval
    mode for good. Why: it is deep-copied while the live network is in train mode, and Avalanche
    switches modes on whatever `strategy.model` is at the time. A shadow in train mode would use
    batch statistics at evaluation and overwrite its BatchNorm running stats, which are meant to
    be copied from the live network after every step."""
    shadow = getattr(rule, "shadow", None)
    if shadow is None:
        return
    shadow.eval()
    # Pin every submodule, not only the top: Avalanche saves and restores the train flag of each
    # submodule around evaluation. Instance overrides make .train()/.eval() no-ops on the shadow.
    for m in shadow.modules():
        m.train = (lambda mod: (lambda mode=True: mod))(m)


class LevelMonitor(SupervisedPlugin):
    """Per-task, per-level record of what the update rule actually did.

    updates/<lvl>           optimizer steps the level took during the task (B: every minibatch;
                            D: about 1/period of them; the remainder accumulated at the end of a
                            task is discarded by the rule's reset at the next task)
    displacement/<lvl>      ||θ_end - θ_start|| of the level's parameters over the task
    rel_displacement/<lvl>  the same divided by ||θ_start||
    D only (means over the task's minibatches): data_grad_norm, surprise, penalty_grad_norm,
    and penalty_ratio = mean penalty_grad_norm / mean data_grad_norm of the same level.
    C only: deployed_displacement/<lvl> and deployed_rel_displacement/<lvl>, the same measures
    on the shadow (the network that is evaluated and checkpointed); the plain keys are the live one.
    BatchNorm running statistics are buffers, not parameters, so they are not in the displacement.
    """

    def __init__(self, learner, rule):
        super().__init__()
        self.levels, self.rule = learner.levels, rule
        deployed = rule.eval_model()
        self.deployed_levels = deployed.levels if deployed is not learner else None
        self.steps = 0
        self.rows = []

    @staticmethod
    def _flat(lvl):
        return torch.cat([p.detach().flatten() for p in lvl.parameters()]).clone()

    def _updates(self):
        if hasattr(self.rule, "updates"):
            return list(self.rule.updates)
        return [self.steps] * len(self.levels)  # one optimizer: every step updates every level

    def before_training_exp(self, strategy, **kw):
        self._start = [self._flat(l) for l in self.levels]
        if self.deployed_levels is not None:
            self._start_dep = [self._flat(l) for l in self.deployed_levels]
        self._upd0 = self._updates()
        if hasattr(self.rule, "stats"):
            self.rule.stats, self.rule.stats_n = {}, 0

    def after_update(self, strategy, **kw):
        self.steps += 1

    def after_training_exp(self, strategy, **kw):
        row = {}
        for lvl, s, u0, u1 in zip(self.levels, self._start, self._upd0, self._updates()):
            d = (self._flat(lvl) - s).norm().item()
            row[f"updates/{lvl.name}"] = u1 - u0
            row[f"displacement/{lvl.name}"] = d
            row[f"rel_displacement/{lvl.name}"] = d / (s.norm().item() + 1e-12)
        if self.deployed_levels is not None:
            for lvl, s in zip(self.deployed_levels, self._start_dep):
                d = (self._flat(lvl) - s).norm().item()
                row[f"deployed_displacement/{lvl.name}"] = d
                row[f"deployed_rel_displacement/{lvl.name}"] = d / (s.norm().item() + 1e-12)
        n = getattr(self.rule, "stats_n", 0)
        if n:
            means = {k: v.item() / n for k, v in self.rule.stats.items()}
            row.update(means)
            for lvl in self.levels:
                pen, dat = means.get(f"penalty_grad_norm/{lvl.name}"), means.get(f"data_grad_norm/{lvl.name}")
                if pen is not None and dat:
                    row[f"penalty_ratio/{lvl.name}"] = pen / dat
        self.rows.append(row)


def build(cfg, backbone, device, strategy_kwargs, plugins):
    """Returns (learner, rule, strategy, monitor, level_info).

    strategy_kwargs: the runner's common Avalanche kwargs (minibatch sizes, epochs, device,
    evaluator); model/optimizer/criterion are owned here. The plugins list gets the monitor.
    """
    n = cfg["nesting"]
    learner = NestedLearner(backbone, depth=n["depth"], periods=n.get("periods"),
                            lr_ratios=n.get("lr_ratios"), ema_taus=n.get("ema_taus"),
                            ordering=SLOW_SIDE_TO_ORDERING[n["slow_side"]]).to(device)
    # Built after .to(device), so C's shadow (a deep copy of the learner) is on the device too.
    rule = build_rule(cfg["rule"], learner, lr=cfg["lr"], momentum=cfg["momentum"],
                      weight_decay=cfg["weight_decay"], **cfg["rule_kwargs"])
    install_fresh_grad_clip(rule, cfg["grad_clip_norm"])
    keep_shadow_in_eval_mode(rule)
    monitor = LevelMonitor(learner, rule)
    kw = {k: v for k, v in strategy_kwargs.items() if k not in ("model", "optimizer", "criterion")}
    strategy = NestedStrategy(learner, rule, plugins=list(plugins) + [monitor],
                              **{k: v for k, v in kw.items() if k != "plugins"})
    # Effective lr, stepping period and EMA rate of each level, input side first (result file).
    level_info = []
    for lvl in learner.levels:
        lr = cfg["lr"] * lvl.lr_ratio if cfg["rule"] == "per_level_lr" else cfg["lr"]
        period = lvl.period if cfg["rule"] in ("periodic", "nested_objectives") else 1
        info = {"level": lvl.idx, "blocks": lvl.name, "lr": lr, "period": period}
        if cfg["rule"] == "ema":
            info["ema_tau"] = lvl.ema_tau
        level_info.append(info)
    return learner, rule, strategy, monitor, level_info


def checkpoint_module(rule):
    """The network whose weights are saved after each task: the deployed model's backbone, so
    the state_dict loads straight into MODEL_BUILDERS['reduced_resnet18_blocks']."""
    return rule.eval_model().backbone
