"""Builds the nested arms for stage1_runner: model + update rule + strategy.

Glue between my runner and the team's code in src/nested/ and src/strategies/. Only
two rules are wired here (A and C are my teammate's):
  per_level_lr       variant B (and Flat = B with lr_ratios all 1)
  nested_objectives  variant D, "one reading": periodic stepping (as A) + a penalty on the
                     faster level's squared gradient norm, via double backward

Config (see make_cfg in stage1_runner):
  rule        one of the two names above
  nesting     {depth, slow_side, lr_ratios (B) | periods (D)}; rate vectors listed FASTEST FIRST
  rule_kwargs D only: {surprise_weight, accumulate}
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


class LevelMonitor(SupervisedPlugin):
    """Per-task, per-level record of what the update rule actually did.

    updates/<lvl>           optimizer steps the level took during the task (B: every minibatch;
                            D: about 1/period of them; the remainder accumulated at the end of a
                            task is discarded by the rule's reset at the next task)
    displacement/<lvl>      ||θ_end - θ_start|| of the level's parameters over the task
    rel_displacement/<lvl>  the same divided by ||θ_start||
    D only (means over the task's minibatches): data_grad_norm, surprise, penalty_grad_norm,
    and penalty_ratio = mean penalty_grad_norm / mean data_grad_norm of the same level.
    BatchNorm running statistics are buffers, not parameters, so they are not in the displacement.
    """

    def __init__(self, learner, rule):
        super().__init__()
        self.levels, self.rule = learner.levels, rule
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
                            lr_ratios=n.get("lr_ratios"),
                            ordering=SLOW_SIDE_TO_ORDERING[n["slow_side"]]).to(device)
    rule = build_rule(cfg["rule"], learner, lr=cfg["lr"], momentum=cfg["momentum"],
                      weight_decay=cfg["weight_decay"], **cfg["rule_kwargs"])
    install_fresh_grad_clip(rule, cfg["grad_clip_norm"])
    monitor = LevelMonitor(learner, rule)
    kw = {k: v for k, v in strategy_kwargs.items() if k not in ("model", "optimizer", "criterion")}
    strategy = NestedStrategy(learner, rule, plugins=list(plugins) + [monitor],
                              **{k: v for k, v in kw.items() if k != "plugins"})
    # Effective lr and stepping period of each level, input side first (for the result file).
    level_info = []
    for lvl in learner.levels:
        lr = cfg["lr"] * lvl.lr_ratio if cfg["rule"] == "per_level_lr" else cfg["lr"]
        period = lvl.period if cfg["rule"] != "per_level_lr" else 1
        level_info.append({"level": lvl.idx, "blocks": lvl.name, "lr": lr, "period": period})
    return learner, rule, strategy, monitor, level_info


def checkpoint_module(rule):
    """The network whose weights are saved after each task: the deployed model's backbone, so
    the state_dict loads straight into MODEL_BUILDERS['reduced_resnet18_blocks']."""
    return rule.eval_model().backbone
