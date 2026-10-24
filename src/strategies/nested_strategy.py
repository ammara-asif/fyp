import torch.nn as nn
from avalanche.training.templates import SupervisedTemplate


class NestedStrategy(SupervisedTemplate):
    """Avalanche template that delegates the update to an UpdateRule.
    Sequential fine-tuning: no replay, no regularisation, no freezing."""

    def __init__(self, learner, rule, criterion=None, reset_rule_per_exp=True, **kwargs):
        self.rule = rule
        self._live_model = learner
        self.reset_rule_per_exp = reset_rule_per_exp
        super().__init__(
            model=learner,
            optimizer=rule.primary_optimizer,
            criterion=criterion or nn.CrossEntropyLoss(),
            **kwargs,
        )

    # Avalanche wants to rebuild the optimizer per experience; the rule owns that.
    def make_optimizer(self, **kwargs):
        if self.reset_rule_per_exp:
            self.rule.reset()

    def training_epoch(self, **kwargs):
        for self.mbatch in self.dataloader:
            if self._stop_training:
                break
            self._unpack_minibatch()
            self._before_training_iteration(**kwargs)

            self.rule.zero_grad()
            self.loss = 0
            self._before_forward(**kwargs)
            self.mb_output = self.forward()
            self._after_forward(**kwargs)
            self.loss = self.loss + self.rule.compute_loss(self)

            self._before_backward(**kwargs)
            self.rule.backward(self.loss, self)
            self._after_backward(**kwargs)

            self._before_update(**kwargs)
            self.rule.step()
            self._after_update(**kwargs)
            self._after_training_iteration(**kwargs)

    # Evaluate whatever the rule says is the deployed model (shadow for EMA).
    def _before_eval(self, **kwargs):
        self.model = self.rule.eval_model()
        super()._before_eval(**kwargs)

    def _after_eval(self, **kwargs):
        super()._after_eval(**kwargs)
        self.model = self._live_model