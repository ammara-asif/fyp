import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from avalanche.training.plugins import SupervisedPlugin


def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    hsic = (X.T @ Y).norm() ** 2
    return float(hsic / ((X.T @ X).norm() * (Y.T @ Y).norm() + 1e-12))


def probe_acc(ftr, ytr, fte, yte) -> float:
    sc = StandardScaler().fit(ftr)
    clf = LogisticRegression(max_iter=1000, C=1.0).fit(sc.transform(ftr), ytr)
    return float(clf.score(sc.transform(fte), yte))


class StructuralProbePlugin(SupervisedPlugin):
    """Per-level representation measurements.

    Gate 1 is decided on *probe forgetting*: for each task j and level l, the
    drop in linear-probe accuracy between the moment task j was learned and the
    end of the stream. Class-IL accuracy cannot decide it, because under plain
    sequential training every old task ends at exactly 0% in every arm.
    """

    def __init__(self, benchmark, out_dir, device, n_cka=512, n_probe_train=1000,
                 n_probe_test=500, n_grad=256, mb=256, use_wandb=False, seed=0):
        super().__init__()
        self.bm, self.out_dir, self.device = benchmark, out_dir, device
        self.n_probe_train, self.n_probe_test = n_probe_train, n_probe_test
        self.n_grad, self.mb = n_grad, mb
        self.use_wandb = use_wandb
        self.rng = np.random.RandomState(seed)
        os.makedirs(out_dir, exist_ok=True)
        self.cka_x, _ = self._sample(benchmark.test_stream[0].dataset, n_cka)
        self.prev_cka = None
        self.prev_batch = None
        self.rows: List[dict] = []
        # (level, task) -> probe accuracy measured right after that task was learned
        self.probe_at_learn: Dict[Tuple[int, int], float] = {}

    # ---------------- helpers
    def _sample(self, dataset, n):
        idx = self.rng.choice(len(dataset), min(n, len(dataset)), replace=False)
        xs, ys = [], []
        for b in DataLoader(Subset(dataset, idx), batch_size=self.mb):
            xs.append(b[0]); ys.append(b[1])
        return torch.cat(xs), torch.cat(ys)

    @torch.no_grad()
    def _feats(self, model, x):
        model.eval()
        out = None
        for i in range(0, len(x), self.mb):
            f = [h.cpu() for h in model.level_features(x[i:i + self.mb].to(self.device))]
            out = f if out is None else [torch.cat([a, b]) for a, b in zip(out, f)]
        return out

    def _level_grads(self, model, crit, x, y):
        model.eval()
        loss = crit(model(x.to(self.device)), y.to(self.device))
        res = []
        for lvl in model.levels:
            ps = list(lvl.parameters())
            g = torch.autograd.grad(loss, ps, retain_graph=True, allow_unused=True)
            res.append(torch.cat([gi.flatten() for gi in g if gi is not None]).detach())
        return res

    def _log(self, row):
        self.rows.append(row)
        if self.use_wandb:
            import wandb
            key = f"{row['metric']}/L{row['level']}"
            if "target_task" in row and row["target_task"] is not None:
                key += f"/task{row['target_task']}"
            wandb.log({key: row["value"], "exp": row["exp"]})

    def _flush(self):
        pd.DataFrame(self.rows).to_csv(os.path.join(self.out_dir, "structural.csv"),
                                       index=False)

    def probe_forgetting(self) -> Dict[str, float]:
        """Mean drop in probe accuracy per level, over every task but the last."""
        last = {}
        for r in self.rows:
            if r["metric"] == "probe_acc":
                last[(r["level"], r["target_task"])] = (r["value"], r["blocks"])
        drops: Dict[int, List[float]] = {}
        names: Dict[int, str] = {}
        for (lvl, task), (final, blocks) in last.items():
            if (lvl, task) not in self.probe_at_learn:
                continue
            learned = self.probe_at_learn[(lvl, task)]
            drops.setdefault(lvl, []).append(100.0 * (learned - final))
            names[lvl] = blocks
        out = {}
        for lvl, vals in sorted(drops.items()):
            out[f"probe_forgetting/L{lvl}"] = float(np.mean(vals))
            out[f"probe_forgetting_blocks/L{lvl}"] = names[lvl]
        return out

    # ---------------- hooks
    def before_training_exp(self, strategy, **kw):
        t = strategy.experience.current_experience
        if t == 0 or self.prev_batch is None:
            return
        # Measure on the model that is evaluated, so grad_cosine, CKA and probes
        # all describe the same network (the shadow, for the EMA arm).
        model, crit = strategy.rule.eval_model(), nn.CrossEntropyLoss()
        was_training = model.training
        xc, yc = self._sample(strategy.experience.dataset, self.n_grad)
        g_prev = self._level_grads(model, crit, *self.prev_batch)
        g_cur = self._level_grads(model, crit, xc, yc)
        for lvl, a, b in zip(model.levels, g_prev, g_cur):
            self._log(dict(exp=t, metric="grad_cosine", level=lvl.idx, blocks=lvl.name,
                           target_task=None,
                           value=float(F.cosine_similarity(a, b, dim=0))))
        model.train(was_training)
        strategy.model.train()

    def after_training_exp(self, strategy, **kw):
        t = strategy.experience.current_experience
        model = strategy.rule.eval_model()

        # CKA drift on the fixed probe set
        feats = self._feats(model, self.cka_x)
        if self.prev_cka is not None:
            for lvl, a, b in zip(model.levels, self.prev_cka, feats):
                self._log(dict(exp=t, metric="cka_vs_prev", level=lvl.idx,
                               blocks=lvl.name, target_task=None,
                               value=linear_cka(a, b)))
        self.prev_cka = feats

        # Linear probes on every task seen so far
        for j in range(t + 1):
            xtr, ytr = self._sample(self.bm.train_stream[j].dataset, self.n_probe_train)
            xte, yte = self._sample(self.bm.test_stream[j].dataset, self.n_probe_test)
            ftr, fte = self._feats(model, xtr), self._feats(model, xte)
            for lvl, a, b in zip(model.levels, ftr, fte):
                v = probe_acc(a.numpy(), ytr.numpy(), b.numpy(), yte.numpy())
                self._log(dict(exp=t, metric="probe_acc", level=lvl.idx,
                               blocks=lvl.name, target_task=j, value=v))
                if j == t:
                    self.probe_at_learn[(lvl.idx, j)] = v

        self.prev_batch = self._sample(strategy.experience.dataset, self.n_grad)
        self._flush()
        strategy.model.train()
