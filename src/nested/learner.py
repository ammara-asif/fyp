from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Level:
    idx: int                 # position along depth (0 = input side)
    rate_idx: int            # index into the rate vectors (0 = fastest)
    block_names: List[str]
    modules: List[nn.Module]
    last_block_idx: int      # index into backbone.blocks() of the level's last block
    period: int = 1          # k_i for periodic stepping / nested objectives
    lr_ratio: float = 1.0    # for per-level LR
    ema_tau: float = 0.0     # for EMA coupling (0 = shadow tracks live exactly)

    def parameters(self):
        for m in self.modules:
            yield from m.parameters()

    @property
    def name(self):
        return "+".join(self.block_names)


class NestedLearner(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        depth: int = 3,
        periods: Optional[List[int]] = None,
        lr_ratios: Optional[List[float]] = None,
        ema_taus: Optional[List[float]] = None,
        ordering: str = "slow_deep",
    ):
        super().__init__()
        self.backbone = backbone
        self.depth = depth
        self.ordering = ordering

        periods = list(periods) if periods is not None else [1] * depth
        lr_ratios = list(lr_ratios) if lr_ratios is not None else [1.0] * depth
        # default EMA rate from period: k=1 -> tau=0, k=16 -> tau=0.9375
        ema_taus = list(ema_taus) if ema_taus is not None else [1.0 - 1.0 / k for k in periods]
        for v, n in [(periods, "periods"), (lr_ratios, "lr_ratios"), (ema_taus, "ema_taus")]:
            assert len(v) == depth, f"{n} must have length depth={depth}"

        blocks = backbone.blocks()
        chunks = np.array_split(np.arange(len(blocks)), depth)
        # chunk 0 is the input side. slow_deep -> input side gets the slowest rate.
        rate_order = list(reversed(range(depth))) if ordering == "slow_deep" else list(range(depth))

        self.levels: List[Level] = []
        for ci, chunk in enumerate(chunks):
            ri = rate_order[ci]
            self.levels.append(
                Level(
                    idx=ci,
                    rate_idx=ri,
                    block_names=[blocks[i][0] for i in chunk],
                    modules=[blocks[i][1] for i in chunk],
                    last_block_idx=int(chunk[-1]),
                    period=int(periods[ri]),
                    lr_ratio=float(lr_ratios[ri]),
                    ema_tau=float(ema_taus[ri]),
                )
            )

    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def level_features(self, x):
        """Pooled activation at the output of each level. Used by CKA / probe metrics."""
        outs = self.backbone.forward_blocks(x)
        feats = []
        for lvl in self.levels:
            h = outs[lvl.last_block_idx]
            if h.dim() == 4:
                h = h.mean(dim=(2, 3))
            feats.append(h)
        return feats

    def levels_fast_to_slow(self):
        return sorted(self.levels, key=lambda l: l.rate_idx)

    def describe(self):
        return [
            dict(level=l.idx, blocks=l.name, period=l.period, lr_ratio=l.lr_ratio, ema_tau=l.ema_tau)
            for l in self.levels
        ]