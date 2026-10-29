from dataclasses import dataclass
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
    feature_block_idx: int   # block whose output is probed (never the classifier)
    period: int = 1          # k_i for periodic stepping / nested objectives
    lr_ratio: float = 1.0    # for per-level LR
    ema_tau: float = 0.0     # for EMA coupling (0 = shadow tracks live exactly)

    def parameters(self):
        for m in self.modules:
            yield from m.parameters()

    @property
    def name(self):
        return "+".join(self.block_names)

    @property
    def feature_name(self):
        return self._feature_name

    def _set_feature_name(self, n):
        self._feature_name = n


class NestedLearner(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        depth: int = 3,
        periods: Optional[List[int]] = None,
        lr_ratios: Optional[List[float]] = None,
        ema_taus: Optional[List[float]] = None,
        ordering: str = "slow_deep",
        head_block: str = "fc",
        level_blocks: Optional[List[List[int]]] = None,
    ):
        """level_blocks: explicit block indices per level, input side first, e.g.
        [[0, 1, 2, 3], [4], [5]] = (stem..layer3 | layer4 | fc). Default: blocks
        split into `depth` equal-count chunks."""
        super().__init__()
        self.backbone = backbone
        self.depth = depth
        self.ordering = ordering
        self.head_block = head_block

        periods = list(periods) if periods is not None else [1] * depth
        lr_ratios = list(lr_ratios) if lr_ratios is not None else [1.0] * depth
        # default EMA rate from period: k=1 -> tau=0, k=16 -> tau=0.9375
        ema_taus = list(ema_taus) if ema_taus is not None else [1.0 - 1.0 / k for k in periods]
        for v, n in [(periods, "periods"), (lr_ratios, "lr_ratios"), (ema_taus, "ema_taus")]:
            assert len(v) == depth, f"{n} must have length depth={depth}"

        blocks = backbone.blocks()
        names = [b[0] for b in blocks]
        # Last block that is a representation rather than the classifier.
        last_repr = max(i for i, n in enumerate(names) if n != head_block)

        if level_blocks is not None:
            assert len(level_blocks) == depth, "level_blocks must have one entry per level"
            flat = [i for grp in level_blocks for i in grp]
            assert sorted(flat) == list(range(len(blocks))), \
                f"level_blocks must cover blocks 0..{len(blocks) - 1} exactly once"
            chunks = [np.array(grp) for grp in level_blocks]
        else:
            chunks = np.array_split(np.arange(len(blocks)), depth)
        # chunk 0 is the input side. slow_deep -> input side gets the slowest rate.
        rate_order = list(reversed(range(depth))) if ordering == "slow_deep" else list(range(depth))

        self.levels: List[Level] = []
        for ci, chunk in enumerate(chunks):
            ri = rate_order[ci]
            last = int(chunk[-1])
            # Probing the classifier's own output is near-tautological, so a level
            # ending in the head is probed at its last representation block instead.
            feat = last
            while feat >= 0 and names[feat] == head_block:
                feat -= 1
            if feat < 0:
                feat = last_repr
            lvl = Level(
                idx=ci,
                rate_idx=ri,
                block_names=[names[i] for i in chunk],
                modules=[blocks[i][1] for i in chunk],
                last_block_idx=last,
                feature_block_idx=feat,
                period=int(periods[ri]),
                lr_ratio=float(lr_ratios[ri]),
                ema_tau=float(ema_taus[ri]),
            )
            lvl._set_feature_name(names[feat])
            self.levels.append(lvl)

    def forward(self, x):
        return self.backbone(x)

    @torch.no_grad()
    def level_features(self, x):
        """Pooled activation at each level's last representation block.

        Never the classifier's logits: probing those measures the head, not the
        representation, and makes the deepest level's probe uninterpretable.
        """
        outs = self.backbone.forward_blocks(x)
        feats = []
        for lvl in self.levels:
            h = outs[lvl.feature_block_idx]
            if h.dim() == 4:
                h = h.mean(dim=(2, 3))
            feats.append(h)
        return feats

    def levels_fast_to_slow(self):
        return sorted(self.levels, key=lambda l: l.rate_idx)

    def describe(self):
        return [
            dict(level=l.idx, blocks=l.name, probe_at=l.feature_name,
                 period=l.period, lr_ratio=l.lr_ratio, ema_tau=l.ema_tau)
            for l in self.levels
        ]
