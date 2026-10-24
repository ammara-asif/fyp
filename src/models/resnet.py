import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ReducedResNet18(nn.Module):
    """ResNet-18 with nf=20 base width (Mammoth 'resnet18 reduced'), 32x32 input."""

    def __init__(self, num_classes=100, nf=20):
        super().__init__()
        self.in_planes = nf
        self.stem = nn.Sequential(
            nn.Conv2d(3, nf, 3, 1, 1, bias=False), nn.BatchNorm2d(nf), nn.ReLU(inplace=True)
        )
        self.layer1 = self._make_layer(nf, 2, 1)
        self.layer2 = self._make_layer(nf * 2, 2, 2)
        self.layer3 = self._make_layer(nf * 4, 2, 2)
        self.layer4 = self._make_layer(nf * 8, 2, 2)
        self.fc = nn.Linear(nf * 8, num_classes)

    def _make_layer(self, planes, n, stride):
        layers = []
        for s in [stride] + [1] * (n - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def blocks(self):
        """Depth-ordered (input side first). This is what NestedLearner partitions."""
        return [
            ("stem", self.stem),
            ("layer1", self.layer1),
            ("layer2", self.layer2),
            ("layer3", self.layer3),
            ("layer4", self.layer4),
            ("fc", self.fc),
        ]

    def forward_blocks(self, x):
        outs = []
        for name, m in self.blocks():
            if name == "fc":
                x = F.adaptive_avg_pool2d(x, 1).flatten(1)
            x = m(x)
            outs.append(x)
        return outs

    def forward(self, x):
        return self.forward_blocks(x)[-1]