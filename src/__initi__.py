# src/__init__.py
"""
ADNN: Autonomous Dynamic-Nesting Networks
Imports main classes to make them available as src.models, src.nested, etc.
"""

# Models
from src.models.resnet import ReducedResNet18

# Nested Components
from src.nested.learner import NestedLearner
from src.nested.update_rules import build_rule, RULES

# Strategies
from src.strategies.nested_strategy import NestedStrategy

# Metrics
from src.metrics.structural import StructuralProbePlugin

# Utils
from src.utils import seed_everything, count_params

__all__ = [
    "ReducedResNet18",
    "NestedLearner",
    "build_rule",
    "RULES",
    "NestedStrategy",
    "StructuralProbePlugin",
    "seed_everything",
    "count_params",
]