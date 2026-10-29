"""
ADNN: Autonomous Dynamic-Nesting Networks.

Deliberately imports nothing. Stage 1 depends on Avalanche and scikit-learn;
Stage 2 depends on neither. Importing submodules here would force every stage
to install every dependency. Import what you need directly, e.g.

    from src.nested.learner import NestedLearner
    from src.strategies.nested_strategy import NestedStrategy   # needs avalanche
"""
