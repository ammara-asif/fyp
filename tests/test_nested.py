"""CPU tests for the nested arms (B, D, and Flat through the nested path).
No Drive, no wandb, no CIFAR download. Force CPU so equality checks are deterministic.

Run:  CUDA_VISIBLE_DEVICES= WANDB_MODE=disabled python -m pytest tests -q
"""
import os
import sys

os.environ["WANDB_MODE"] = "disabled"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest  # noqa: E402
import torch  # noqa: E402

from fyp_infra import run_id, run_sweep, setup_paths  # noqa: E402
import stage1_runner as sr  # noqa: E402

# synthetic: 2 experiences x 2 classes x 20 images, batch 8 -> 5 minibatches per epoch;
# 4 epochs -> 20 minibatches per task, so D's levels step 20 / 5 / 1 times (periods 1, 4, 16).
TINY = dict(benchmark="synthetic", n_experiences=2, epochs=4, nf=4, batch_size=8,
            num_workers=0, stage="test", lr=0.05)
LEVELS = ["stem+layer1", "layer2+layer3", "layer4+fc"]  # input side first


@pytest.fixture
def paths(tmp_path):
    return setup_paths(str(tmp_path))


def _run(cfg, paths):
    return run_sweep([cfg], sr.run_experiment, paths)[run_id(cfg)]["metrics"]


def test_old_run_ids_unchanged():
    # IDs printed by the executed control notebook (runs/stage1_control_2.ipynb).
    base = dict(stage="stage1_control", strategy="naive", epochs=10, seed=0,
                eval_split="test", save_checkpoints=True)
    assert run_id(sr.make_cfg(**base, lr=0.1, lr_schedule="flat")) == "4b2bd0082f5c"
    assert run_id(sr.make_cfg(**base, lr=0.6, lr_schedule="llrd", level_ratios=(1, 4, 16),
                              level_order="fast_deep")) == "9d5d9fcc4194"


def test_nested_cfg_normalisation():
    a = sr.arm_cfg("nested_d", **TINY)
    b = sr.arm_cfg("nested_d", **TINY, rule_kwargs=dict(surprise_weight=0.1))
    assert run_id(a) == run_id(b)  # writing out a default gives the same run
    assert run_id(a) != run_id(sr.arm_cfg("nested_d", **TINY, rule_kwargs=dict(surprise_weight=0.0)))
    assert "lr_ratios" not in a["nesting"]
    with pytest.raises(KeyError):  # a field the rule would ignore is an error, not hashed
        sr.arm_cfg("nested_b", **TINY, nesting=dict(periods=[1, 4, 16]))
    with pytest.raises(KeyError):
        sr.arm_cfg("nested_b", **TINY, nesting=dict(ordering="slow_deep"))
    assert sr.arm_label(sr.arm_cfg("flat", **TINY)) == "flat_nl"
    assert sr.arm_label(sr.arm_cfg("nested_b", **TINY)) == "nested_b"
    assert sr.arm_label(a) == "nested_d_lam0.1"


def test_block_resnet_init_matches_my_resnet():
    # If this holds, Flat via the new path starts from exactly the weights my old Flat runs used.
    torch.manual_seed(0)
    mine = sr.ReducedResNet18(100, 20).state_dict()
    torch.manual_seed(0)
    team = sr.BlockResNet18(100, 20).state_dict()
    assert len(mine) == len(team)
    for (ka, va), (kb, vb) in zip(mine.items(), team.items()):
        assert va.shape == vb.shape and torch.equal(va, vb), (ka, kb)


def test_flat_nested_path_equals_naive(paths):
    # Same model, same seed, plain SGD: the nested code path must reproduce Naive exactly.
    naive = sr.make_cfg(**TINY, model="reduced_resnet18_blocks", strategy="naive")
    flat = sr.arm_cfg("flat", **TINY)
    mn, mf = _run(naive, paths), _run(flat, paths)
    assert mn["acc_matrix"] == mf["acc_matrix"]
    assert mn["train_loss_per_exp"] == pytest.approx(mf["train_loss_per_exp"], rel=1e-6)
    assert mn["n_params"] == mf["n_params"] and mn["optimizer_steps"] == mf["optimizer_steps"]


def test_b_per_level_lrs(paths):
    m = _run(sr.arm_cfg("nested_b", **dict(TINY, lr=0.16, epochs=1)), paths)
    assert m["lr_per_group"] == pytest.approx({LEVELS[0]: 0.01, LEVELS[1]: 0.04, LEVELS[2]: 0.16})
    assert all(r[f"updates/{l}"] == 5 for r in m["level_log_per_exp"] for l in LEVELS)


def test_d_runs_steps_and_checkpoints(paths):
    cfg = sr.arm_cfg("nested_d", **TINY, save_checkpoints=True)
    m = _run(cfg, paths)
    assert m["optimizer_steps"] == 40  # minibatch backward passes, same as every other arm
    for row in m["level_log_per_exp"]:
        assert [row[f"updates/{l}"] for l in LEVELS] == [1, 5, 20]
        assert row[f"penalty_ratio/{LEVELS[0]}"] > 0 and row[f"penalty_ratio/{LEVELS[1]}"] > 0
        assert f"penalty_ratio/{LEVELS[2]}" not in row  # fastest level: data gradient only
    # checkpoint loads into the plain model the analysis code builds
    model = sr.MODEL_BUILDERS[cfg["model"]](cfg, 4)
    sd = torch.load(os.path.join(m["checkpoint_dir"], "after_exp01.pt"))
    model.load_state_dict(sd)
    groups, head = model.depth_groups()
    assert sum(p.numel() for g in groups for mod in g for p in mod.parameters()) \
        + sum(p.numel() for p in head.parameters()) == m["n_params"]


def test_fresh_grad_clip_does_not_reclip_accumulator():
    import nested_builder as nb
    from src.nested.learner import NestedLearner
    from src.nested.update_rules import build_rule

    torch.manual_seed(0)
    learner = NestedLearner(sr.BlockResNet18(4, 4), depth=3, periods=[1, 4, 16]).train()
    rule = build_rule("nested_objectives", learner, lr=0.1, momentum=0.0, weight_decay=0.0)
    nb.install_fresh_grad_clip(rule, 1e-3)
    x, y = torch.randn(8, 3, 32, 32), torch.randint(0, 4, (8,))
    params = list(learner.parameters())

    def total_norm():
        return torch.stack([p.grad.norm() for p in params if p.grad is not None]).norm().item()

    rule.backward(torch.nn.functional.cross_entropy(learner(x), y), None)
    first = [p.grad.clone() if p.grad is not None else None for p in params]
    assert total_norm() == pytest.approx(1e-3, rel=1e-4)  # fresh gradient clipped
    # same batch, no step in between -> same fresh gradient; the buffer must hold 2x, not be clipped to 1e-3
    rule.backward(torch.nn.functional.cross_entropy(learner(x), y), None)
    assert total_norm() == pytest.approx(2e-3, rel=1e-3)
    for p, g in zip(params, first):
        if g is not None:
            assert torch.allclose(p.grad, 2 * g, rtol=1e-3, atol=1e-9)
